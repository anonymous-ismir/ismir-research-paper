"""
Add YT coverage + sync scores to audioldm2 / tango2 result CSVs.

For each row: send the WAV at ``audio_wav_path`` to Gemini for per-clip cue
segmentation (JSON), map cues to ``AudioCue``, then call ``AudioEvaluator``
``yt_coverage_score`` and ``yt_sync_score`` (same logic as ``evaluator.py``).
Appends ``yt_gemini_cues_json`` with the raw ``cues`` list Gemini returned (JSON object).

After each row is updated, the CSV is rewritten atomically so you can stop and resume:
rows with missing scores or missing ``yt_gemini_cues_json`` are finished on the next run;
cached cues are reused (no second Gemini call) unless you pass ``--force``.

Gemini ``generate_content`` is retried on transient HTTP errors (429/502/503/504, etc.)
with exponential backoff. Each processed row also writes a JSON backup under
``<csv_dir>/<csv_stem>_yt_per_prompt/`` (or ``--prompt-backup-dir/<csv_stem>/``).

Requires: GEMINI_API_KEY or GOOGLE_API_KEY, backend deps (torch, laion_clap, ...).
Run from repo: ``python -m Evaluation.yt_scores_tango_audioldm`` with cwd ``backend``.
"""

import sys
import os
import torch
# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)       
print("Project root added to sys.path:", project_root)
# from __future__ import annotations
from Variable.dataclases import AudioCue
from dotenv import load_dotenv

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.abspath(os.path.join(_EVAL_DIR, ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)


load_dotenv(os.path.join(_BACKEND_ROOT, ".env"))


logger = logging.getLogger(__name__)

GEMINI_CLIP_PROMPT = """
You are an expert movie trailer analyst and sound designer.
You are given a short audio CLIP from a movie trailer

Your job is to (for THIS clip only, ignoring anything before/after it in the full trailer):

. Identify as many distinct audio sources as you can (music, ambience, SFX, voices, etc.). try to look for narration, any background music (describe the music as much as you can), any envirmental sounds explosion etc and try to identify as much sounds as you can with approximately when they start, and are there upto how much duration with relative loudness: like say narrator is speaking at -10 db

3. For each audio source, estimate:
   - a short audio_class label (e.g., "epic orchestral music with percussive hits",  "whoosh sound effect", "dramatic impact sound effect", "orchestral build-up with strings", "metallic clang / sword sound", "deep male narrator", "heavy footsteps / stomp", "rhythmic percussion with synth bass")
   - starting_time: when this sound begins within this clip, in seconds, relative to the
     START of the clip (0 is the first frame of this clip)
   - duration: how long the sound lasts, in seconds (relative within this clip) try to be precise and exact upto seconds level.
   - weight_db: a relative loudness weight in dB (negative values, e.g. -5 is very loud,
     -25 is background, -40 or less is very faint)

IMPORTANT:
- ONLY consider what happens inside this clip.
- If you are uncertain, make your best guess but stay plausible for a modern movie trailer.
- Generate as many cues as you can feel there are in the clip.

Return STRICTLY a single JSON object in the following form:
{{
 
  "cues": [
    {{
      "audio_class": "short label here",
      "starting_time": 0,
      "duration": 4,
      "weight_db": -10.5
    }}
  ]
}}

Do NOT wrap the JSON in markdown. Do NOT add explanations.
""".strip()


def _exc_csv(exc: BaseException, max_len: int = 2000) -> str:
    import traceback

    parts = "".join(traceback.format_exception_only(
        type(exc), exc)).strip().replace("\n", " | ")
    if len(parts) > max_len:
        return parts[: max_len - 3] + "..."
    return parts


def _parse_json_from_model_text(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        return None
    json_str = response_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", json_str)
    if match:
        json_str = match.group()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


_GEMINI_TRANSIENT_HTTP_CODES = frozenset({408, 429, 502, 503, 504})


def _is_transient_gemini_error(exc: BaseException) -> bool:
    """True for rate limits / overload / temporary upstream failures."""
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _GEMINI_TRANSIENT_HTTP_CODES:
        return True
    # google.genai.errors.ServerError / ClientError
    mod = type(exc).__module__
    if mod.startswith("google.genai") and code is not None:
        try:
            c = int(code)
        except (TypeError, ValueError):
            return False
        return c in _GEMINI_TRANSIENT_HTTP_CODES
    return False


def _generate_content_with_retries(
    client: Any,
    *,
    model_name: str,
    contents: List[Any],
    max_retries: int,
    retry_base_sec: float,
) -> Any:
    """Call ``generate_content`` with extra backoff beyond SDK defaults (503 spikes)."""
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
            )
        except BaseException as e:
            last_exc = e
            if attempt >= max_retries or not _is_transient_gemini_error(e):
                raise
            delay = retry_base_sec * (2**attempt) + random.uniform(0.25, 1.25)
            logger.warning(
                "[YT-GEMINI] transient error (%s); retry %d/%d in %.1fs",
                _exc_csv(e, max_len=400),
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _wait_upload_active(client: Any, f: Any, timeout_s: float = 120.0) -> Any:
    """Poll file state until ACTIVE (Files API)."""
    deadline = time.time() + timeout_s
    name = getattr(f, "name", None) or getattr(f, "uri", None)
    if not name:
        return f
    while True:
        state = getattr(f, "state", None)
        state_name = getattr(
            state, "name", None) if state is not None else None
        if state_name is None and state is not None:
            state_name = str(state)
        if state_name in (None, "ACTIVE", "STATE_UNSPECIFIED"):
            return f
        if state_name == "FAILED":
            raise RuntimeError(f"Uploaded file failed processing: {f}")
        if time.time() > deadline:
            raise TimeoutError(
                f"File {name} not ACTIVE after {timeout_s}s (last state={state_name})")
        time.sleep(1.5)
        f = client.files.get(name=name)


def gemini_cues_from_wav(
    wav_path: str,
    model_name: str,
    *,
    inline_max_bytes: int = 18 * 1024 * 1024,
    max_retries: int = 10,
    retry_base_sec: float = 3.0,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Returns (list of cue dicts from model JSON, error_message).
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "Missing GEMINI_API_KEY or GOOGLE_API_KEY"

    if not os.path.isfile(wav_path):
        return None, f"Audio file not found: {wav_path}"

    try:
        import google.genai as genai
        from google.genai import types
    except ImportError as e:
        return None, f"google.genai not installed: {e}"

    client = genai.Client(api_key=api_key)
    size = os.path.getsize(wav_path)
    audio_part: Any
    upload_name: Optional[str] = None

    if size <= inline_max_bytes:
        with open(wav_path, "rb") as fh:
            data = fh.read()
        audio_part = types.Part.from_bytes(data=data, mime_type="audio/wav")
        logger.info(
            "[YT-GEMINI] Using inline audio (%d bytes): %s", size, wav_path)
    else:
        logger.info("[YT-GEMINI] Uploading audio (%d bytes): %s",
                    size, wav_path)
        up = client.files.upload(file=wav_path)
        up = _wait_upload_active(client, up)
        upload_name = getattr(up, "name", None)
        audio_part = up

    try:
        response = _generate_content_with_retries(
            client,
            model_name=model_name,
            contents=[GEMINI_CLIP_PROMPT, audio_part],
            max_retries=max_retries,
            retry_base_sec=retry_base_sec,
        )
    finally:
        if upload_name:
            try:
                client.files.delete(name=upload_name)
            except Exception as del_e:
                logger.warning(
                    "[YT-GEMINI] Could not delete uploaded file: %s", del_e)

    text = getattr(response, "text", None)
    if not text:
        return None, "Empty Gemini response text"

    parsed = _parse_json_from_model_text(text)
    if not parsed:
        return None, f"Could not parse JSON from model output (first 500 chars): {text[:500]!r}"

    cues = parsed.get("cues")
    if not isinstance(cues, list):
        return None, f"Parsed JSON missing 'cues' list: keys={list(parsed.keys())}"

    out: List[Dict[str, Any]] = []
    for i, c in enumerate(cues):
        if not isinstance(c, dict):
            continue
        out.append(c)
    logger.info("[YT-GEMINI] Parsed %d raw cue entries from model", len(out))
    logger.info("[YT-GEMINI] Cues: %s", out)
    return out, None


def gemini_dicts_to_audio_cues(cue_dicts: List[Dict[str, Any]]) -> List[AudioCue]:
    cues: List[AudioCue] = []
    for i, d in enumerate(cue_dicts):
        label = str(d.get("audio_class") or "").strip() or "unknown sound"
        try:
            start_s = float(d.get("starting_time", 0.0))
        except (TypeError, ValueError):
            start_s = 0.0
        try:
            dur_s = float(d.get("duration", 0.0))
        except (TypeError, ValueError):
            dur_s = 0.0
        if dur_s <= 0:
            dur_s = 0.1
        try:
            w_db = float(d["weight_db"])
        except (KeyError, TypeError, ValueError):
            w_db = -12.0

        cues.append(
            AudioCue(
                id=i,
                audio_type="SFX",
                audio_class=label,
                start_time_ms=int(round(start_s * 1000.0)),
                duration_ms=max(1, int(round(dur_s * 1000.0))),
                weight_db=w_db,
                fade_ms=500,
            )
        )
    return cues


def _score_to_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, dict) and "error" in val:
        return ""
    if isinstance(val, (int, float)):
        return repr(val)
    return str(val)


def _error_from_scores(cov: Any, sync: Any) -> str:
    errs = []
    if isinstance(cov, dict) and cov.get("error"):
        errs.append(f"coverage: {cov.get('error')}")
    if isinstance(sync, dict) and sync.get("error"):
        errs.append(f"sync: {sync.get('error')}")
    return " | ".join(errs)


def _cue_dicts_to_csv_json(cue_dicts: Optional[List[Dict[str, Any]]]) -> str:
    """Single CSV cell: compact JSON object {\"cues\": [...]} (Gemini-shaped)."""
    if not cue_dicts:
        return ""
    try:
        return json.dumps(
            {"cues": cue_dicts},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return ""


def _parse_stored_cue_dicts(cell: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Load cue dicts from ``yt_gemini_cues_json`` cell; None if missing/invalid."""
    if not cell or not str(cell).strip():
        return None
    try:
        obj = json.loads(str(cell).strip())
        if not isinstance(obj, dict):
            return None
        cues = obj.get("cues")
        if not isinstance(cues, list):
            return None
        return [c for c in cues if isinstance(c, dict)]
    except json.JSONDecodeError:
        return None


def _row_fully_done(row: Dict[str, Any]) -> bool:
    """
    Row does not need more work: cues on disk and either both numeric scores
    or a finished YT semantic error from the evaluator (empty score cells).
    """
    if (row.get("yt_gemini_cues_json") or "").strip() == "":
        return False
    sc = row.get("yt_coverage_score")
    ss = row.get("yt_sync_score")
    if sc not in (None, "") and ss not in (None, ""):
        return True
    err = (row.get("yt_gemini_eval_error") or "").strip()
    if ("coverage:" in err or "sync:" in err) and not err.startswith("evaluator:"):
        return True
    return False


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    dname = os.path.dirname(path)
    if dname:
        os.makedirs(dname, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def _per_prompt_backup_filename(row_idx: int, wav_path: str) -> str:
    stem = os.path.splitext(os.path.basename(wav_path))[0] or "prompt"
    stem = re.sub(r"[^\w\-.]", "_", stem)[:120]
    return f"row_{row_idx:05d}_{stem}.json"


def _save_per_prompt_result_backup(
    *,
    backup_dir: str,
    csv_path: str,
    row_idx: int,
    row: Dict[str, Any],
    gemini_model: str,
    cue_dicts: Optional[List[Dict[str, Any]]],
    from_cache: bool,
    dry_run: bool,
    stage: str,
) -> None:
    """
    Write one JSON file per processed row and log/print a one-line summary.
    ``stage`` is a short tag: gemini_error, dry_run, eval_error, ok.
    """
    wav = (row.get("audio_wav_path") or "").strip()
    fname = _per_prompt_backup_filename(row_idx, wav or "missing.wav")
    out_path = os.path.join(backup_dir, fname)
    payload: Dict[str, Any] = {
        "stage": stage,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": os.path.abspath(csv_path),
        "row_index": row_idx,
        "gemini_model": gemini_model,
        "dry_run": dry_run,
        "from_cached_cues": from_cache,
        "story_prompt": row.get("story_prompt") or "",
        "audio_wav_path": wav,
        "total_seconds": row.get("total_seconds"),
        "cues": cue_dicts if cue_dicts is not None else [],
        "yt_coverage_score": row.get("yt_coverage_score"),
        "yt_sync_score": row.get("yt_sync_score"),
        "yt_gemini_eval_error": row.get("yt_gemini_eval_error") or "",
        "yt_gemini_cues_json_cell": row.get("yt_gemini_cues_json") or "",
    }
    _atomic_write_json(out_path, payload)
    cov = row.get("yt_coverage_score")
    sync = row.get("yt_sync_score")
    err = (row.get("yt_gemini_eval_error") or "")[:240]
    summary = (
        f"[YT-PROMPT-BACKUP] stage={stage} row={row_idx} cues={len(cue_dicts or [])} "
        f"cov={cov!r} sync={sync!r} file={out_path}"
    )
    if err:
        summary += f" err={err!r}"
    print(summary)
    logger.info(summary)


def _write_results_csv_atomic(
    csv_path: str, fieldnames: List[str], rows: List[Dict[str, Any]]
) -> None:
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp_path, csv_path)


def process_results_csv(
    csv_path: str,
    evaluator: Any,
    *,
    gemini_model: str,
    dry_run: bool,
    force: bool,
    gemini_max_retries: int,
    gemini_retry_base_sec: float,
    prompt_backup_dir: Optional[str],
) -> None:
    new_cols = [
        "yt_coverage_score",
        "yt_sync_score",
        "yt_gemini_cues_json",
        "yt_gemini_eval_error",
    ]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
        for c in new_cols:
            if c not in fieldnames:
                fieldnames.append(c)

    csv_abs = os.path.abspath(csv_path)
    stem = os.path.splitext(os.path.basename(csv_abs))[0]
    if prompt_backup_dir:
        backup_dir = os.path.abspath(
            os.path.join(prompt_backup_dir, stem)
        )
    else:
        backup_dir = os.path.join(
            os.path.dirname(csv_abs), f"{stem}_yt_per_prompt"
        )
        backup_dir = os.path.abspath(backup_dir)
    logger.info("[YT-SCORES] Per-prompt JSON backups -> %s", backup_dir)

    if not dry_run:
        old_cols = set(reader.fieldnames or [])
        if any(c not in old_cols for c in new_cols):
            _write_results_csv_atomic(csv_path, fieldnames, rows)
            logger.info(
                "[YT-SCORES] Wrote header with new columns to %s", csv_path
            )

    updated = 0
    processed = 0
    for idx, row in enumerate(rows):
        story = (row.get("story_prompt") or "").strip()
        wav = (row.get("audio_wav_path") or "").strip()
        if not story or not wav:
            logger.warning(
                "[YT-SCORES] row=%d skip: missing story_prompt or audio_wav_path",
                idx,
            )
            continue

        if not force and _row_fully_done(row):
            logger.info("[YT-SCORES] row=%d skip: cues + scores already present", idx)
            continue

        logger.info(
            "[YT-SCORES] row=%d story_len=%d wav=%s", idx, len(story), wav
        )
        processed += 1

        try:
            total_sec = float(row.get("total_seconds") or 16.0)
        except (TypeError, ValueError):
            total_sec = 16.0

        gemini_err: Optional[str] = None
        cue_dicts: Optional[List[Dict[str, Any]]] = None
        from_cache = False

        if not force:
            cached = _parse_stored_cue_dicts(row.get("yt_gemini_cues_json"))
            if cached is not None and len(cached) > 0:
                cue_dicts = cached
                from_cache = True
                logger.info(
                    "[YT-SCORES] row=%d resume: using stored yt_gemini_cues_json (%d cues)",
                    idx,
                    len(cached),
                )

        if cue_dicts is None:
            try:
                cue_dicts, gemini_err = gemini_cues_from_wav(
                    wav,
                    gemini_model,
                    max_retries=gemini_max_retries,
                    retry_base_sec=gemini_retry_base_sec,
                )
            except Exception as e:
                gemini_err = _exc_csv(e)
                logger.exception("[YT-SCORES] row=%d Gemini call failed", idx)

            if gemini_err:
                row["yt_gemini_eval_error"] = gemini_err
                row["yt_coverage_score"] = ""
                row["yt_sync_score"] = ""
                row["yt_gemini_cues_json"] = ""
                updated += 1
                logger.error("[YT-SCORES] row=%d %s", idx, gemini_err)
                _save_per_prompt_result_backup(
                    backup_dir=backup_dir,
                    csv_path=csv_path,
                    row_idx=idx,
                    row=row,
                    gemini_model=gemini_model,
                    cue_dicts=None,
                    from_cache=from_cache,
                    dry_run=dry_run,
                    stage="gemini_error",
                )
                if not dry_run:
                    _write_results_csv_atomic(csv_path, fieldnames, rows)
                    logger.info(
                        "[YT-SCORES] checkpoint saved %s after row=%d",
                        csv_path,
                        idx,
                    )
                continue

        cues_json_cell = _cue_dicts_to_csv_json(cue_dicts)
        audio_cues = gemini_dicts_to_audio_cues(cue_dicts or [])
        src = "cache" if from_cache else "Gemini"
        logger.info(
            "[YT-SCORES] row=%d %s -> %d AudioCues; total_duration_sec=%.4f sample=%s",
            idx,
            src,
            len(audio_cues),
            total_sec,
            [c.audio_class for c in audio_cues[:3]],
        )

        if dry_run:
            row["yt_gemini_cues_json"] = cues_json_cell
            logger.info(
                "[YT-SCORES] row=%d dry-run: skip evaluator + CSV cells", idx
            )
            _save_per_prompt_result_backup(
                backup_dir=backup_dir,
                csv_path=csv_path,
                row_idx=idx,
                row=row,
                gemini_model=gemini_model,
                cue_dicts=cue_dicts,
                from_cache=from_cache,
                dry_run=True,
                stage="dry_run",
            )
            continue

        row["yt_gemini_cues_json"] = cues_json_cell

        try:
            cov = evaluator.yt_coverage_score(story, audio_cues)
            sync = evaluator.yt_sync_score(
                story, audio_cues, total_duration_sec=total_sec
            )
        except Exception as e:
            row["yt_gemini_eval_error"] = f"evaluator: {_exc_csv(e)}"
            row["yt_coverage_score"] = ""
            row["yt_sync_score"] = ""
            updated += 1
            logger.exception("[YT-SCORES] row=%d evaluator failed", idx)
            _save_per_prompt_result_backup(
                backup_dir=backup_dir,
                csv_path=csv_path,
                row_idx=idx,
                row=row,
                gemini_model=gemini_model,
                cue_dicts=cue_dicts,
                from_cache=from_cache,
                dry_run=False,
                stage="eval_error",
            )
            _write_results_csv_atomic(csv_path, fieldnames, rows)
            logger.info(
                "[YT-SCORES] checkpoint saved %s after row=%d", csv_path, idx
            )
            continue

        row["yt_coverage_score"] = _score_to_cell(cov)
        row["yt_sync_score"] = _score_to_cell(sync)
        eval_err = _error_from_scores(cov, sync)
        row["yt_gemini_eval_error"] = eval_err or gemini_err or ""

        logger.info(
            "[YT-SCORES] row=%d yt_coverage_score=%s yt_sync_score=%s eval_err=%r",
            idx,
            row["yt_coverage_score"],
            row["yt_sync_score"],
            row["yt_gemini_eval_error"],
        )
        updated += 1
        _save_per_prompt_result_backup(
            backup_dir=backup_dir,
            csv_path=csv_path,
            row_idx=idx,
            row=row,
            gemini_model=gemini_model,
            cue_dicts=cue_dicts,
            from_cache=from_cache,
            dry_run=False,
            stage="ok",
        )
        _write_results_csv_atomic(csv_path, fieldnames, rows)
        logger.info(
            "[YT-SCORES] checkpoint saved %s after row=%d", csv_path, idx
        )

    if dry_run:
        logger.info(
            "[YT-SCORES] dry-run: no CSV write; rows_processed=%d", processed
        )
        return

    logger.info("[YT-SCORES] Finished %s (%d row updates this run)", csv_path, updated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YT coverage/sync from Gemini-labeled Tango2 / AudioLDM2 WAVs.")
    default_results = os.path.join(_EVAL_DIR, "Results")
    parser.add_argument(
        "--audioldm-csv",
        default=os.path.join(default_results, "Results_from_server/audioldm2_results.csv"),
        help="Path to audioldm2_results.csv",
    )
    parser.add_argument(
        "--tango-csv",
        default=os.path.join(default_results, "Results_from_server/tango2_results.csv"),
        help="Path to tango2_results.csv",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get(
            "GEMINI_AUDIO_ANALYSIS_MODEL", "gemini-3-flash-preview"),
        help="Gemini model id for audio understanding (override with GEMINI_AUDIO_ANALYSIS_MODEL)",
    )
    parser.add_argument(
        "--only",
        choices=("both", "audioldm", "tango"),
        default="both",
        help="Which CSV to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call Gemini only; skip CLAP evaluator and do not write CSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every row: ignore resume skip and cached yt_gemini_cues_json (calls Gemini again)",
    )
    parser.add_argument(
        "--gemini-max-retries",
        type=int,
        default=10,
        help="Extra retries for transient Gemini errors (HTTP 408/429/502/503/504) with exponential backoff",
    )
    parser.add_argument(
        "--gemini-retry-base-sec",
        type=float,
        default=3.0,
        help="Initial backoff seconds for Gemini retries (doubles each attempt, plus jitter)",
    )
    parser.add_argument(
        "--prompt-backup-dir",
        default=None,
        help=(
            "Optional base directory for per-row JSON backups; files go under "
            "<this_dir>/<csv_stem>/. Default: <csv_dir>/<csv_stem>_yt_per_prompt/"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    targets: List[str] = []
    if args.only in ("both", "audioldm"):
        targets.append(os.path.abspath(args.audioldm_csv))
    if args.only in ("both", "tango"):
        targets.append(os.path.abspath(args.tango_csv))

    evaluator = None
    if not args.dry_run:
        logger.info(
            "[YT-SCORES] Initializing AudioEvaluator (CLAP + YT embeddings)...")
        from Evaluation.evaluator import AudioEvaluator

        evaluator = AudioEvaluator()

    for path in targets:
        if not os.path.isfile(path):
            logger.error("[YT-SCORES] Missing CSV: %s", path)
            continue
        logger.info("[YT-SCORES] Processing %s", path)
        process_results_csv(
            path,
            evaluator,
            gemini_model=args.gemini_model,
            dry_run=args.dry_run,
            force=args.force,
            gemini_max_retries=args.gemini_max_retries,
            gemini_retry_base_sec=args.gemini_retry_base_sec,
            prompt_backup_dir=args.prompt_backup_dir,
        )


if __name__ == "__main__":
    main()
