"""
Baseline evaluation: generate audio from story prompts with Tango2 / AudioLDM2 only,
then run the same non–YouTube metrics as `evaluation.py` (no yt_coverage / yt_sync).

Reads prompts from `Results/final_results.csv`, batches generation, appends rows to:
  Results/tango2_results.csv
  Results/audioldm2_results.csv

WAV exports (per prompt, hashed filename):
  Results/infernce_results/tango2_audios/
  Results/infernce_results/audioldm_audios/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from pydub import AudioSegment

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.abspath(os.path.join(_EVAL_DIR, ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from Variable.configurations import SFX_RATE, AUDIO_LDM2, TANGO2  # noqa: E402
from helper.audio_conversions import audio_to_base64  # noqa: E402
from helper.lib import init_models  # noqa: E402
from model.audioLDM_model import AudioLDM2Model  # noqa: E402
from model.tango2_model import Tango2Model  # noqa: E402

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

RESULT_FIELDNAMES = [
    "story_prompt",
    "total_seconds",
    "clap_score",
    "audio_richness_spectral_flatness",
    "audio_richness_spectral_entropy",
    "noise_floor_db",
    "audio_onsets",
    "cinematic_dynamic_range_db",
    "cinematic_crest_factor",
    "cinematic_spectral_flatness",
    "cinematic_spectral_entropy",
    "cinematic_spectral_centroid_hz",
    "spectral_kl_divergence",
    "fad_score",
    "audio_wav_path",
    "audio_export_error",
    "error",
]


def _exc_for_csv(exc: BaseException, *, max_len: int = 2000) -> str:
    parts = "".join(traceback.format_exception_only(type(exc), exc)).strip().replace("\n", " | ")
    if len(parts) > max_len:
        return parts[: max_len - 3] + "..."
    return parts


def _run_stage(
    label: str,
    fn: Callable[[], _T],
) -> Tuple[Optional[_T], str]:
    try:
        return fn(), ""
    except Exception as e:
        logger.exception("[BASELINE-EVAL] stage=%s: %s", label, e)
        return None, f"{label}: {_exc_for_csv(e)}"


def _write_header_if_needed(csv_path: str, fieldnames: Sequence[str]) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(fieldnames)).writeheader()


def append_row(csv_path: str, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    _write_header_if_needed(csv_path, fieldnames)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(fieldnames)).writerow({k: row.get(k, "") for k in fieldnames})


def _load_existing_prompts(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    done: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            p = (r.get("story_prompt") or "").strip()
            if p:
                done.add(p)
    return done


def _hash_short(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _waveform_to_segment(wave: Any, sample_rate: int = SFX_RATE) -> AudioSegment:
    if hasattr(wave, "detach"):
        wave = wave.detach().cpu().numpy()
    y = np.asarray(wave, dtype=np.float64).squeeze()
    if y.ndim > 1:
        y = y.flatten()
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    return AudioSegment(
        pcm.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


def _read_story_prompts_from_final_results(path: str, dedupe: bool) -> List[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "story_prompt" not in reader.fieldnames:
            raise ValueError(f"Column 'story_prompt' missing in {path}")
        raw: List[str] = []
        for row in reader:
            p = (row.get("story_prompt") or "").strip()
            if p:
                raw.append(p)
    if not dedupe:
        return raw
    seen: set[str] = set()
    out: List[str] = []
    for p in raw:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _chunks(items: List[str], n: int) -> List[List[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _evaluate_row(
    evaluator: Any,
    audio_base64: str,
    story_prompt: str,
) -> Tuple[Dict[str, Any], str]:
    metrics: Dict[str, Any] = {
        "clap_score": "",
        "audio_richness_spectral_flatness": "",
        "audio_richness_spectral_entropy": "",
        "noise_floor_db": "",
        "audio_onsets": "",
        "cinematic_dynamic_range_db": "",
        "cinematic_crest_factor": "",
        "cinematic_spectral_flatness": "",
        "cinematic_spectral_entropy": "",
        "cinematic_spectral_centroid_hz": "",
        "spectral_kl_divergence": "",
        "fad_score": "",
    }
    errs: List[str] = []

    v, err = _run_stage("metric_clap_score", lambda: evaluator.get_clap_score(audio_base64, story_prompt))
    metrics["clap_score"] = v if err == "" else ""
    if err:
        errs.append(err)

    v, err = _run_stage("metric_audio_richness", lambda: evaluator.get_audio_richness(audio_base64))
    if err == "" and v is not None:
        flatness, spec_entropy = v
        metrics["audio_richness_spectral_flatness"] = flatness
        metrics["audio_richness_spectral_entropy"] = spec_entropy
    elif err:
        errs.append(err)

    v, err = _run_stage("metric_noise_floor", lambda: evaluator.get_noise_floor(audio_base64))
    metrics["noise_floor_db"] = v if err == "" else ""
    if err:
        errs.append(err)

    v, err = _run_stage(
        "metric_audio_onsets",
        lambda: evaluator.evaluate_sync_from_audio_base64(audio_base64),
    )
    metrics["audio_onsets"] = v if err == "" else ""
    if err:
        errs.append(err)

    v, err = _run_stage(
        "metric_cinematic",
        lambda: evaluator.get_cinematic_acoustic_metrics(audio_base64),
    )
    if err == "":
        c = v or {}
        metrics["cinematic_dynamic_range_db"] = c.get("dynamic_range_db", "")
        metrics["cinematic_crest_factor"] = c.get("crest_factor", "")
        metrics["cinematic_spectral_flatness"] = c.get("spectral_flatness", "")
        metrics["cinematic_spectral_entropy"] = c.get("spectral_entropy", "")
        metrics["cinematic_spectral_centroid_hz"] = c.get("spectral_centroid_hz", "")
    else:
        errs.append(err)

    return metrics, "; ".join(errs)


def _generate_batch(model_name: str, prompts: List[str], *, duration: int, tango_steps: int, audioldm_steps: int) -> List[Any]:
    if model_name == TANGO2:
        return Tango2Model.generate_for_batch(prompts, steps=tango_steps, duration=duration)
    if model_name == AUDIO_LDM2:
        return AudioLDM2Model.generate_for_batch(prompts, steps=audioldm_steps, duration=duration)
    raise ValueError(f"Unsupported model_name={model_name!r}")


def run_for_model(
    *,
    model_name: str,
    prompts: List[str],
    out_csv: str,
    audio_export_dir: str,
    evaluator: Any,
    batch_size: int,
    duration: int,
    tango_steps: int,
    audioldm_steps: int,
    resume: bool,
) -> None:
    if resume:
        already = _load_existing_prompts(out_csv)
        before = len(prompts)
        prompts = [p for p in prompts if p not in already]
        logger.info(
            "[BASELINE-EVAL] resume model=%s skipped=%d remaining=%d",
            model_name,
            before - len(prompts),
            len(prompts),
        )
    if not prompts:
        return

    os.makedirs(audio_export_dir, exist_ok=True)

    for batch in _chunks(prompts, max(1, batch_size)):
        t_batch0 = time.perf_counter()
        try:
            waves = _generate_batch(
                model_name,
                batch,
                duration=duration,
                tango_steps=tango_steps,
                audioldm_steps=audioldm_steps,
            )
        except Exception as e:
            logger.exception("[BASELINE-EVAL] batch generation failed model=%s", model_name)
            for prompt in batch:
                append_row(
                    out_csv,
                    RESULT_FIELDNAMES,
                    {
                        "story_prompt": prompt[:5000],
                        "total_seconds": "",
                        **{k: "" for k in RESULT_FIELDNAMES if k not in ("story_prompt", "total_seconds", "error")},
                        "audio_wav_path": "",
                        "audio_export_error": "",
                        "error": f"batch_generation: {_exc_for_csv(e)}",
                    },
                )
            continue

        gen_elapsed = time.perf_counter() - t_batch0
        per_gen = gen_elapsed / max(len(batch), 1)

        if len(waves) != len(batch):
            logger.warning(
                "[BASELINE-EVAL] batch len mismatch model=%s prompts=%d waves=%d",
                model_name,
                len(batch),
                len(waves),
            )

        for i, prompt in enumerate(batch):
            wave = waves[i] if i < len(waves) else None
            if wave is None:
                append_row(
                    out_csv,
                    RESULT_FIELDNAMES,
                    {
                        "story_prompt": prompt[:5000],
                        "total_seconds": "",
                        **{k: "" for k in RESULT_FIELDNAMES if k not in ("story_prompt", "total_seconds", "error")},
                        "audio_wav_path": "",
                        "audio_export_error": "",
                        "error": "batch_generation: missing waveform for this prompt index",
                    },
                )
                continue

            row_start = time.perf_counter()
            err_parts: List[str] = []
            base_metrics = {
                k: ""
                for k in RESULT_FIELDNAMES
                if k
                not in (
                    "story_prompt",
                    "total_seconds",
                    "error",
                    "audio_wav_path",
                    "audio_export_error",
                )
            }
            audio_wav_path = ""
            audio_export_error = ""
            try:
                segment = _waveform_to_segment(wave, SFX_RATE)
                wav_name = f"prompt_{_hash_short(prompt)}.wav"
                audio_wav_path = os.path.abspath(os.path.join(audio_export_dir, wav_name))
                try:
                    segment.export(audio_wav_path, format="wav")
                except Exception as ex:
                    audio_export_error = _exc_for_csv(ex)
                    err_parts.append(f"audio_export: {audio_export_error}")
                audio_base64 = audio_to_base64(segment)
                m, ev_err = _evaluate_row(evaluator, audio_base64, prompt)
                base_metrics.update(m)
                if ev_err:
                    err_parts.append(ev_err)
            except Exception as e:
                err_parts.append(f"row_pipeline: {_exc_for_csv(e)}")

            eval_elapsed = time.perf_counter() - row_start
            total_seconds = per_gen + eval_elapsed

            append_row(
                out_csv,
                RESULT_FIELDNAMES,
                {
                    "story_prompt": prompt[:5000],
                    "total_seconds": total_seconds,
                    **base_metrics,
                    "audio_wav_path": audio_wav_path,
                    "audio_export_error": audio_export_error,
                    "error": "; ".join(err_parts),
                },
            )
            logger.info(
                "[BASELINE-EVAL] model=%s prompt_len=%d total_s=%.3f",
                model_name,
                len(prompt),
                total_seconds,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Tango2 / AudioLDM2 evaluation from final_results.csv prompts.")
    default_results = os.path.join(_EVAL_DIR, "Results", "final_results.csv")
    parser.add_argument("--input-csv", type=str, default=default_results, help="Path to final_results.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_EVAL_DIR, "Results"),
        help="Directory for tango2_results.csv, audioldm2_results.csv, and infernce_results/<model>_audios/",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per generate_for_batch call")
    parser.add_argument("--duration", type=int, default=10, help="Generation duration in seconds")
    parser.add_argument("--tango-steps", type=int, default=100, help="Tango2 diffusion steps")
    parser.add_argument("--audioldm-steps", type=int, default=200, help="AudioLDM2 inference steps")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[TANGO2, AUDIO_LDM2],
        choices=[TANGO2, AUDIO_LDM2],
        help="Which baselines to run",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="Evaluate every CSV row, not unique story_prompt")
    parser.add_argument("--resume", action="store_true", help="Skip story_prompt values already present in output CSVs")
    parser.add_argument("--max-prompts", type=int, default=0, help="If >0, cap number of prompts after dedupe")
    parser.add_argument("--debug", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_csv = os.path.abspath(args.input_csv)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    prompts = _read_story_prompts_from_final_results(input_csv, dedupe=not args.no_dedupe)
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    logger.info(
        "[BASELINE-EVAL] prompts=%d dedupe=%s batch_size=%s duration=%s",
        len(prompts),
        not args.no_dedupe,
        args.batch_size,
        args.duration,
    )

    logger.info("[BASELINE-EVAL] init_models() …")
    init_models()

    evaluator: Any = None
    try:
        from Evaluation.evaluator import AudioEvaluator

        evaluator = AudioEvaluator()
    except Exception as e:
        logger.exception("[BASELINE-EVAL] AudioEvaluator unavailable: %s", e)
        sys.exit(1)

    paths = {
        TANGO2: os.path.join(out_dir, "tango2_results.csv"),
        AUDIO_LDM2: os.path.join(out_dir, "audioldm2_results.csv"),
    }
    infernce_root = os.path.join(out_dir, "infernce_results")
    audio_dirs = {
        TANGO2: os.path.join(infernce_root, "tango2_audios"),
        AUDIO_LDM2: os.path.join(infernce_root, "audioldm_audios"),
    }

    for m in args.models:
        run_for_model(
            model_name=m,
            prompts=list(prompts),
            out_csv=paths[m],
            audio_export_dir=audio_dirs[m],
            evaluator=evaluator,
            batch_size=args.batch_size,
            duration=args.duration,
            tango_steps=args.tango_steps,
            audioldm_steps=args.audioldm_steps,
            resume=bool(args.resume),
        )

    logger.info("[BASELINE-EVAL] done outputs: %s", ", ".join(paths[m] for m in args.models))


if __name__ == "__main__":
    main()
