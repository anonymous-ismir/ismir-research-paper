from typing import List, Optional, Dict, cast
import logging
from collections import defaultdict

from pydub import AudioSegment
from Variable.dataclases import Cue, AudioCue, NarratorCue, AudioCueWithAudioBase64
from Variable.model_map import SPECIALIST_MAP
from helper.audio_conversions import audio_to_base64
from Tools.play_audio import create_audio_from_audiocue

logger = logging.getLogger(__name__)

# Types that have generate_for_batch; others are processed per-cue via create_audio_from_audiocue
BATCH_AUDIO_TYPES = ("SFX", "AMBIENCE", "MUSIC")

def _apply_cue_postprocess(segment: AudioSegment, cue: AudioCue) -> AudioSegment:
    """Apply cue-specific fade and weight to a raw segment (from batch generation)."""
    # Trim to cue duration if segment is longer
    if len(segment) > cue.duration_ms:
        segment = cast(AudioSegment, segment[: cue.duration_ms])
    fade_ms = cue.fade_ms if cue.fade_ms is not None else 0
    if fade_ms > 0:
        fade_time = min(fade_ms, cue.duration_ms // 2)
        segment = cast(AudioSegment, segment.fade_in(fade_time).fade_out(fade_time))
    segment = cast(AudioSegment, segment + cue.weight_db)
    return segment

def _segment_to_base64_and_wrap(cue: Cue, segment: AudioSegment) -> AudioCueWithAudioBase64:
    """Convert AudioSegment to base64 and wrap with cue."""
    base64_data = audio_to_base64(segment)
    return AudioCueWithAudioBase64(
        audio_cue=cue,
        audio_base64=base64_data,
        duration_ms=cue.duration_ms,
    )

def _process_batch_type(audio_type: str, cues: List[AudioCue]) -> List[AudioCueWithAudioBase64]:
    """
    Generate audio for all cues of a batch type (SFX, AMBIENCE, MUSIC) using
    the specialist's generate_for_batch. Parallelism is handled inside lib.py.
    """
    if not cues:
        return []
    specialist_func = SPECIALIST_MAP[audio_type]
    prompts = [c.audio_class for c in cues]
    duration_ms = max(c.duration_ms for c in cues)
    logger.info(
        "Batch generation for %s: %d prompts, duration_ms=%d (parallelism in lib)",
        audio_type,
        len(prompts),
        duration_ms,
    )
    segments = specialist_func(prompts, duration_ms)
    results = []
    for cue, segment in zip(cues, segments):
        try:
            processed = _apply_cue_postprocess(segment, cue)
            results.append(_segment_to_base64_and_wrap(cue, processed))
            logger.info("Successfully generated audio for cue %s (%s)", getattr(cue, "id", "N/A"), audio_type)
        except Exception as e:
            logger.error("Error post-processing cue %s (%s): %s", getattr(cue, "id", "N/A"), audio_type, e)
    return results


def _process_single_cue(cue: Cue) -> Optional[AudioCueWithAudioBase64]:
    """Generate audio for one cue (NARRATOR, MOVIE_BGM, or any non-batch type) via create_audio_from_audiocue."""
    try:
        audio_data = create_audio_from_audiocue(cue)
        return _segment_to_base64_and_wrap(cue, audio_data)
    except Exception as e:
        logger.error("Failed to process cue %s: %s", getattr(cue, "id", "unknown"), e)
        return None


def parallel_audio_generation(cues: List[Cue]) -> List[AudioCueWithAudioBase64]:
    """
    Generate audio for cues by type: for all audio_types in BATCH_AUDIO_TYPES,
    process all cues of that type in a batch (parallelism in lib.py), then process
    the rest (NARRATOR, MOVIE_BGM, etc.) individually.
    """
    if not cues:
        return []

    logger.info("Cues in parallel_audio_generation: %d total", len(cues))

    # Separate cues into batch and non-batch types, preserving input order
    batch_type_cues_map: Dict[str, List[AudioCue]] = {t: [] for t in BATCH_AUDIO_TYPES}
    non_batch_cues: List[Cue] = []

    for c in cues:
        if c.audio_type in BATCH_AUDIO_TYPES and isinstance(c, AudioCue):
            batch_type_cues_map[c.audio_type].append(c)
        else:
            non_batch_cues.append(c)

    results: List[AudioCueWithAudioBase64] = []

    # For each batch audio type (SFX, AMBIENCE, MUSIC, etc.), process the cues in batch in the same order as BATCH_AUDIO_TYPES
    for audio_type in BATCH_AUDIO_TYPES:
        if batch_type_cues_map[audio_type]:
            # Equivalent to parallel_audio_generation.py (48-57)
            results.extend(_process_batch_type(audio_type, batch_type_cues_map[audio_type]))

    # Process the non-batch cues individually (NARRATOR, MOVIE_BGM, etc.)
    for cue in non_batch_cues:
        out = _process_single_cue(cue)
        if out:
            results.append(out)
            logger.info("Successfully generated audio for cue %s (%s)", getattr(cue, "id", "N/A"), cue.audio_type)

    results.sort(key=lambda x: x.audio_cue.start_time_ms)
    logger.info("Completed audio generation: %d/%d cues generated successfully", len(results), len(cues))
    return results
