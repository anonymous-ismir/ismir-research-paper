import base64
import io
from pydub import AudioSegment
from Variable.dataclases import AudioCue, NarratorCue, Cue
from Variable.configurations import DEFAULT_WEIGHT_DB, SOUND_TYPES

def dict_to_cue(d: dict) -> Cue:
    """Convert a dict (e.g. from JSON or model_dump) to AudioCue or NarratorCue."""
    a_type = str(d.get("audio_type") or "SFX").upper()
    sid = int(d.get("id") or 0)
    start_ms = int(d.get("start_time_ms") or 0)
    dur_ms = int(d.get("duration_ms") or 2000)
    is_narrator = (
        a_type == "NARRATOR"
        or (d.get("story") is not None and d.get("story") != "")
        or (d.get("narrator_description") is not None and d.get("narrator_description") != "")
    )
    if is_narrator:
        return NarratorCue(
            id=sid,
            story=str(d.get("story") or ""),
            narrator_description=str(d.get("narrator_description") or ""),
            audio_type="NARRATOR",
            start_time_ms=start_ms,
            duration_ms=dur_ms,
        )
    w_db = d.get("weight_db")
    f_ms = d.get("fade_ms")
    return AudioCue(
        id=sid,
        audio_class=str(d.get("audio_class") or "ambient texture"),
        audio_type=a_type if a_type in SOUND_TYPES else "SFX",
        start_time_ms=start_ms,
        duration_ms=dur_ms,
        weight_db=float(w_db) if w_db is not None else DEFAULT_WEIGHT_DB,
        fade_ms=int(f_ms) if f_ms is not None else 500,
    )


def audio_cue_to_dict(cue: Cue) -> dict:
    """Convert AudioCue or NarratorCue to dictionary for serialization."""
    base = {
        "id": cue.id,
        "audio_type": cue.audio_type,
        "start_time_ms": cue.start_time_ms,
        "duration_ms": cue.duration_ms,
    }
    if isinstance(cue, NarratorCue):
        base["story"] = cue.story
        base["narrator_description"] = cue.narrator_description
        return base
    else:
        base["audio_class"] = cue.audio_class
        base["weight_db"] = cue.weight_db
        base["fade_ms"] = cue.fade_ms
        return base

# Helper function to convert AudioSegment to base64
def audio_to_base64(audio: AudioSegment, format: str = "wav") -> str:
    """Convert AudioSegment to base64 encoded string"""
    buffer = io.BytesIO()
    audio.export(buffer, format=format)
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_base64

def base64_to_audio(audio_base64: str) -> AudioSegment:
    """Convert base64 encoded string to AudioSegment"""
    audio_bytes = base64.b64decode(audio_base64)
    return AudioSegment.from_file(io.BytesIO(audio_bytes))