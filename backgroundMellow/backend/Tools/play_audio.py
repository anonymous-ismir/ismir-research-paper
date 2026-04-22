# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)

import numpy as np
from pydub import AudioSegment
from Variable.dataclases import AudioCue, NarratorCue, Cue
from Variable.model_map import SPECIALIST_MAP
from helper.lib import get_model
import logging

logger = logging.getLogger(__name__)

def create_audio_from_audiocue(audio_cue: Cue) -> AudioSegment:
    """
    Create a single audio clip from a single cue (AudioCue or NarratorCue).
    """
    logger.info(f"Creating audio from cue: {audio_cue}\n\n")
    
    if isinstance(audio_cue, NarratorCue):
        logger.info(f"Creating audio from narrator cue: {audio_cue.id} ({audio_cue.audio_type})")
        specialist_func = SPECIALIST_MAP[audio_cue.audio_type]
        raw = specialist_func(audio_cue.story, audio_cue.narrator_description)
        return raw
    
    else:
        logger.info(f"Creating audio from audio cue: {audio_cue.audio_class} ({audio_cue.audio_type})")
        specialist_func = SPECIALIST_MAP[audio_cue.audio_type]
        audio_clip = specialist_func(audio_cue.audio_class, audio_cue.duration_ms)
        fade_ms = audio_cue.fade_ms
        # Safeguard: only apply fade if we have a positive duration
        if fade_ms is not None and fade_ms > 0:
            fade_time = min(fade_ms, audio_cue.duration_ms // 2)
            processed_clip = audio_clip.fade_in(fade_time).fade_out(fade_time)
        else:
            processed_clip = audio_clip
        processed_clip = processed_clip + audio_cue.weight_db
        return processed_clip


def save_audio_from_audiocue(audio_cue: Cue, output_path: str) -> AudioSegment:
    processed_clip = create_audio_from_audiocue(audio_cue)
    processed_clip.export(output_path, format="wav")
    logger.info(f"Saved audio to {output_path}")
    return processed_clip

# # test this function
# if __name__ == "__main__":
#     # Mock test audio cues
#     test_cues = [
#         AudioCue(id=1, audio_class="gun_shot.bang_bang.mp3", audio_type="MOVIE_BGM", start_time_ms=0, duration_ms=10000, fade_ms=0, weight_db=0.0),
#         # AudioCue(audio_class="Footsteps on gravel", audio_type="SFX", start_time_ms=2000, duration_ms=3000, fade_ms=500, weight_db=0.0),
#         # AudioCue(audio_class="Soft piano music", audio_type="MUSIC", start_time_ms=0, duration_ms=7000, fade_ms=2000, weight_db=-10.0),
#     ]
#     total_duration = 8000  # 8 seconds
#     final_audio = create_audio_from_audiocue(test_cues[0])
#     final_audio.export("test_play_audio.wav", format="wav")
