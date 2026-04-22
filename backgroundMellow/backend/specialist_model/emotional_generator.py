import logging
import numpy as np
from pydub import AudioSegment

from Variable.configurations import STEPS, EMOTIONAL_RATE, model_config
from helper.lib import get_model

logger = logging.getLogger(__name__)

def emotional_music_generator(prompt: str, duration_ms: int):
    """Generates a background music track."""
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms)")

    duration_s = int(duration_ms / 1000.0)
    audio_arr = get_model(model_config.music_model_name).generate(prompt, steps=STEPS, duration=duration_s)

    if audio_arr is None :
        raise ValueError(
            f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array."
        )

    # waveform = audio_arr.squeeze().cpu().numpy()

    # if waveform.size == 0:
    #     raise ValueError(f"Generated audio waveform is empty for prompt: '{prompt}'")

    # logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")

    # audio_bytes = (waveform * 32767 * EMOTIONAL_GAIN).astype(np.int16).tobytes()
    segment = AudioSegment(
        data=audio_arr.tobytes(),
        sample_width=2,
        frame_rate=EMOTIONAL_RATE,
        channels=1,
    )
    return segment

def emotional_music_generator_for_batch(prompts: list[str], duration_ms: int):
    """Generates a background music track for a batch of prompts."""
    logger.info(f"Generating for batch of {len(prompts)} prompts with duration {duration_ms}ms")
    duration_s = int(duration_ms / 1000.0)
    model_cls = get_model(model_config.music_model_name)
    audio_arr = model_cls.generate_for_batch(prompts, steps=STEPS, duration=duration_s)
    if audio_arr is None :
        raise ValueError(f"Failed to generate audio for batch of {len(prompts)} prompts. Model returned empty array.")
    segments = [AudioSegment(
        data=audio_arr_item.tobytes(),
        sample_width=2,
        frame_rate=EMOTIONAL_RATE,
        channels=1,
    ) for audio_arr_item in audio_arr]
    return segments

# TESTING


# import sys
# import os

# import torch

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)

# if __name__ == "__main__":
#     # Test the Emotional Music generator
#     test_prompt = "A calm and soothing background music with emotional undertones"
#     test_duration_ms = 5000  # 5 seconds
#     music_segment = emotional_music_generator(test_prompt, test_duration_ms)
#     samples = np.array(music_segment.get_array_of_samples())
#     music = torch.from_numpy(samples).unsqueeze(0).float() / 32767.0
#     # Play the generated music (uncomment the next line if running in an environment that supports audio playback)
#     # play(music_segment)
#     torchaudio.save("Debug/test_" + test_prompt.replace(" ", "_") + ".wav", music, EMOTIONAL_RATE)
#     print(f"[TEST] Generated Emotional Music AudioSegment: {music_segment}")
