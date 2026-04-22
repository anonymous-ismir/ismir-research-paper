# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)

import logging
import numpy as np
from pydub import AudioSegment
from helper.lib import get_model
from Variable.configurations import STEPS, ENV_RATE, ENV_GAIN, model_config
logger = logging.getLogger(__name__)

def environment_generator(prompt: str, duration_ms: int):
    """Generates an ambient environmental sound."""
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms)")
    duration_s = int(duration_ms / 1000.0)
    audio_arr = get_model(model_config.env_model_name).generate(prompt, steps=STEPS, duration=duration_s)

    if audio_arr is None :
        logger.error(
            f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array."
        )
        return AudioSegment.silent(duration=duration_ms)

    # waveform = audio_arr.squeeze().cpu().numpy()

    # if waveform.size == 0:
    #     logger.error(f"Generated audio waveform is empty for prompt: '{prompt}'")
    #     return AudioSegment.silent(duration=duration_ms)

    # logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")

    # audio_bytes = (waveform * 32767 * ENV_GAIN).astype(np.int16).tobytes()
    segment = AudioSegment(
        data=audio_arr.tobytes(),
        sample_width=2,
        frame_rate=ENV_RATE,
        channels=1,
    )
    return segment


def environment_generator_for_batch(prompts: list[str], duration_ms: int):
    """Generates an ambient environmental sound for a batch of prompts."""
    logger.info(f"Generating for batch of {len(prompts)} prompts with duration {duration_ms}ms")
    duration_s = int(duration_ms / 1000.0)
    model_cls = get_model(model_config.env_model_name)
    audio_arr = []
    audio_arr = model_cls.generate_for_batch(prompts, steps=STEPS, duration=duration_s)
    segments = [AudioSegment(
        data=audio_arr_item.tobytes(),
        sample_width=2,
        frame_rate=ENV_RATE,
        channels=1,
    ) for audio_arr_item in audio_arr]
    return segments 

# TESTING


# if __name__ == "__main__":  
#     # Import torch and torchaudio only for testing
#     import torch
#     import torchaudio
    
#     logger.info("Project root added to sys.path: %s", project_root)
    
#     # test Env generator
#     test_prompt="gunshot ambient sound"
#     test_duration_ms=4000  # 4 seconds
#     env_audio=environment_generator(test_prompt,test_duration_ms)
    
#     samples = np.array(env_audio.get_array_of_samples())
#     env = torch.from_numpy(samples).unsqueeze(0).float() / 32767.0
#     # Play the generated environment sound (uncomment the next line if running in an environment that supports audio playback)
#     # play(env_audio)
    
#     # Create Debug directory if it doesn't exist
#     os.makedirs("Debug", exist_ok=True)
#     torchaudio.save("Debug/test_" + test_prompt.replace(" ", "_") + ".wav", env, ENV_RATE)
    # logger.info("Generated Environment AudioSegment")

