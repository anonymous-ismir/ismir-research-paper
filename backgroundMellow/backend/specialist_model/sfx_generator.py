# import sys
# import os
# import torch
# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# # Cinemaudio-studio root (for tango_new when using Tango2)
# cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
# if cinema_studio_root not in sys.path:
#     sys.path.append(cinema_studio_root)       
# print("Project root added to sys.path:", project_root)

from helper.lib import get_model
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)
from Variable.configurations import STEPS, SFX_RATE, model_config

def sfx_generator(prompt: str, duration_ms: int):
    """Generates a short sound effect using the specified model.

    Args:
        prompt: Text description of the sound effect.
        duration_ms: Target duration in milliseconds.
        model_name: One of "TangoFlux", "ElevenLabs", or "Tango2" (default: Tango2).
    """
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms) with model={model_config.sfx_model_name}")

    duration_s = int(duration_ms / 1000.0)
    model_cls = get_model(model_config.sfx_model_name)
    audio_arr = model_cls.generate(prompt, steps=STEPS, duration=duration_s)

    if audio_arr is None :
        raise ValueError(f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array.")

    # waveform = audio_arr.squeeze().cpu().numpy()

    # if waveform.size == 0:
    #     raise ValueError(f"Generated audio waveform is empty for prompt: '{prompt}'")

    # logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")

    # audio_bytes = (waveform * 32767 * SFX_GAIN).astype(np.int16).tobytes()
    segment = AudioSegment(
        data=audio_arr.tobytes(),
        sample_width=2,
        frame_rate=SFX_RATE,
        channels=1,
    )
    return segment

def sfx_generator_for_batch(prompts: list[str], duration_ms: int):
    """Generates a short sound effect using the specified model.

    Args:
        prompts: List of text descriptions of the sound effect.
        duration_ms: Target duration in milliseconds.
        model_name: One of "TangoFlux", "ElevenLabs", or "Tango2" (default: Tango2).
    """
    model_name = model_config.sfx_model_name
    logger.info(f"Generating for batch of {len(prompts)} prompts with duration {duration_ms}ms with model={model_name}")

    duration_s = int(duration_ms / 1000.0)
    model_cls = get_model(model_name)
    audio_arr = []
    audio_arr = model_cls.generate_for_batch(prompts, steps=STEPS, duration=duration_s)

    segments = [AudioSegment(
        data=audio_arr_item.tobytes(),
        sample_width=2,
        frame_rate=SFX_RATE,
        channels=1,
    ) for audio_arr_item in audio_arr]
    return segments


# TESTING

# if __name__ == "__main__":
#     # Test the SFX generator
#     test_prompts = ["Rolling thunder with lightning strikes","Gun shots","Explosion"]
#     test_duration_ms = 5000  # 5 seconds
#     sfx_audio = sfx_generator_for_batch(test_prompts, test_duration_ms)
#     for i in range(len(test_prompts)):
#         import soundfile as sf
#         # Convert int16 samples to float32 and normalize to [-1, 1] range
#         samples = np.array(sfx_audio[i].get_array_of_samples(), dtype=np.float32) / 32767.0
#         # soundfile expects numpy array, shape (n_samples,) or (n_samples, n_channels)
#         samples = samples.astype(np.float32)
#         os.makedirs("Debug", exist_ok=True)
#         sf.write("Debug/test_" + test_prompts[i].replace(" ", "_") + ".wav", samples, SFX_RATE)
#         print(f"[TEST] Generated SFX AudioSegment: {sfx_audio[i]}")
        
