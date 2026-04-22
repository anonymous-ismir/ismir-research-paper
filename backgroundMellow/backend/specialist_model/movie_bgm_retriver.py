# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)

import logging
import os
from pydub import AudioSegment
from Variable.configurations import PATH_TO_MOVIE_BGMS
logger = logging.getLogger(__name__)



def movie_bgm_retriver(path: str, duration_ms: int):
    """Generates an movie background music sound from data."""
    logger.info(f"Retrieving: '{path}' ({duration_ms}ms)")
    duration_s = int(duration_ms / 1000.0)
    # Build an absolute path so this works regardless of current working directory
    backend_root = os.path.dirname(os.path.dirname(__file__))
    bgm_dir = os.path.join(backend_root, PATH_TO_MOVIE_BGMS)
    full_path = os.path.join(bgm_dir, path)
    
    if(full_path[-4:] != ".mp3"):
        full_path = full_path + ".mp3"
    audio_segment = AudioSegment.from_file(full_path)
    stretch_factor = duration_s / audio_segment.duration_seconds
    logger.info(f"Stretch factor: {stretch_factor}")
    # if(stretch_factor > 1): audio_segment = stretch_compression(audio_segment, stretch_factor)
    # elif(stretch_factor < 1): audio_segment = stretch_expansion(audio_segment, stretch_factor)
    # else: audio_segment = audio_segment
 
    return audio_segment


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
