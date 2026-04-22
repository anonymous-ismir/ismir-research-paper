# import os
# import sys
# # Add project root to path
# # import sys
# # import os

# # # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)
    
# # Cinemaudio-studio root (for tango_new when using Tango2)
# cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
# if cinema_studio_root not in sys.path:
#     sys.path.append(cinema_studio_root)    

import threading
import logging
import torch
import numpy as np
import soundfile as sf
import os
import tempfile
from model.base_sound_model import SoundEffectsModel
from Variable.configurations import SFX_RATE
from tango_new.tango2.tango import Tango
logger = logging.getLogger(__name__)

# Thread-local storage used to track worker_id per thread in parallel generation
_thread_local = threading.local()
class Tango2Model(SoundEffectsModel):
    """Sound effects using Tango2 (declare-lab/tango)."""

    _instance = None
    _lock = threading.Lock()
    _clap_model = None
    _clap_lock = threading.Lock()
    
    def __init__(self):
        self.model = None
        self.clap_model = None
        self.clap_lock = threading.Lock()


    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    
                    cls._instance = Tango(name="declare-lab/tango")
        return cls._instance

    @classmethod
    def _get_clap_model(cls):
        """
        Lazily initialize and return a shared CLAP model instance for semantic cropping.
        """
        if cls._clap_model is None:
            with cls._clap_lock:
                if cls._clap_model is None:
                    import laion_clap

                    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
                    clap_model.load_ckpt()
                    cls._clap_model = clap_model
        return cls._clap_model

    @classmethod
    def generate(cls, prompt: str, steps: int = 100, duration: int = 10, **kwargs):
        model = cls.get_instance()
        logger.info(f"Generating audio from Tango2 for prompt: {prompt} with duration {duration} seconds")
        return model.generate(prompt, steps=steps, duration=float(duration))

    @classmethod
    def generate_for_batch(cls, prompts: list[str], steps: int = 100, duration: int = 10, **kwargs):
        model = cls.get_instance()

        logger.info(f"Generating audio from Tango2 for batch of {len(prompts)} prompts with duration {duration} seconds")
        audios = model.generate_for_batch(prompts, steps=steps, duration=float(duration))

        # Apply semantic cropping using CLAP to better align each audio with its text prompt.
        try:
            clap_model = cls._get_clap_model()
        except Exception as e:
            logger.warning("Failed to initialize CLAP model for semantic cropping: %s", e)
            return audios

        for i, audio in enumerate(audios):
            try:
                cropped = cls.semantic_audio_crop(
                    audio,
                    prompts[i],
                    duration,
                    clap_model,
                    sample_rate=SFX_RATE,
                )
                audios[i] = cropped
            except Exception as e:
                logger.warning("Semantic audio crop failed for prompt index %s: %s", i, e)
        
        logger.info("Semantic audio crop completed for batch of %d prompts", len(audios))

        return audios

    
    @staticmethod
    def semantic_audio_crop(audio, text_prompt, target_duration_sec, clap_model, sample_rate: int):
        """
        Find the window of audio that semantically matches the prompt best.

        Args:
            audio: 1D wave tensor/array.
            text_prompt: Text description corresponding to the audio.
            target_duration_sec: Desired duration of the cropped clip in seconds.
            clap_model: Initialized CLAP model.
            sample_rate: Sample rate of the audio.
        """
        # Convert to 1D numpy array on CPU
        if isinstance(audio, torch.Tensor):
            y = audio.detach().cpu().numpy()
        else:
            y = np.asarray(audio)

        if y.ndim > 1:
            y = np.squeeze(y)

        sr = sample_rate
        target_samples = int(target_duration_sec * sr)

        if len(y) <= target_samples or target_samples <= 0:
            # Already shorter than or equal to target; return original
            return audio

        # Create 1-second step overlapping chunks
        step_samples = sr  # 1 second hops
        chunks = []
        starts = []

        for start in range(0, len(y) - target_samples + 1, step_samples):
            chunks.append(y[start : start + target_samples])
            starts.append(start)

        if not chunks:
            return audio

        # Save chunks to temp files for CLAP
        temp_paths = []
        try:
            for chunk in chunks:
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(temp_path, chunk, sr)
                temp_paths.append(temp_path)

            # Get CLAP embeddings for all chunks and the text
            audio_embeds = clap_model.get_audio_embedding_from_filelist(x=temp_paths, use_tensor=True)
            text_embed = clap_model.get_text_embedding([text_prompt], use_tensor=True)

            # Calculate Cosine Similarity to find the best match
            similarities = torch.nn.functional.cosine_similarity(audio_embeds, text_embed)
            best_index = int(torch.argmax(similarities).item())

            best_start = starts[best_index]
            best_audio = y[best_start : best_start + target_samples]
        finally:
            # Clean up temp files
            for p in temp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Convert back to the original type
        if isinstance(audio, torch.Tensor):
            return torch.from_numpy(best_audio).to(audio.device, dtype=audio.dtype)
        return best_audio


# Testing
# if __name__ == "__main__":
#     import soundfile as sf
#     prompt = ["Rolling thunder and lightning strikes", "girl making a phone call","people cheering in a stadium while rolling thunder and lightning strikes"]
#     audio_arrs = Tango2Model.generate_for_batch(prompt, duration=3)
#     for i, audio_arr in enumerate(audio_arrs):
#         sf.write(f"tango2_out_{i}.wav", audio_arr, 16000)