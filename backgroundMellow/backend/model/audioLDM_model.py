import os
import sys
# Add project root to path
# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# # Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)
    
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)    

import threading
import logging
from typing import Any, List, cast

import torch

# Compatibility shim:
# diffusers==0.18.x imports `cached_download` from huggingface_hub, but recent
# huggingface_hub versions removed that symbol. We provide a local fallback so
# this file can run without modifying the Python environment.
try:
    import huggingface_hub as _hf_hub

    if not hasattr(_hf_hub, "cached_download"):
        import hashlib
        import os
        from urllib.parse import urlparse
        from urllib.request import urlretrieve

        def _cached_download_compat(
            url_or_filename,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            use_auth_token=None,
            user_agent=None,
            revision=None,
            etag_timeout=10,
            token=None,
            **kwargs,
        ):
            del proxies, resume_download, use_auth_token, user_agent, revision, etag_timeout, token, kwargs
            if local_files_only:
                raise FileNotFoundError(
                    f"local_files_only=True and file not found in cache: {url_or_filename}"
                )

            parsed = urlparse(str(url_or_filename))
            if parsed.scheme not in ("http", "https"):
                if os.path.exists(url_or_filename):
                    return url_or_filename
                raise FileNotFoundError(f"Path does not exist: {url_or_filename}")

            cache_root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            os.makedirs(cache_root, exist_ok=True)
            cache_name = hashlib.sha256(str(url_or_filename).encode("utf-8")).hexdigest()
            cache_path = os.path.join(cache_root, cache_name)

            if force_download or (not os.path.exists(cache_path)):
                urlretrieve(url_or_filename, cache_path)

            return cache_path

        setattr(_hf_hub, "cached_download", _cached_download_compat)
except Exception:
    # Best-effort shim; continue and let diffusers raise real import errors.
    pass

try:
    from diffusers import AudioLDM2Pipeline as _PipelineClass  # type: ignore[attr-defined]
    _PIPELINE_KIND = "audioldm2"
    _DEFAULT_REPO_ID = "cvssp/audioldm2-large"
except Exception:
    # Older diffusers versions (e.g., 0.18.x) don't expose AudioLDM2Pipeline.
    from diffusers import AudioLDMPipeline as _PipelineClass  # type: ignore[attr-defined]
    _PIPELINE_KIND = "audioldm"
    _DEFAULT_REPO_ID = "cvssp/audioldm-l-full"

from model.base_sound_model import SoundEffectsModel


logger = logging.getLogger(__name__)


class AudioLDM2Model(SoundEffectsModel):
    """Sound effects model using Hugging Face AudioLDM2."""

    _instance = None
    _lock = threading.Lock()

    REPO_ID = _DEFAULT_REPO_ID

    def __init__(self):
        super().__init__(model_name="audioldm2")
        # Pipeline is created lazily via get_instance; kept untyped here so
        # static checkers don't require diffusers type information.
        self.pipeline = None

    @classmethod
    def get_instance(cls):
        """Lazily initialize and return a shared AudioLDM pipeline instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    device = cls._get_device()

                    # Prefer float16 on CUDA, fall back to float32 otherwise.
                    torch_dtype = torch.float16 if device == "cuda" else torch.float32

                    logger.info(
                        "Loading %s pipeline from %s on %s with dtype=%s",
                        _PIPELINE_KIND,
                        cls.REPO_ID,
                        device,
                        torch_dtype,
                    )

                    pipe = _PipelineClass.from_pretrained(
                        cls.REPO_ID,
                        torch_dtype=torch_dtype,
                    )
                    pipe = pipe.to(device)
                    cls._instance = pipe

        return cls._instance

    @classmethod
    def generate(
        cls,
        prompt: str,
        steps: int = 200,
        duration: int = 10,
        negative_prompt: str | None = None,
        num_waveforms_per_prompt: int = 1,
        **kwargs,
    ):
        """
        Generate a single waveform for a text prompt.

        Returns:
            1D numpy array (waveform at SFX_RATE).
        """
        pipe = cls.get_instance()

        logger.info(
            "Generating audio from %s for prompt: %s (duration=%ss, steps=%s)",
            _PIPELINE_KIND,
            prompt,
            duration,
            steps,
        )

        pipe_args = {
            "num_inference_steps": steps,
            "audio_length_in_s": float(duration),
            **kwargs,
        }
        if negative_prompt is not None:
            pipe_args["negative_prompt"] = negative_prompt
        if num_waveforms_per_prompt != 1:
            pipe_args["num_waveforms_per_prompt"] = num_waveforms_per_prompt

        result = cast(Any, pipe)(prompt, **pipe_args)

        # diffusers returns a list of waveforms; we use the first by default.
        audios = getattr(result, "audios", result[0] if isinstance(result, tuple) else result)
        audio = audios[0]

        return audio

    @classmethod
    def generate_for_batch(
        cls,
        prompts: List[str],
        steps: int = 200,
        duration: int = 10,
        negative_prompt: str | List[str] | None = None,
        num_waveforms_per_prompt: int = 1,
        **kwargs,
    ):
        """
        Generate audio for a batch of prompts.

        Args mirror the single-prompt generate method but operate on a list of prompts.
        Returns:
            List of 1D numpy arrays (one per prompt when num_waveforms_per_prompt==1).
        """
        pipe = cls.get_instance()

        logger.info(
            "Generating audio from %s for batch of %d prompts "
            "(duration=%ss, steps=%s)",
            _PIPELINE_KIND,
            len(prompts),
            duration,
            steps,
        )

        pipe_args = {
            "num_inference_steps": steps,
            "audio_length_in_s": float(duration),
            **kwargs,
        }
        if negative_prompt is not None:
            pipe_args["negative_prompt"] = negative_prompt
        if num_waveforms_per_prompt != 1:
            pipe_args["num_waveforms_per_prompt"] = num_waveforms_per_prompt

        result = cast(Any, pipe)(prompts, **pipe_args)

        audios = getattr(result, "audios", result[0] if isinstance(result, tuple) else result)

        return audios



## test the model
if __name__ == "__main__":
    prompt = "The sound of a hammer hitting a wooden surface"
    audio = AudioLDM2Model.generate(prompt)
    import soundfile as sf
    sf.write("Debug/audioLDM2_test.wav", audio, 16000)