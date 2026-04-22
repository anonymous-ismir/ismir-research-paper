import threading
import logging
import torch

class SoundEffectsModel:
    """Base class for sound effects models. Use get_model(model_name) to get a concrete implementation."""

    _instance = None
    _lock = threading.Lock()
    _device = None  # Subclasses (e.g. TangoFluxModel) may set and use for device selection

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.model = None

    @classmethod
    def _get_device(cls) -> str:
        """Select best available device: CUDA > MPS > CPU. Override in subclasses if needed."""
        existing = getattr(cls, "_device", None)
        if existing is not None:
            return str(existing)
        device = "cpu"
        try:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                backends = getattr(torch, "backends", None)
                mps = getattr(backends, "mps", None) if backends is not None else None
                if mps is not None and mps.is_available():
                    device = "mps"
        except Exception:
            device = "cpu"
        logger = logging.getLogger(__name__)
        logger.info(f"Using device: {device}")
        cls._device = device
        return device

    @classmethod
    def get_instance(cls):
        raise NotImplementedError("get_instance is not implemented for SoundEffectsModel")

    @classmethod
    def generate(cls, prompt: str, steps: int = 100, duration: int = 10, **kwargs):
        raise NotImplementedError("generate is not implemented for SoundEffectsModel")
    
    @classmethod
    def generate_for_batch(cls, prompts: list[str], steps: int = 100, duration: int = 10, **kwargs):
        raise NotImplementedError("generate_for_batch is not implemented for SoundEffectsModel")
