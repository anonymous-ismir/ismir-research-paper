import threading
import logging
import threading
import os
import concurrent.futures
from typing import Optional, List, Tuple
from Variable.configurations import PARALLEL_EXECUTION, PARALLEL_WORKERS
from model.base_sound_model import SoundEffectsModel
logger = logging.getLogger(__name__)

# Thread-local storage used to track worker_id per thread in parallel generation
_thread_local = threading.local()


class TangoFluxModel(SoundEffectsModel):
    """Manages TangoFlux model instances with support for parallel execution.
    
    Note: TangoFlux models are NOT thread-safe. For parallel execution, we maintain
    a pool of model instances (one per worker) to avoid race conditions.
    For sequential execution, we use a single instance with a lock.
    """

    _instance = None
    _lock = threading.Lock()
    _generate_lock = threading.Lock()  # Lock for serializing generate() calls in sequential mode
    _model_pool = []  # Pool of model instances for parallel execution
    _pool_lock = threading.Lock()
    _pool_size = 0
    _device = None  # lazily determined compute device
    
    def __init__(self):
        self.model = None
        self.device = None
        self.generate_lock = threading.Lock()
        self.pool_lock = threading.Lock()
        self.pool_size = 0
        self.pool = []


    @classmethod
    def _create_model(cls):
        """Create a TangoFluxInference model on the selected device."""
        from tangoflux import TangoFluxInference
        device = cls._get_device()
        logger = logging.getLogger(__name__)
        try:
            return TangoFluxInference(name="declare-lab/TangoFlux", device=device)
        except TypeError:
            logger.warning(
                "TangoFluxInference does not accept 'device' kwarg; "
                "falling back to library defaults."
            )
            return TangoFluxInference(name="declare-lab/TangoFlux")

    @classmethod
    def get_instance(cls):
        """Get the primary singleton instance (for sequential mode)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_model()
        return cls._instance
    
    @classmethod
    def _get_model_from_pool(cls, worker_id: int):
        """Get a model instance from the pool for parallel execution."""
        with cls._pool_lock:
            # Ensure pool is large enough
            while len(cls._model_pool) <= worker_id:
                cls._model_pool.append(cls._create_model())
            return cls._model_pool[worker_id]
    
    @classmethod
    def initialize_pool(cls, pool_size: int):
        """Pre-initialize model pool for parallel execution.
        
        This method is idempotent - it only creates new models if the pool
        is smaller than the requested size. Safe to call multiple times.
        """
        with cls._pool_lock:
            current_size = len(cls._model_pool)
            if current_size >= pool_size:
                # Pool already has enough models
                return
            
            cls._pool_size = max(cls._pool_size, pool_size)
            # Pre-create models up to pool_size
            logger = logging.getLogger(__name__)
            logger.info(f"Initializing TangoFlux model pool: {current_size} -> {pool_size} models")
            while len(cls._model_pool) < pool_size:
                cls._model_pool.append(cls._create_model())
            logger.info(f"TangoFlux model pool initialized with {len(cls._model_pool)} models")
            
    @classmethod
    def _get_current_worker_id(cls) -> Optional[int]:
        """Get worker ID from thread-local storage (set by parallel_audio_generation or generate_for_batch)."""
        return getattr(_thread_local, "worker_id", None)

    @classmethod
    def generate(cls, prompt: str, steps: int = 100, duration: int = 10, worker_id: Optional[int] = None, **kwargs):
        """Generate audio for a single prompt. Uses pool model when worker_id is set, else singleton with lock."""
        if worker_id is None:
            worker_id = cls._get_current_worker_id()
        if worker_id is not None:
            model = cls._get_model_from_pool(worker_id)
            return model.generate(prompt, steps=steps, duration=duration)
        model = cls.get_instance()
        with cls._generate_lock:
            return model.generate(prompt, steps=steps, duration=duration)

    @classmethod
    def generate_for_batch(
        cls,
        prompts: List[str],
        steps: int = 100,
        duration: int = 10,
        **kwargs,
    ) -> List:
        """
        Generate audio for multiple prompts in parallel (one TangoFlux model per worker thread)
        or sequentially, following PARALLEL_EXECUTION and PARALLEL_WORKERS.
        """
        if not prompts:
            return []
        if PARALLEL_EXECUTION:
            max_workers = min(len(prompts), PARALLEL_WORKERS)
            cls.initialize_pool(max_workers)
            logger.info(
                "TangoFlux generate_for_batch: PARALLEL for %d prompts with %d workers",
                len(prompts),
                max_workers,
            )

            def _generate_one(item: Tuple[int, str]):
                idx, prompt = item
                worker_id = idx % max_workers
                _thread_local.worker_id = worker_id
                model = cls._get_model_from_pool(worker_id)
                audio = model.generate(prompt, steps=steps, duration=duration)
                return (idx, audio)

            results: List[Tuple[int, object]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_generate_one, (i, p)): i for i, p in enumerate(prompts)
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        idx, audio = future.result()
                        results.append((idx, audio))
                    except Exception as e:
                        i = future_to_idx[future]
                        logger.error("TangoFlux generate_for_batch error for prompt index %s: %s", i, e)
            results.sort(key=lambda x: x[0])
            return [audio for _, audio in results]
        else:
            logger.info("TangoFlux generate_for_batch: SEQUENTIAL for %d prompts", len(prompts))
            model = cls.get_instance()
            with cls._generate_lock:
                return [
                    model.generate(p, steps=steps, duration=duration) for p in prompts
                ]


    