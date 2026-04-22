from headers.imports import Dict, Tuple
import os

# --- Reading Speed Configuration ---
READING_SPEED_WPS = 2.0  # Words Per Second

# --- Duration Configuration ---
DEFAULT_SFX_DURATION_MS = 2000  # 2 seconds (for short SFX)
DEFAULT_FADE_MS = 500           # 500ms fade in/out for most sounds

# --- Weight/Volume Configuration ( LOUD/LITTLE logic ) ---

DEFAULT_WEIGHT_DB = 0.0
LOUD_WEIGHT_DB = 6.0
FAINT_WEIGHT_DB = -6.0

MODIFIER_WORDS = {
    "loud": LOUD_WEIGHT_DB,
    "faint": FAINT_WEIGHT_DB,
    "little": FAINT_WEIGHT_DB,
    "soft": FAINT_WEIGHT_DB,
    "quiet": FAINT_WEIGHT_DB,
    "roaring": LOUD_WEIGHT_DB,
    "blaring": LOUD_WEIGHT_DB,
}




# Specialist model configuartions
# Note: STEPS=48 to avoid IndexError in scheduler 
# The scheduler creates (steps+1) sigmas, so with STEPS=48, we get 49 sigmas (indices 0-48)
# When step_index=48, it accesses step_index+1=49 which is valid
STEPS=48

# Parallel execution configuration
PARALLEL_EXECUTION = True  # Set to False for sequential execution (thread-safe but slower)
PARALLEL_WORKERS = 2  # Number of worker threads/processes for parallel execution (default: 2)


SFX_RATE=16000
SFX_GAIN=0.5

ENV_RATE=16000
ENV_GAIN=0.7
    
EMOTIONAL_RATE=16000
EMOTIONAL_GAIN=0.8


PATH_TO_MOVIE_BGMS = "data/movie_bgms"
PATH_TO_MOVIE_BGM_METADATA = "data/metadata/movie_bgms.csv"


SOUND_TYPES = ["SFX", "AMBIENCE", "MUSIC", "NARRATOR","MOVIE_BGM"]


# Supported model names: use these with get_model() or generate_sound()
TANGO_FLUX = "TangoFlux"
ELEVEN_LABS = "ElevenLabs"
AUDIO_LDM2 = "AudioLDM2"
TANGO2 = "Tango2"


SFX_MODEL = TANGO2
ENV_MODEL = TANGO2
MUSIC_MODEL = TANGO2


class ModelConfig:
    
    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = ModelConfig()
        return cls._instance
    
    def __init__(self):
        self.sfx_model_name = TANGO2
        self.env_model_name = TANGO2
        self.music_model_name = TANGO2
        self.narrator_model_name = "parlertts"
            
        self.fill_coverage_by_llm = False
        self.use_dsp = True  
        
        self.use_movie_bgms = False
        self.use_narrator = True
        self.decide_audio_model_name = "gemini-3-flash-preview"
        
        self.use_llm_to_predict_align = False
        self.use_dsp_to_predict_align = False
        self.use_avg_llm_and_dsp_to_predict_align = True
        self.use_dl_based_llm_and_dsp_alignment_predictor = False
        
        
model_config = ModelConfig.get_instance()