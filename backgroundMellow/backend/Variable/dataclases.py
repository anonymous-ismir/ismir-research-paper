from headers.imports import dataclass
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Sequence
from Variable.configurations import READING_SPEED_WPS
@dataclass
class BaseCue:
    """Base class for all cue types. Common fields shared by AudioCue and NarratorCue."""
    id: int
    audio_type: str
    start_time_ms: int
    duration_ms: int
    
@dataclass
class AudioCue(BaseCue):
    """Stores all information needed for a single sound event."""
    audio_class: str  # Prompt to send to the specialist (e.g., "rain", "dog bark")
    weight_db: float  # Volume adjustment in decibels (dB)
    fade_ms: int = 500  # Default fade in/out time
@dataclass
class NarratorCue(BaseCue):
    """Stores all information needed for a narrator TTS cue."""
    story: str
    narrator_description: str
    weight_db: float = 0

Cue = Union[AudioCue, NarratorCue]
@dataclass
class AudioCueWithAudioBase64:
    audio_cue: Cue
    audio_base64: str
    duration_ms: int

# Request/Response Models
class DecideCuesRequest(BaseModel):
    story_text: str = Field(..., description="The story text to analyze")
    speed_wps: Optional[float] = Field(READING_SPEED_WPS, description="Words per second reading speed")
class DecideCuesResponse(BaseModel):
    cues: Sequence[Union[AudioCue, NarratorCue]]
    total_duration_ms: int
    message: str
class CueRequest(BaseModel):
    """Permissive model for parsing cue JSON (AudioCue or NarratorCue)."""
    id: Optional[int] = 0
    audio_type: Optional[str] = "SFX"
    start_time_ms: Optional[int] = 0
    duration_ms: Optional[int] = 2000
    audio_class: Optional[str] = None
    weight_db: Optional[float] = None
    fade_ms: Optional[int] = None
    story: Optional[str] = None
    narrator_description: Optional[str] = None
class GenerateAudioFromCuesRequest(BaseModel):
    cues: List[CueRequest]
    total_duration_ms: int
    
class GenerateAudioFromCuesResponse(BaseModel):
    audio_cues: List[AudioCueWithAudioBase64]
    message: str = Field(..., description="Message indicating success or failure")
    
class GenerateFromStoryRequest(BaseModel):
    story_text: str = Field(..., description="The story text to process")
    speed_wps: Optional[float] = Field(READING_SPEED_WPS, description="Words per second reading speed")
    
class GenerateFromStoryResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio data")
    
class GenerateAudioCuesWithAudioBase64Request(BaseModel):
    cues: List[AudioCueWithAudioBase64]
    story_text: str = Field(..., description="The story text to process")
    speed_wps: Optional[float] = Field(READING_SPEED_WPS, description="Words per second reading speed")

class CheckMissingCuesResponse(BaseModel):
    """Response for check-missing-audio-cues: cues that need audio generated (for loading UI)."""
    missing_cues: List[dict] = Field(default_factory=list, description="Cues not covered by story (id, audio_class, etc.)")
    total_duration_ms: int = Field(0, description="Total duration in ms")
    
class GenerateAudioCuesWithAudioBase64Response(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio data")
    message: str = Field(..., description="Message indicating success or failure")
    
class EvaluateAudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio data")
    text: str = Field(..., description="The text to evaluate the audio against")
    
class EvaluateAudioResponse(BaseModel):
    clap_score: float = Field(..., description="The CLAP score of the audio")
    spectral_richness: float = Field(..., description="The spectral richness of the audio")
    noise_floor: float = Field(..., description="The noise floor of the audio")
    audio_onsets: int = Field(..., description="The number of audio onsets in the audio")
    message: str = Field(..., description="Message indicating success or failure")
