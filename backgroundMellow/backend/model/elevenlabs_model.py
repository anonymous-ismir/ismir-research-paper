import threading
import logging
import os
from model.base_sound_model import SoundEffectsModel
logger = logging.getLogger(__name__)


class ElevenLabsModel(SoundEffectsModel):
    _instance = None
    # _lock = threading.Lock()
    # @classmethod
    # def get_instance(cls):
    #     if cls._instance is None:
    #         with cls._lock:
    #             if cls._instance is None:
    #                 print(f"Initializing ElevenLabs client with api key: {os.getenv('ELEVEN_LABS_KEY')}")
    #                 cls._instance = ElevenLabs(
    #                     api_key=os.getenv("ELEVEN_LABS_KEY"),
    #                 )
    #         return cls._instance
    #     return cls._instance
        
    @classmethod
    def generate(cls, prompt: str, steps: int = 100, duration: int = 10, **kwargs):
        # steps and duration are ignored; ElevenLabs API determines length
        from elevenlabs.client import ElevenLabs
        elevenlabs = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_KEY"))
        audio = elevenlabs.text_to_sound_effects.convert(text=prompt)
        return audio

    @classmethod
    def generate_for_batch(cls, prompts: list[str], steps: int = 100, duration: int = 10, **kwargs):
        from elevenlabs.client import ElevenLabs
        elevenlabs = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_KEY"))
        audio_arr = []
        for prompt in prompts:
            audio = elevenlabs.text_to_sound_effects.convert(text=prompt)
            audio_arr.append(audio)
        return audio_arr