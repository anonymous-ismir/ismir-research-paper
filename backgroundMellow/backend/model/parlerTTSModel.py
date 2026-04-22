import threading
import logging
import os
from typing import cast
import numpy as np
from pydub import AudioSegment
import torch
from model.base_sound_model import SoundEffectsModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

logger = logging.getLogger(__name__)
HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
)

class ParlerTTSModel(SoundEffectsModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        
    

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # HF_TOKEN = (
                    #     os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                    #     or os.getenv("HF_TOKEN")
                    #     or os.getenv("HUGGING_FACE_HUB_TOKEN")
                    # )
                    # model = ParlerTTSForConditionalGeneration.from_pretrained(
                    #     "ai4bharat/indic-parler-tts", token=HF_TOKEN
                    # )
                    # tokenizer = AutoTokenizer.from_pretrained(
                    #     "ai4bharat/indic-parler-tts", token=HF_TOKEN
                    # )
                    # description_tokenizer = AutoTokenizer.from_pretrained(
                    #     model.config.text_encoder._name_or_path, token=HF_TOKEN
                    # )
                    # cls._instance = {
                    #     "model": model,
                    #     "tokenizer": tokenizer,
                    #     "description_tokenizer": description_tokenizer,
                    # }
                    
                    cls._instance = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1")
                    cls._instance.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
                    return cls._instance

    @classmethod
    def _tts_output_to_audio_segment(cls, raw):
        if cls._instance is None:
            cls.get_instance()
        sample_rate = cls._instance.config.sampling_rate or 44100
        """Convert TTS output (torch.Tensor or np.ndarray, float32) to AudioSegment."""
        if hasattr(raw, "cpu"):
            raw = raw.cpu().numpy().squeeze()
        audio_arr = np.asarray(raw, dtype=np.float32)
        audio_arr = np.clip(audio_arr, -1.0, 1.0)
        gain = 0.9
        audio_bytes = (audio_arr * 32767 * gain).astype(np.int16).tobytes()
        seg = AudioSegment(
            data=audio_bytes,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        return seg 
    
    
    @classmethod
    def generate(cls, prompt: str, description: str):
        if cls._instance is None or cls._instance.tokenizer is None:
            cls.get_instance()
        input_ids = cls._instance.tokenizer(description, return_tensors="pt").input_ids
        prompt_input_ids = cls._instance.tokenizer(prompt, return_tensors="pt").input_ids
        logger.info(f"Generating audio from ParlerTTS for prompt: {prompt} with description: {description} a rate {cls._instance.config.sampling_rate}")
        generation = cls._instance.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        return cls._tts_output_to_audio_segment(generation)

    
    @classmethod
    def generate_for_batch(cls, prompts: list[str], descriptions: list[str]):
        if cls._instance is None:
            cls.get_instance()
        audio_arrs = []
        for prompt, description in zip(prompts, descriptions):
            audio_arr = cls.generate(prompt, description)
            audio_arrs.append(audio_arr)
        return audio_arrs
    
    @classmethod
    def get_sample_rate(cls):
        if cls._instance is None:
            cls.get_instance()
        return cls._instance.config.sampling_rate


# Testing        
# if __name__ == "__main__":
#     import soundfile as sf
#     prompt = "hey there, how are you doing today?"
#     description = "A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting."
#     generation = ParlerTTSModel.generate(prompt, description)
#     audio_arr = generation.cpu().numpy().squeeze()
#     sf.write("parler_tts_out.wav", audio_arr, ParlerTTSModel._instance.config.sampling_rate)
    