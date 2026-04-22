import logging
import torch
from typing import List, Any
import numpy as np
import librosa
from pydub import AudioSegment
import base64
import io
from transformers import pipeline

logger = logging.getLogger(__name__)
class WordAligner:
    def __init__(self, model_id: str = "openai/whisper-base"):
        """
        Initializes the Whisper model via Hugging Face pipeline.
        The key parameter is return_timestamps="word" to get per-word timing.
        """
        logger.info(f"[*] Loading {model_id} for Word Alignment...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # The magic parameter here is return_timestamps="word"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            device=self.device,
            return_timestamps="word",
        )
        logger.info("[+] Aligner ready.")

    def _align_audio_array(self, audio_array: np.ndarray, sr: int) -> List[dict]:
        """Internal method to resample and run inference."""
        # Whisper strictly requires 16kHz audio
        if sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)

        # Ensure it's 1D (mono)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Run the model
        result: Any = self.pipe(audio_array)

        # Hugging Face ASR pipeline can return either a dict with "chunks"
        # or a simpler structure. We handle the dict-with-chunks case here.
        chunks = result["chunks"] if isinstance(result, dict) and "chunks" in result else []

        # Format the output to match the requested structure
        formatted_timestamps: List[dict] = []
        for chunk in chunks:
            # Sometimes the end timestamp can be None at the very end of the file
            start_time = (
                round(chunk["timestamp"][0], 2)
                if chunk["timestamp"][0] is not None
                else 0.0
            )
            end_time = (
                round(chunk["timestamp"][1], 2)
                if chunk["timestamp"][1] is not None
                else start_time + 0.2
            )

            formatted_timestamps.append(
                {
                    "word": chunk["text"].strip(),
                    "start": start_time,
                    "end": end_time,
                }
            )

        return formatted_timestamps

    def get_timestamps_from_base64(self, base64_audio: str, sample_rate: int = 16000) -> List[dict]:
        """Processes a base64 encoded audio string."""
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(base64_audio)
        import soundfile as sf
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        if sr != sample_rate:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=sample_rate)
        return self._align_audio_array(audio_array, sample_rate)

    def get_timestamps_from_parler_audio(self, parler_audio: AudioSegment) -> List[dict]:
        """
        Processes the audio output from ParlerTTSModel (pydub.AudioSegment).
        """
        samples = parler_audio.get_array_of_samples()
        audio_array = np.asarray(samples, dtype=np.float32) / 32768.0
        sr = parler_audio.frame_rate
        return self._align_audio_array(audio_array, sr)
