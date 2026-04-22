import torch
import librosa
import laion_clap
import numpy as np

# 1. INITIALIZE CLAP MODEL (The "Semantic Judge")
# This model converts both audio and text into a shared vector space.
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # Downloads pre-trained weights automatically

def calculate_semantic_alignment(audio_path, text_prompt):
    # Get Audio Embedding
    audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
    # Get Text Embedding
    text_embed = model.get_text_embedding([text_prompt], use_tensor=True)
    
    # Cosine Similarity: Closer to 1.0 means perfect semantic match
    similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed)
    return similarity.item()

# 2. ONSET DETECTION (The "Timing Judge")
# Use this to find exactly when a 'beat' or 'shot' occurs in your audio.
def detect_temporal_events(audio_path):
    y, sr = librosa.load(audio_path)
    # Detects sudden increases in energy (onsets)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # You can compare these 'onset_times' to your 'JSON Script timestamps'
    # Sync Error = abs(Predicted_Timestamp - Detected_Onset)
    return onset_times

# --- RUNNING THE TEST ---
# story_segment = "A heavy gunshot echoes through the valley."
# score_file = "output_score.wav"
# print(f"CLAP Alignment Score: {calculate_semantic_alignment(score_file, story_segment)}")