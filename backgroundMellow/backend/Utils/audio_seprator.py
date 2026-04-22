import os
import subprocess
import librosa
import numpy as np
import torch
from panns_inference import AudioTagging, labels

# --- CONFIGURATION ---
INPUT_AUDIO = "movie_scene_mixed.wav" # Replace with your audio file
OUTPUT_DIR = "separated_stems"

def separate_sources(audio_path, output_dir):
    """
    Step 1: Uses Demucs to physically separate the audio.
    Extracts: vocals (narrator), bass, drums, and other (SFX/Ambience/Synths).
    """
    print(f"[*] Starting Demucs separation on {audio_path}...")
    
    # We use the subprocess module to call the Demucs CLI directly
    # 'htdemucs' is the default high-quality Hybrid Transformer model
    command = [
        "demucs",
        "-n", "htdemucs",
        "-o", output_dir,
        audio_path
    ]
    
    subprocess.run(command, check=True)
    
    # Demucs creates a subfolder based on the model name and input file name
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_folder = os.path.join(output_dir, "htdemucs", base_name)
    
    print(f"[+] Separation complete! Stems saved to: {stem_folder}")
    return stem_folder

def identify_sfx(audio_path, top_k=5):
    """
    Step 2: Uses PANNs (CNN14) to identify what sounds are in the audio.
    """
    print(f"\n[*] Analyzing sounds in {audio_path} using PANNs...")
    
    # Load audio using librosa (PANNs expects 32kHz sample rate)
    audio, _ = librosa.load(audio_path, sr=32000)
    
    # Add a batch dimension: shape becomes (1, audio_length)
    audio_expanded = audio[None, :] 
    
    # Initialize the PANNs AudioTagging model
    # (This will download the CNN14 weights on the first run)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AudioTagging(checkpoint_path=None, device=device)
    
    # Run inference
    # clipwise_output contains the probabilities for all 527 AudioSet classes
    (clipwise_output, embedding) = model.inference(audio_expanded)
    
    # Get the predicted probabilities for the first (and only) clip in the batch
    probabilities = clipwise_output[0]
    
    # Sort indices by highest probability
    sorted_indices = np.argsort(probabilities)[::-1]
    
    print("\n[+] Top Detected Sounds:")
    print("-" * 30)
    detected_sounds = []
    
    for i in range(top_k):
        class_index = sorted_indices[i]
        class_name = labels[class_index]
        prob = probabilities[class_index]
        
        # Only print if the model is somewhat confident (> 10%)
        if prob > 0.10:
            print(f"{i+1}. {class_name}: {prob*100:.1f}%")
            detected_sounds.append({"label": class_name, "confidence": float(prob)})
            
    if not detected_sounds:
        print("No strong sound signatures detected.")
        
    return detected_sounds

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_AUDIO):
        print(f"Error: Could not find '{INPUT_AUDIO}'. Please provide a valid audio file.")
    else:
        # 1. Run Separation
        stem_directory = separate_sources(INPUT_AUDIO, OUTPUT_DIR)
        
        # 2. Analyze the 'Other' stem (where the SFX and Ambience are)
        other_stem_path = os.path.join(stem_directory, "other.wav")
        
        if os.path.exists(other_stem_path):
            identified_sfx = identify_sfx(other_stem_path, top_k=5)
        else:
            print(f"Error: Could not find the 'other.wav' stem in {stem_directory}")
            
        # Optional: You can also analyze the 'vocals.wav' stem to confirm it is just speech
        # vocals_stem_path = os.path.join(stem_directory, "vocals.wav")
        # identify_sfx(vocals_stem_path)