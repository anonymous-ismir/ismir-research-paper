import librosa
import numpy as np
from scipy.stats import entropy
from scipy.special import rel_entr

def get_audio_distribution(file_path, n_mfcc=13):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Flatten all MFCC coefficients into a single distribution of 'sounds'
    # We want to see the statistical "vibe" of the audio
    data = mfccs.flatten()
    
    # Create a histogram to represent the Probability Density Function (PDF)
    # We use a fixed range to ensure both distributions are comparable
    hist, bin_edges = np.histogram(data, bins=100, range=(-100, 100), density=True)
    
    # Add a tiny epsilon to avoid division by zero in KL calculation
    return hist + 1e-10

def calculate_kl(path_ref, path_gen):
    dist_p = get_audio_distribution(path_ref)
    dist_q = get_audio_distribution(path_gen)
    
    # Scipy's entropy(p, q) calculates KL Divergence
    kl_div = entropy(dist_p, dist_q)
    return kl_div

# --- Example Usage ---
# ground_truth = "dataset/forest_reference.wav"
# generated = "outputs/forest_ai_gen.wav"
# score = calculate_kl(ground_truth, generated)
# print(f"KL Divergence Score: {score:.4f}") 
# (Lower is better; 0.0 means identical statistical distributions)