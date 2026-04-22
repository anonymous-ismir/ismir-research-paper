import sys
import os
import torch
# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)       
print("Project root added to sys.path:", project_root)
print("Cinemaudio-studio root added to sys.path:", cinema_studio_root)
import soundfile as sf
from IPython.display import Audio
from tango2.tango import Tango

tango = Tango("declare-lab/tango2")

# prompt = "Piercing, terrified girl's scream"
prompts = [
    "Rolling thunder with lightning strikes",
    "A dog barks and rustles with some clicking",
    "Water flowing and trickling"
]
print("starting generation")
audios = tango.generate_for_batch(prompts,disable_progress=False,duration=3)
print("generation complete")
for i, audio in enumerate(audios):
    sf.write(f"Debug/{prompts[i]}.wav", audio, samplerate=16000)
    Audio(data=audio, rate=16000)
    
audios = tango.generate_for_batch(prompts,disable_progress=False)
print("generation complete")
for i, audio in enumerate(audios):
    sf.write(f"Debug/10_seconds_{prompts[i]}.wav", audio, samplerate=16000)
    Audio(data=audio, rate=16000)