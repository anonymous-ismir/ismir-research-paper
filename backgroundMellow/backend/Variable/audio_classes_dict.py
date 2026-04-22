from headers.imports import Dict, Tuple
# --- Keyword to Specialist Mapping ---
# This dictionary maps keywords to (audio_prompt, specialist_type)
# We handle multi-word keys (like "dog barking") by checking them first.
SOUND_KEYWORDS: Dict[str, Tuple[str, str]] = {
    # SFX (Short)
    "dog barking": ("dog bark", "SFX"),
    "barking": ("dog bark", "SFX"),
    "ran": ("running footsteps", "SFX"),
    "running": ("running footsteps", "SFX"),
    "footsteps": ("footsteps", "SFX"),
    "door creaked": ("door creak", "SFX"),
    "creaked": ("door creak", "SFX"),
    "shout": ("shout", "SFX"),
    "screamed": ("scream", "SFX"),
    "laughed": ("laughing", "SFX"),

    # AMBIENCE (Longer, environmental)
    "rain": ("raining", "AMBIENCE"),
    "storm": ("thunder storm", "AMBIENCE"),
    "wind": ("howling wind", "AMBIENCE"),
    "forest": ("forest sounds", "AMBIENCE"),
    "city": ("city traffic", "AMBIENCE"),
    "shelter": ("rain on a roof", "AMBIENCE"), # Contextual!
    
    # MUSIC (Longer, emotional)
    "suddenly": ("suspenseful stinger", "MUSIC"), # "Stinger" is a short music cue
    "sad": ("sad violin", "MUSIC"),
    "happy": ("upbeat happy music", "MUSIC"),
    "felt a chill": ("eerie suspense music", "MUSIC"),
    "scared": ("scary horror music", "MUSIC"),
    "emotional": ("emotional piano", "MUSIC"),
}