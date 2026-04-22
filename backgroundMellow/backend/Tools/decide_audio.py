# import sys
# import os


# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)

import logging
import json
import re
import warnings

from numpy import True_
from Variable.dataclases import AudioCue, NarratorCue, Cue
from Variable.configurations import MODIFIER_WORDS, DEFAULT_WEIGHT_DB, DEFAULT_SFX_DURATION_MS,model_config
from Utils.prompts import (
    gemini_audio_prompt_with_narrator_without_movie_bgms,
    gemini_add_movie_bgms,
)
import math
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import google.genai as genai
import spacy
from Variable.configurations import PATH_TO_MOVIE_BGM_METADATA, SOUND_TYPES
try:
    nlp = spacy.load("en_core_web_sm")
    nlp_available = True
except Exception:
    nlp = None
    nlp_available = False
from helper.lib import read_movie_bgms_csv
from Utils.llm import query_llm
import os
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(backend_dir, ".env"))
logger = logging.getLogger(__name__)
GEMINI_AVAILABLE = True
USE_NEW_GENAI = True


def _classify_audio_type(word: str, pos_tag: str, context: str = "") -> Tuple[str | None, str | None]:
    """
    Classifies a word/phrase into audio type (SFX/AMBIENCE/MUSIC) and generates a prompt.
    Returns (audio_type, audio_prompt)
    """
    word_lower = word.lower()
    
    # Sound-producing verbs -> SFX
    sound_verbs = {
        'bark', 'barking', 'barked', 'barked',
        'cat','cat purring', 'cat hissing', 'cat meowing', 'cat purring', 'cat hissing',
        'run', 'running', 'ran', 'runs',
        'scream', 'screaming', 'screamed', 'screams',
        'shout', 'shouting', 'shouted', 'shouts',
        'laugh', 'laughing', 'laughed', 'laughs',
        'cry', 'crying', 'cried', 'cries',
        'knock', 'knocking', 'knocked', 'knocks',
        'crash', 'crashing', 'crashed', 'crashes',
        'slam', 'slamming', 'slammed', 'slams',
        'bang', 'banging', 'banged', 'bangs',
        'whistle', 'whistling', 'whistled', 'whistles',
        'clap', 'clapping', 'clapped', 'claps',
        'step', 'stepping', 'stepped', 'steps',
        'walk', 'walking', 'walked', 'walks',
    }
    
    # Environmental/background nouns -> AMBIENCE
    environment_words = {
        'rain', 'raining', 'rainy',
        'storm', 'stormy', 'thunder',
        'wind', 'windy', 'blowing',
        'forest', 'jungle', 'wood',
        'city', 'urban', 'traffic',
        'ocean', 'sea', 'waves', 'beach',
        'river', 'stream', 'waterfall',
        'fire', 'burning', 'crackling',
        'snow', 'snowing', 'snowy',
        'desert', 'mountain', 'valley',
        'shelter', 'roof', 'indoors', 'room',
        'street', 'park', 'garden',
    }
    
    # Emotional/mood words -> MUSIC
    emotion_words = {
        'sad', 'sadness', 'melancholy', 'depressed',
        'happy', 'happiness', 'joy', 'joyful',
        'scared', 'scary', 'frightened', 'fear', 'fearful',
        'suspense', 'suspenseful', 'tense', 'tension',
        'eerie', 'creepy', 'horror', 'horrifying',
        'emotional', 'emotional', 'feeling',
        'calm', 'peaceful', 'serene', 'tranquil',
        'excited', 'exciting', 'thrilling',
        'romantic', 'love', 'loving',
        'angry', 'anger', 'furious',
        'sudden', 'suddenly', 'abrupt',
    }
    
    # Check for verbs that produce sounds
    if word_lower in sound_verbs or pos_tag in ['VERB', 'VBG', 'VBD', 'VBP', 'VBZ']:
        # Generate a descriptive prompt
        if 'bark' in word_lower:
            return ("SFX", "dog barking")
        elif 'run' in word_lower or 'step' in word_lower or 'walk' in word_lower:
            return ("SFX", "footsteps running")
        elif 'scream' in word_lower or 'shout' in word_lower:
            return ("SFX", "person shouting")
        elif 'laugh' in word_lower:
            return ("SFX", "person laughing")
        elif 'knock' in word_lower:
            return ("SFX", "door knocking")
        elif 'crash' in word_lower or 'slam' in word_lower or 'bang' in word_lower:
            return ("SFX", f"{word_lower} sound")
        else:
            return ("SFX", f"{word_lower} sound effect")
    
    # Check for environmental/background sounds
    elif word_lower in environment_words:
        if 'rain' in word_lower:
            return ("AMBIENCE", "rain falling")
        elif 'storm' in word_lower or 'thunder' in word_lower:
            return ("AMBIENCE", "thunderstorm")
        elif 'wind' in word_lower:
            return ("AMBIENCE", "wind blowing")
        elif 'forest' in word_lower or 'jungle' in word_lower or 'wood' in word_lower:
            return ("AMBIENCE", "forest ambience")
        elif 'city' in word_lower or 'traffic' in word_lower or 'urban' in word_lower:
            return ("AMBIENCE", "city traffic")
        elif 'ocean' in word_lower or 'sea' in word_lower or 'wave' in word_lower:
            return ("AMBIENCE", "ocean waves")
        elif 'fire' in word_lower:
            return ("AMBIENCE", "fire crackling")
        elif 'shelter' in word_lower or 'roof' in word_lower:
            return ("AMBIENCE", "rain on roof")
        else:
            return ("AMBIENCE", f"{word_lower} ambience")
    
    # Check for emotional/mood words
    elif word_lower in emotion_words:
        if 'sad' in word_lower:
            return ("MUSIC", "sad emotional music")
        elif 'happy' in word_lower or 'joy' in word_lower:
            return ("MUSIC", "happy upbeat music")
        elif 'scared' in word_lower or 'scary' in word_lower or 'fear' in word_lower:
            return ("MUSIC", "scary horror music")
        elif 'suspense' in word_lower or 'tense' in word_lower:
            return ("MUSIC", "suspenseful music")
        elif 'eerie' in word_lower or 'creepy' in word_lower or 'horror' in word_lower:
            return ("MUSIC", "eerie suspense music")
        elif 'sudden' in word_lower or 'suddenly' in word_lower:
            return ("MUSIC", "dramatic stinger")
        elif 'calm' in word_lower or 'peaceful' in word_lower:
            return ("MUSIC", "calm soothing music")
        else:
            return ("MUSIC", f"{word_lower} emotional music")
    
    # Default: try to infer from POS tag
    elif pos_tag in ['ADJ', 'JJ', 'JJR', 'JJS']:
        # Adjectives might be emotional -> MUSIC
        return ("MUSIC", f"{word_lower} background music")
    elif pos_tag in ['NOUN', 'NN', 'NNS']:
        # Nouns might be environmental -> AMBIENCE
        return ("AMBIENCE", f"{word_lower} ambient sound")
    
    # If we can't classify, return None to skip
    return (None, None)

def _extract_audio_cues_nlp(story_text: str, speed_wps: float):
    """
    Uses spaCy NLP to extract audio cues from text.
    """
    if not nlp_available or nlp is None:
        # Fallback to simple extraction if NLP is not available
        return _extract_audio_cues_simple(story_text, speed_wps)
    doc = nlp(story_text)
    words = story_text.lower().split()
    total_words = len(words)
    total_duration_ms = math.ceil((total_words / speed_wps) * 1000)
    
    cues_to_play: List[AudioCue] = []
    current_weight_db = DEFAULT_WEIGHT_DB
    last_cue_index: Dict[str, int] = {"SFX": -1, "AMBIENCE": -1, "MUSIC": -1}
    
    # Build word index mapping for accurate position tracking
    word_positions = []
    char_idx = 0
    for word in words:
        # Find the word in the original text
        pos = story_text.lower().find(word, char_idx)
        if pos != -1:
            word_positions.append((pos, word))
            char_idx = pos + len(word)
        else:
            word_positions.append((char_idx, word))
            char_idx += len(word) + 1
    
    # Process each token with better position tracking
    processed_indices = set()  # Track which words we've processed to avoid duplicates
    
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        
        # Find which word index this token corresponds to
        token_start = token.idx
        word_idx = 0
        for i, (pos, word) in enumerate(word_positions):
            if pos <= token_start < pos + len(word):
                word_idx = i
                break
        
        # Skip if we've already processed this word
        if word_idx in processed_indices:
            continue
        
        current_time_ms = math.ceil((word_idx / speed_wps) * 1000)
        
        # Check for weight modifiers
        if token.text.lower() in MODIFIER_WORDS:
            current_weight_db = MODIFIER_WORDS[token.text.lower()]
            logger.debug(f"Word '{token.text}' at {current_time_ms}ms. Setting weight to: {current_weight_db}dB")
            processed_indices.add(word_idx)
            continue
        
        # Classify the token
        audio_type, audio_prompt = _classify_audio_type(
            token.text, 
            token.pos_,
            context=token.sent.text
        )
        
        if audio_type and audio_prompt:
            logger.info(f"Detected '{token.text}' ({token.pos_}) -> {audio_type}: '{audio_prompt}' at {current_time_ms}ms")
            
            # Create the AudioCue
            cue = AudioCue(
                id=word_idx,
                audio_class=audio_prompt,
                start_time_ms=current_time_ms,
                duration_ms=DEFAULT_SFX_DURATION_MS,
                weight_db=current_weight_db,
                audio_type=audio_type
            )
            
            # Handle smart duration for AMBIENCE/MUSIC
            if audio_type in ["AMBIENCE", "MUSIC"]:
                last_idx = last_cue_index[audio_type]
                if last_idx != -1:
                    prev_cue = cues_to_play[last_idx]
                    prev_cue.duration_ms = current_time_ms - prev_cue.start_time_ms
                    logger.debug(f"Updating previous '{prev_cue.audio_type}' cue to end at {current_time_ms}ms.")
                
                cue.duration_ms = total_duration_ms - current_time_ms
                last_cue_index[audio_type] = len(cues_to_play)
            
            cues_to_play.append(cue)
            current_weight_db = DEFAULT_WEIGHT_DB
            processed_indices.add(word_idx)
    
    return cues_to_play, total_duration_ms

def _extract_audio_cues_simple(story_text: str, speed_wps: float):
    """
    Simple fallback approach without spaCy - uses basic pattern matching.
    """
    words = story_text.lower().split()
    total_words = len(words)
    total_duration_ms = math.ceil((total_words / speed_wps) * 1000)
    
    cues_to_play: List[AudioCue] = []
    current_weight_db = DEFAULT_WEIGHT_DB
    last_cue_index: Dict[str, int] = {"SFX": -1, "AMBIENCE": -1, "MUSIC": -1}

    i = 0
    while i < total_words:
        current_time_ms = math.ceil((i / speed_wps) * 1000)
        word = words[i]
        
        # Check for weight modifiers
        if word in MODIFIER_WORDS:
            current_weight_db = MODIFIER_WORDS[word]
            logger.debug(f"Word '{word}' at {current_time_ms}ms. Setting weight to: {current_weight_db}dB")
            i += 1
            continue
        
        # Try to classify the word
        # Simple heuristic: check if it looks like a verb (ends with -ing, -ed, -s) or known keywords
        audio_type, audio_prompt = _classify_audio_type(word, "", context="")
        
        if audio_type and audio_prompt:
            logger.info(f"Detected '{word}' -> {audio_type}: '{audio_prompt}' at {current_time_ms}ms")
            
            cue = AudioCue(
                id=i,
                audio_class=audio_prompt,
                start_time_ms=current_time_ms,
                duration_ms=DEFAULT_SFX_DURATION_MS,
                weight_db=current_weight_db,
                audio_type=audio_type
            )
            
            if audio_type in ["AMBIENCE", "MUSIC"]:
                last_idx = last_cue_index[audio_type]
                if last_idx != -1:
                    prev_cue = cues_to_play[last_idx]
                    prev_cue.duration_ms = current_time_ms - prev_cue.start_time_ms
                    logger.debug(f"Updating previous '{prev_cue.audio_type}' cue to end at {current_time_ms}ms.")

                cue.duration_ms = total_duration_ms - current_time_ms
                last_cue_index[audio_type] = len(cues_to_play)
            
            cues_to_play.append(cue)
            current_weight_db = DEFAULT_WEIGHT_DB
        
        i += 1
    
    return cues_to_play, total_duration_ms

def _parse_gemini_cues(response_text) -> List[Dict]:
    """
    Parse Gemini response into a list of cue dicts.
    Handles raw lists, dicts with 'audio_cues'/'cues'/'results',
    or JSON strings (optionally wrapped in markdown).
    """
    # Already-parsed structures
    if isinstance(response_text, list):
        return response_text

    if isinstance(response_text, dict):
        cues = (
            response_text.get("audio_cues")
            or response_text.get("cues")
            or response_text.get("results")
        )
        return cues if isinstance(cues, list) else []

    # Fallback: string with JSON content (possibly in ```json``` fences)
    if not isinstance(response_text, str):
        return []

    json_str = response_text.replace("```json", "").replace("```", "").strip()
    json_match = re.search(r"\[[\s\S]*?\]", json_str)
    if json_match:
        json_str = json_match.group()

    try:
        parsed = json.loads(json_str)
    except Exception:
        logger.error("Failed to parse Gemini JSON response")
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        cues = parsed.get("audio_cues") or parsed.get("cues") or parsed.get("results")
        return cues if isinstance(cues, list) else []
    return []

def query_gemini(
    story_text: str,
    speed_wps: float,
    narrator_enabled: bool = True,
    movie_bgms_enabled: bool = True,
):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables. Set it with: export GEMINI_API_KEY='your-key'")
        return None
    try:
        model_name = model_config.decide_audio_model_name
        try:
            prompt_value = gemini_audio_prompt_with_narrator_without_movie_bgms.format_prompt(
                story_text=story_text,
                speed_wps=speed_wps,
            )
            prompt = prompt_value.to_string()
        except Exception as e:
            logger.error(f"Error formatting base audio prompt: {e}", exc_info=True)
            return None

        base_response = None
        try:
            base_response = query_llm(
                llm_name="gemini", model_name=model_name, prompt=prompt
            )
        except Exception as e:
            logger.error(f"\n\nModel {model_name} failed on base cues: {e}\n\n")
            return None

        if not base_response:
            logger.error("Gemini base audio prompt returned empty response")
            return None

        audio_cues: List[Dict] = _parse_gemini_cues(base_response)

        # -------- Stage 2: Optional movie BGMs, conditioned on existing cues --------
        if movie_bgms_enabled:
            try:
                movie_bgms_csv = read_movie_bgms_csv()
                if movie_bgms_csv is None:
                    logger.warning(
                        "Skipping movie BGM stage because metadata CSV is missing."
                    )
                    return audio_cues
                prompt_value = gemini_add_movie_bgms.format_prompt(
                    story_text=story_text,
                    speed_wps=speed_wps,
                    movie_bgms_csv=movie_bgms_csv,
                    already_added_audio_cues=audio_cues,
                )
                prompt = prompt_value.to_string()
            except Exception as e:
                logger.error(
                    f"Error formatting movie BGM prompt: {e}", exc_info=True
                )
                # If movie-BGM prompt fails, still return base cues
                return audio_cues

            movie_response = None
            try:
                movie_response = query_llm(
                    llm_name="gemini", model_name=model_name, prompt=prompt
                )
            except Exception as e:
                logger.error(
                    f"\n\nModel {model_name} failed on movie BGMs: {e}\n\n"
                )
                return audio_cues

            if movie_response:
                movie_cues = _parse_gemini_cues(movie_response)
                # movie prompt returns full merged list (base + MOVIE_BGM) or empty;
                # prefer movie_cues when non-empty, otherwise keep base audio_cues.
                if isinstance(movie_cues, list) and movie_cues:
                    audio_cues = movie_cues

        return audio_cues
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return None

def local_llm_fallback(story_text: str, speed_wps: float):
    """
    Uses local LLM to decide audio cues with precise timing based on reading speed.
    """
    logger.info(f"[DECIDER] Starting Local LLM Fallback...")

    # Extract cues directly from story using keyword matching
    story_lower = story_text.lower()
    words_list = story_text.split()

    words = story_text.split()
    total_duration_ms = math.ceil((len(words) / speed_wps) * 1000)

    sound_keywords = {
        'rain': ('rain falling', 'AMBIENCE', ['rain', 'raining', 'rainy', 'raindrop']),
        'dog': ('dog barking', 'SFX', ['dog', 'barking', 'bark', 'barked']),
        'run': ('footsteps running', 'SFX', ['run', 'ran', 'running', 'runs']),
        'shelter': ('shelter ambience', 'AMBIENCE', ['shelter', 'roof', 'indoors']),
        'loud': ('loud sound', 'SFX', ['loud', 'loudly']),
        'suddenly': ('dramatic stinger', 'MUSIC', ['suddenly', 'sudden', 'abrupt']),
        'started': ('sound starting', 'SFX', ['started', 'start', 'began']),
        'heard': ('sound heard', 'SFX', ['heard', 'hear', 'hearing']),
    }

    found_sounds = {}
    for key, (audio_class, audio_type, keywords) in sound_keywords.items():
        for i, word in enumerate(words_list):
            word_lower = word.lower().strip('.,!?;:')
            if any(kw in word_lower for kw in keywords):
                if key not in found_sounds or i < found_sounds[key]['word_index']:
                    found_sounds[key] = {
                        'audio_class': audio_class,
                        'audio_type': audio_type,
                        'word_index': i,
                        'keywords': keywords
                    }

    gemini_cues = []
    for key, sound_info in found_sounds.items():
        # Calculate timing based on reading speed
        word_idx = sound_info['word_index']
        start_ms = math.ceil((word_idx / speed_wps) * 1000)

        # Determine duration based on audio type
        if sound_info['audio_type'] == 'SFX':
            duration_ms = 2000  # Default 2 seconds for SFX
        elif sound_info['audio_type'] == 'AMBIENCE':
            # AMBIENCE continues until end of story
            duration_ms = max(1000, total_duration_ms - start_ms)
        else:  # MUSIC
            duration_ms = 5000  # Default 5 seconds for MUSIC

        weight_db = 0.0
        if 'loud' in story_lower and word_idx < len(words_list):
            for j in range(max(0, word_idx - 2), min(len(words_list), word_idx + 3)):
                if 'loud' in words_list[j].lower():
                    weight_db = 6.0
                    break

        gemini_cues.append({
            "audio_class": sound_info['audio_class'],
            "audio_type": sound_info['audio_type'],
            "word_index": word_idx,
            "start_time_ms": start_ms,
            "duration_ms": duration_ms,
            "weight_db": weight_db
        })

    if gemini_cues:
        logger.info(f"Fallback extracted {len(gemini_cues)} cues from keyword matching")
    return gemini_cues

def decide_audio_llm(story_text: str, speed_wps: float, narrator_enabled: bool = True, movie_bgms_enabled: bool = True):
    """
    Uses LLM to decide audio cues with precise timing based on reading speed.
    The LLM provides start_time_ms and duration_ms calculated from word positions.
    """
    print(f"[DECIDER] Starting Hybrid AI Analysis...")
    
    words = story_text.split()
    total_duration_ms = math.ceil((len(words) / speed_wps) * 1000)
    
    # Step A: Try Gemini first, then fallback to local LLM
    gemini_cues = query_gemini(story_text, speed_wps, narrator_enabled, movie_bgms_enabled)
    
    
    if not gemini_cues:
        gemini_cues = local_llm_fallback(story_text, speed_wps)


    final_cues: List[Cue] = []
    last_cue_idx = {"AMBIENCE": -1, "MUSIC": -1}
    index = 0

    # Sort cues by start_time_ms to ensure proper ordering
    gemini_cues.sort(key=lambda x: x.get("start_time_ms", x.get("word_index", 0) / speed_wps * 1000))

    for item in gemini_cues:
        a_type = str(item.get("audio_type", "SFX")).upper()
        if a_type not in SOUND_TYPES:
            a_type = "SFX"

        # Use LLM-provided timing if available, otherwise calculate from word_index
        start_ms = item.get("start_time_ms")
        if start_ms is None:
            # Fallback: calculate from word_index
            word_idx = item.get("word_index", 0)
            start_ms = math.ceil((word_idx / speed_wps) * 1000)
        else:
            # Ensure start_ms is within bounds
            start_ms = max(0, min(start_ms, total_duration_ms))
        
        # Use LLM-provided duration - the LLM should decide this
        duration_ms = item.get("duration_ms")
        if duration_ms is None:
            # Fallback: use default based on audio type (only if LLM didn't provide duration)
            logger.warning(f"LLM did not provide duration_ms for cue {index}, using fallback")
            if a_type == "SFX":
                duration_ms = DEFAULT_SFX_DURATION_MS
            elif a_type == "AMBIENCE":
                duration_ms = max(1000, total_duration_ms - start_ms)
            elif a_type == "NARRATOR":
                duration_ms = max(1000, total_duration_ms - start_ms)
            elif a_type == "MOVIE_BGM":
                duration_ms = 10000
            else:  # MUSIC
                duration_ms = 5000
        else:
            # LLM provided duration - use it directly, but ensure it's valid
            duration_ms = max(100, duration_ms)  # Minimum 100ms
        
        # Ensure duration doesn't exceed remaining time (safety check)
        max_allowed_duration = total_duration_ms - start_ms
        if duration_ms > max_allowed_duration:
            logger.warning(f"LLM-provided duration_ms {duration_ms} exceeds remaining time {max_allowed_duration}, clamping to {max_allowed_duration}")
            duration_ms = max_allowed_duration
        
        # Handle overlapping cues of the same type (AMBIENCE/MUSIC)
        # Only adjust if LLM didn't provide explicit durations
        if a_type in last_cue_idx:
            prev_idx = last_cue_idx[a_type]
            if prev_idx != -1:
                prev_cue = final_cues[prev_idx]
                # If previous cue's end time overlaps with current start, adjust previous cue
                prev_end_time = prev_cue.start_time_ms + prev_cue.duration_ms
                if prev_end_time > start_ms:
                    # Previous cue extends beyond current start - adjust it to end when current starts
                    # Only do this if the previous cue's duration wasn't explicitly set by LLM
                    # (We can't know this, so we'll adjust to prevent overlap)
                    prev_cue.duration_ms = max(100, start_ms - prev_cue.start_time_ms)
                    logger.debug(f"Adjusted previous {a_type} cue duration to prevent overlap")
            
            last_cue_idx[a_type] = len(final_cues)

        if a_type == "NARRATOR":
            narrator_cue = NarratorCue(
                id=index,
                story=item.get("story", ""),
                narrator_description=item.get("narrator_description", ""),
                audio_type=a_type,
                start_time_ms=start_ms,
                duration_ms=duration_ms
            )
            final_cues.append(narrator_cue)
            index += 1
            continue

        cue = AudioCue(
            id=index,
            audio_class=item.get("audio_class", "ambient texture"),
            audio_type=a_type,
            start_time_ms=start_ms,
            duration_ms=duration_ms,
            weight_db=item.get("weight_db", DEFAULT_WEIGHT_DB)
        )

        final_cues.append(cue)
        logger.info(f"Added cue: {cue}")
        index += 1

    logger.info(f"[DECIDER] Successfully generated {len(final_cues)} cinematic cues with LLM-provided timing.")
    return final_cues, total_duration_ms
   
def decide_audio_cues(story_text: str, speed_wps: float, narrator_enabled: bool = True, movie_bgms_enabled: bool = True):
    """
    Parses the story text using LLM and creates a timed list of AudioCues.
    Falls back to simple extraction if LLM fails.
    """
    logger.info("Starting audio decision process...")
    logger.info(f"Reading Speed: {speed_wps} words/sec")
    
    try:
        cues, total_duration = decide_audio_llm(story_text, speed_wps, narrator_enabled, movie_bgms_enabled)
        if not cues:
            logger.warning("LLM returned no cues, falling back to simple extraction...")
            raise Exception("Failed to generate audio cues with LLM")
            # Fallback to simple extraction if available
            # if nlp_available:
            #     cues, total_duration = _extract_audio_cues_nlp(story_text, speed_wps)
            # else:
            #     cues, total_duration = _extract_audio_cues_simple(story_text, speed_wps)
    except Exception as e:
        logger.error(f"Error in decide_audio_llm: {e}")
        logger.info("Falling back to simple extraction...")
        # Fallback to simple extraction
        if nlp_available:
            cues, total_duration = _extract_audio_cues_nlp(story_text, speed_wps)
        else:
            cues, total_duration = _extract_audio_cues_simple(story_text, speed_wps)
    
    logger.info(f"Finished parsing. Found {len(cues)} audio cues.")
    logger.info(f"\nCues: {cues}\n\n")
    logger.info(f"Total Duration: {total_duration}")
    return cues, total_duration

# if __name__ == "__main__":
#     story_text = "Suddenly rain started so i ran to shelter where i heard loud dog barking"
#     speed_wps = 100
#     cues, total_duration = decide_audio_cues(story_text, speed_wps)
#     print(cues)
#     print(total_duration)