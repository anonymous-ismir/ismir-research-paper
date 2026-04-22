from langchain_core.prompts import PromptTemplate

gemini_audio_prompt = PromptTemplate(
    input_variables=["story_text", "speed_wps", "movie_bgms_csv"],
    template=(
        """Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

Story: "{story_text}"

Reading Speed: {speed_wps} words per second
Total Story Words: Count the words in the story
Total Duration (ms): Calculate as (total_words / {speed_wps}) * 1000

For each sound, you MUST provide:
- audio_class: detailed sound description for SoundGen AI
- audio_type: SFX (short sounds), AMBIENCE (background), or MUSIC (emotional)
- word_index: position (0-based) where the sound should start in the story
- start_time_ms: EXACT start time in milliseconds. Calculate as: (word_index / {speed_wps}) * 1000
- duration_ms: EXACT duration in milliseconds that YOU decide based on:
  * SFX: Decide duration (500-3000ms) based on the specific sound - a single bark might be 800ms, footsteps might be 2000ms, a door slam might be 1200ms
  * AMBIENCE: Decide duration based on story context - how long should this ambience play? Calculate from word_index to where it should end (next scene change, next AMBIENCE, or story end)
  * MUSIC: Decide duration based on emotional arc - how long should this musical element play? Consider the emotional moment and when it should fade (typically 2000-10000ms)
- weight_db: volume adjustment (-10.0 to 5.0, use 6.0 for "loud")

CRITICAL: You MUST provide a specific duration_ms value for EVERY audio cue. Do not leave it to be calculated later. Think about:
- For SFX: How long does this specific sound naturally last?
- For AMBIENCE: When does the scene/environment change in the story?
- For MUSIC: When does the emotional moment peak and fade?

Timing Calculation Rules:
1. start_time_ms = (word_index / {speed_wps}) * 1000 (round to nearest integer)
2. duration_ms = YOUR DECISION based on story context and sound type - provide the exact value
3. For overlapping sounds of the same type (e.g., two AMBIENCE cues), calculate when the first should end (typically when the second starts)
4. Ensure start_time_ms + duration_ms does not exceed total_duration_ms
5. Be precise - your duration_ms values will be used directly without modification

Try keeping as few Audio Cues as possible, not more than 3-4 Audio Cues.

Return ONLY a JSON array with these exact fields:
[
  {{"audio_class": "detailed sound description", "audio_type": "SFX|AMBIENCE|MUSIC", "word_index": 0, "start_time_ms": 0, "duration_ms": 2000, "weight_db": 0.0}}
]

"""
    ),
)

# gemini_audio_prompt_with_narrator_without_movie_bgms = PromptTemplate(
#     input_variables=["story_text", "speed_wps"],
#     template=(
#         """
# You are specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.
# Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

# Story: {story_text}
# Reading Speed: {speed_wps} words per second
# Total Story Words: Count the words in the story
# Cinematic Master Model: Story Analysis & Sync Prompt
# Role & Expertise: You are a Master Sound Designer and Narrative Director Agent. Your task is to perform a deep semantic analysis of the provided story to extract cinematic audio cues and direct a Narrator AI. You must ensure that the audio atmosphere perfectly syncs with the emotional arc and reading pace of the narrator.

# ### 1. NARRATOR AI DIRECTIVES The Narrator AI reads the entire story from word index 0 to the end. You must provide a "Narrator_Style" description that tells the AI exactly how to perform the reading based on the story's genre, mood, and tension.

# Narrator Persona Guidelines: Match the story's context to one of these reference styles or create a custom blend:

# Suspense/Horror: Low pitch, slower pace, breathless or whispering delivery.

# Action/Urgency: Faster pace, higher intensity, clear and sharp articulation.

# Serene/Nature: Calm, moderate pace, smooth intonation with subtle warmth.


# ### 1. NARRATOR DESCRIPTION You must provide a "Narrator_Style" description that tells the AI exactly how to perform the reading based on the story's genre, mood, and tension.

# include the term "very clear audio" to generate the highest quality audio, and "very noisy audio" for high levels of background noise
# Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks in speech
# The remaining speech features (gender, speaking rate, pitch and reverberation)

# Mini Model - Top 20 Speakers
# Speaker	Similarity Score
# Jon	0.908301
# Lea	0.904785
# Gary	0.903516
# Jenna	0.901807
# Mike	0.885742
# Laura	0.882666
# Lauren	0.878320
# Eileen	0.875635
# Alisa	0.874219
# Karen	0.872363
# Barbara	0.871509
# Carol	0.863623
# Emily	0.854932
# Rose	0.852246
# Will	0.851074

# add description like "A male speaker with a monotone and high-pitched voice is delivering his speech at a really low speed in a confined environment." or "A female speaker with a high-pitched voice is delivering her speech at a really fast speed in a noisy environment." or "Jon speaks with a low-pitched voice is delivering his speech at a really slow speed in a quiet environment." or "Lea speaker with a low-pitched voice is delivering her speech at a really fast speed in a noisy environment."


# ### 2. AUDIO CUE ENGINEERING You must identify any number of critical audio cues that ground the story in a professional soundscape.

# SFX (Short Effects): Punctuate specific actions (e.g., twig snapping, door slam). Duration: 500ms–3000ms.

# AMBIENCE (Environment): Constant background textures (e.g., rain, forest hum). Duration: From the trigger word to the next scene change or end of story.
 
# MUSIC (Emotional Score): Sets the heart of the scene (e.g., "Tense orchestral strings"). Duration: 2000ms–10000ms. This will be generated by model

# MOVIE_BGM (Movie Background Music): Sets the mood of the scene (e.g., "Heroic soundtrack"). Duration: 2000ms–10000ms. This will be retrieved from the movie bgms data. So it might not be excatly match to description or story context.

# ### 3. TIMING & SYNC MATH Precision is mandatory for "Multimodal Alignment".

# word_index: The 0-based position of the word that triggers the sound.

# start_time_ms: The start time of the sound.

# duration_ms: You must provide an exact value.

# For Overlaps: If a new AMBIENCE starts, the previous one of the same type should end at that start_time_ms.

# weight_db: volume adjustment (-15.0 to 6.0). 6.0 = "Defeaning/Loud".

# ### 5. OUTPUT CONSTRAINTS

# JSON ONLY: No conversational filler.

# Keep as many audio cues as you want to cover full story into audio. Focus on story context try to find audio sources, what music or sfx should be included in the story.

# Narrator Object: Include a single narrator_description at the root.

# ### EXAMPLE OF EXPECTED ANALYSIS Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

# JSON

# {{

#   "audio_cues": [
#     {{
#         "story": " The part of the story that the narrator will read with given descrpition , make sure to include pauses and breaks as per the narrator description
        
#         # you might break story into multiple parts and make seprate audio cues for each part
#         ", 
#         "narrator_description": "Tapan speaks at a moderate pace with a low-pitched, gravelly tone to convey mystery. Clear, close-sounding recording with a cold, detached emotional depth",
#         ##### audio que for narrator to know how to read the story
#         "audio_type": "NARRATOR",
#         "start_time_ms": 0,
#         "duration_ms": duration_ms,
#     }},
#     {{
#       "audio_class": "Heavy wooden door creaking open slowly with high-frequency friction",
#       "audio_type": "SFX",
#       "word_index": 1,
#       "start_time_ms": 500,
#       "duration_ms": 1500,
#       "weight_db": 2.0
#     }},
#     {{
#       "audio_class": "Heavy rain hitting glass window with distant thunder rumbles",
#       "audio_type": "AMBIENCE",
#       "word_index": 4,
#       "start_time_ms": 2000,
#       "duration_ms": 8000,
#       "weight_db": -5.0
#     }},
#     {{
#       "audio_class": "Dark cinematic suspense pad with low synth drones",
#       "audio_type": "MUSIC",
#       "word_index": 10,
#       "start_time_ms": 5000,
#       "duration_ms": 10000,
#       "weight_db": 0.0
#     }},
   
#   ]
# }}

# """
#     ),
# )



gemini_audio_prompt_with_narrator_without_movie_bgms = PromptTemplate(
    input_variables=["story_text", "speed_wps"],
    template=(
        """
You are a specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.

Story: {story_text}
Reading Speed: {speed_wps} words per second

Role & Expertise: You are a Master Sound Designer, Narrative Director, and Lead Mixing Engineer. Your task is to extract cinematic audio cues, direct a Narrator AI, and critically balance the volume (weight_db) of all overlapping sounds so the final mix is clear, professional, and not distorted.

### 1. NARRATOR AI DIRECTIVES
The Narrator AI reads the entire story. Provide a "Narrator_Style" description telling the AI how to perform based on genre, mood, and tension.
* Include terms like "very clear audio" or "very noisy audio" for background control.
* Use punctuation (commas) to add small breaks in speech.
* Specify speaker style, gender, speaking rate, pitch, and environment.
* Example Reference Speakers: Jon, Lea, Gary, Jenna, Mike, Laura.
* Example Description: "Jon speaks with a low-pitched voice, delivering his speech at a really slow speed in a quiet environment, pausing carefully for suspense."

### 2. AUDIO CUE ENGINEERING
Identify critical audio cues that ground the story:
* SFX (Short Effects): Punctuate specific actions (e.g., twig snapping). Duration: 500ms–3000ms.
* AMBIENCE (Environment): Constant background textures (e.g., rain). Duration: From trigger to scene change.
* MUSIC (Emotional Score): Sets the heart of the scene. Duration: 2000ms–10000ms. Generated by model.
* MOVIE_BGM: Retrieved atmospheric tracks. Duration: 2000ms–10000ms.

### 3. TIMING & SYNC MATH
Precision is mandatory for Multimodal Alignment.
* word_index: The 0-based position of the word that triggers the sound.
* start_time_ms: The exact start time.
* duration_ms: Exact duration. Overlapping ambiences of the same type must cut off the previous one.

### 4. THE MIX HIERARCHY & AUDIO DUCKING (CRITICAL)
You MUST evaluate all sounds playing at the same `start_time_ms`. Overlapping sounds add up and cause distortion. You must assign `weight_db` using this strict hierarchy:
1. NARRATOR (The Anchor): Always the clearest element. Assumed to be at 0.0 dB.
2. SFX (The Action): Loud but brief. Use -2.0 to -6.0 dB so it doesn't overpower the voice. (Exception: Explosions/Jump scares can peak at 2.0 to 4.0 dB).
3. MUSIC / MOVIE_BGM (The Emotion): Must sit UNDER the narrator. Use -10.0 to -15.0 dB.
4. AMBIENCE (The Void): Must sit at the very bottom of the mix. Use -18.0 to -24.0 dB.

AUDIO DUCKING RULE: If the Narrator is speaking, MUSIC and AMBIENCE must be pushed down to lower volumes (-15 to -20 dB). If the Narrator pauses for a long time, MUSIC can swell up slightly. NEVER put overlapping Ambience, Music, and SFX all above -5.0 dB at the same time.

### 5. OUTPUT CONSTRAINTS
* JSON ONLY. No conversational filler.
* Focus on story context to find what music or sfx should be included.

### EXAMPLE OF EXPECTED ANALYSIS 
Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

JSON
{{
  "audio_cues": [
    {{
        "story": "The door creaked open, ... Rain lashed against the window, ... as he stepped into the cold hall.", 
        "narrator_description": "Gary speaks at a moderate pace with a low-pitched, gravelly tone. Very clear audio in a quiet environment.",
        "audio_type": "NARRATOR",
        "start_time_ms": 0,
        "duration_ms": 7500
    }},
    {{
      "audio_class": "Heavy wooden door creaking open slowly",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": -4.0
    }},
    {{
      "audio_class": "Heavy rain hitting glass window with distant thunder",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -20.0
    }},
    {{
      "audio_class": "Dark cinematic suspense pad with low synth drones",
      "audio_type": "MUSIC",
      "word_index": 10,
      "start_time_ms": 5000,
      "duration_ms": 10000,
      "weight_db": -14.0
    }}
  ]
}}
"""
    ),
)


# gemini_add_movie_bgms = PromptTemplate(
#     input_variables=["story_text", "speed_wps", "movie_bgms_csv","already_added_audio_cues"],
#     template=(
#         """
# You are specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.
# Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

# Story: {story_text}
# Reading Speed: {speed_wps} words per second
# Total Story Words: Count the words in the story
# Cinematic Master Model: Story Analysis & Sync Prompt
# Role & Expertise: You are a expert movie background music retriever. You need to retrieve the movie background music that best fits the story and add it to the story to make it more cinematic.

# we have already added some audio cues to the story, so you need to add movie bgm to the story and make sure to not overlap with the already added audio cues.

# Already added audio cues: {already_added_audio_cues}


# ###################### ITS NOT AT ALL NECESSARY TO HAVE MOVIE BGM IN THE STORY. IF YOU FEEL IT IS NOT NECESSARY, YOU CAN SKIP IT. ######################

# For adding movie bgm, you need to follow the following rules:
# MOVIE_BGM (Movie Background Music): Sets the mood of the scene (e.g., "Heroic soundtrack"). Duration: 2000ms–10000ms. This will be retrieved from the movie bgms data. So it might not be excatly match to description or story context.

# ### 4. MOVIE BGM RETRIEVAL
# You must identify the movie bgm that best fits the story. Use the following movie bgms data to choose the best one:
# {movie_bgms_csv}

# Identify what best sound fits to the story from above data and add it to the story. like 

# audio_class: The audio path of the movie bgm you feel is best suited from the data.
# audio_type: "MOVIE_BGM".
# word_index: The 0-based position of the word that triggers the movie bgm.
# duration_ms: The duration of the movie bgm.
# start_time_ms: The start time of the movie bgm.
# weight_db: volume adjustment (-15.0 to 6.0). 6.0 = "Defeaning/Loud".

# Try to keep the duration_ms as close to the original movie bgm duration as possible.
# Also if you choose a movie bgm from the data, you must force the audio_type to be "MOVIE_BGM" and you need to turn off the Music cue by model, ie at a time either you have movie bgm or music cue, not both.

# ### IMPORTANT: Its not at all necessary to have movie bgm in the story. If you feel it is not necessary, you can skip it.###

# You might also remove some musical elements which you don't feel necessary to be included in story after addgint the movie bgm or you can return the already added audio cues as it is.

# ### 5. OUTPUT CONSTRAINTS

# JSON ONLY: No conversational filler.

# Keep as many audio cues as you want to cover full story into audio. Focus on story context try to find audio sources, what music or sfx should be included in the story.

# Narrator Object: Include a single narrator_description at the root.

# ### EXAMPLE OF EXPECTED ANALYSIS Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

# JSON

# {{

#   "audio_cues": [
#     {{
#         "story": " The part of the story that the narrator will read with given descrpition , make sure to include pauses and breaks as per the narrator description
        
#         # you might break story into multiple parts and make seprate audio cues for each part
#         ", 
#         "narrator_description": "Tapan speaks at a moderate pace with a low-pitched, gravelly tone to convey mystery. Clear, close-sounding recording with a cold, detached emotional depth",
#         ##### audio que for narrator to know how to read the story
#         "audio_type": "NARRATOR",
#         "start_time_ms": 0,
#         "duration_ms": duration_ms,
#     }},
#     must include already Added audio cues in the story.
#     and follow below to add movie bgms
#      {{
#       "audio_class": "Audio file path of the movie bgm you feel is best suited from the data.",
#       "audio_type": "MOVIE_BGM", # FORCE TO BE MOVIE_BGM if you take movie bgm from the data
#       "start_time_ms": 10000,# best start time according to you when the movie bgm should start
#       "duration_ms": 10000,# best duration according to you when the movie bgm should end, try to keep it as close to the original movie bgm duration as possible.
#       "weight_db": 0.0 # best weight db/loudness according to you when the movie bgm should be played
#     }},

   
#   ]
# }}

# """
#     ),
# )


gemini_add_movie_bgms = PromptTemplate(
    input_variables=["story_text", "speed_wps", "movie_bgms_csv", "already_added_audio_cues"],
    template=(
        """
You are a Master Music Supervisor and Lead Mixing Engineer for cinematic audio.

Story: {story_text}
Reading Speed: {speed_wps} words per second

Available Movie Background Music Database (CSV format):
{movie_bgms_csv}

Currently Existing Audio Cues:
{already_added_audio_cues}

### 1. YOUR MISSION: THE SYNC PLACEMENT
You must evaluate if any track from the Available Movie Background Music Database strongly fits the emotional arc of the story. 

* SCENARIO A (STRONG MATCH): If a track's description/mood perfectly aligns with the story, add it to the cue list. Its `audio_type` MUST be "MOVIE_BGM". If you do this, you MUST DELETE any existing "audio_type": "MUSIC" cues from the existing list so the soundtracks do not clash.
* SCENARIO B (NO MATCH / SKIP): If none of the available tracks fit the specific mood of the story, DO NOT force it. Silence or ambient sound is better than the wrong music. Simply return the `already_added_audio_cues` exactly as they are without adding a MOVIE_BGM.

### 2. THE MIX HIERARCHY & AUDIO DUCKING (CRITICAL)
If you decide to add a MOVIE_BGM, you must calculate its `weight_db` so it does not distort or drown out the other sounds.
1. NARRATOR: The anchor, assumed to be at 0.0 dB.
2. SFX: Action sounds sit at -2.0 to -6.0 dB.
3. MOVIE_BGM: Must sit UNDER the narrator. Assign a weight of -10.0 to -15.0 dB. 
4. AMBIENCE: Sits at the bottom at -18.0 to -24.0 dB.

### 3. FORMATTING THE MOVIE_BGM CUE
If adding a track, use this exact format:
- "audio_class": The exact audio path or identifier from the CSV.
- "audio_type": "MOVIE_BGM"
- "start_time_ms": The logical start time based on the story's emotional shift.
- "duration_ms": Try to keep it as close to the original CSV duration as possible, unless the scene ends earlier.
- "weight_db": Calculated based on the Mix Hierarchy (e.g., -14.0).

### 4. OUTPUT CONSTRAINTS
* JSON ONLY. No markdown formatting, no conversational filler.
* Output a single JSON object with the root key "audio_cues" containing the final array of dictionaries.

### EXAMPLES OF DECISION MAKING

EXAMPLE 1: A PERFECT MATCH IS FOUND
Decision: A heroic track exists in the CSV for a battle scene.
JSON Output:
{{
  "audio_cues": [
    {{ "audio_type": "NARRATOR", "start_time_ms": 0, "duration_ms": 5000, "narrator_description": "..." }},
    {{ "audio_type": "SFX", "audio_class": "Sword clash", "start_time_ms": 1000, "duration_ms": 1000, "weight_db": -4.0 }},
    {{ 
      "audio_class": "path/to/heroic_battle_theme.wav", 
      "audio_type": "MOVIE_BGM", 
      "start_time_ms": 0, 
      "duration_ms": 15000, 
      "weight_db": -12.0 
    }}
  ]
}}
*(Note: Any original generic "MUSIC" cues were deleted and replaced by the MOVIE_BGM)*

EXAMPLE 2: NO MATCH FOUND (SKIPPING)
Decision: The story is a quiet, tense thriller, but the CSV only contains happy, upbeat comedy tracks. The agent decides to skip.
JSON Output:
{{
  "audio_cues": [
    {{ "audio_type": "NARRATOR", "start_time_ms": 0, "duration_ms": 5000, "narrator_description": "..." }},
    {{ "audio_type": "AMBIENCE", "audio_class": "Quiet wind", "start_time_ms": 0, "duration_ms": 5000, "weight_db": -20.0 }}
  ]
}}
*(Note: The array is returned exactly as it was provided in the prompt, with no MOVIE_BGM added)*
"""
    ),
)

prompt_to_fill_missing_audio_cues = PromptTemplate(
    input_variables=["story_text", "audio_cues", "movie_bgms_csv"],
    template=(
        """
You are an expert audio designer and storyteller assistant, specializing in identifying and engineering cinematic soundscapes for narrative enhancement. Your task is to scrutinize the given story and its currently provided audio cues for any missing critical sound elements and meticulously engineer supplementary cues only when genuinely necessary. Your objective is not simply to fill space, but to elevate immersion and narrative clarity where audio is lacking.

Below are the resources provided to you:
* Story Text: {story_text}
* Pre-existing Audio Cues: {audio_cues}
* Reference Movie BGM Data (CSV): {movie_bgms_csv}

## 1. DETAILED GAP ANALYSIS
Thoroughly read and analyze the story in the context of the present audio cues. Methodically identify any narrative moment or sensory detail (such as dramatic actions, atmospheric transitions, or emotional cues) that is not already reflected in the provided cues. Carefully ask yourself:
- Are any important actions, impacts, or narrative turns missing a SFX cue?
- Is the ambience—such as environment, weather, or location—properly represented for every part of the story? Are there silent passages where there should be a textural background?
- Does the scene lack the proper emotional underpinning through MUSIC or, where applicable, a MOVIE_BGM from the data?
- Are crescendos, moments of suspense, or transitions (scene changes) insufficiently scored by music or background?

> Only if you uncover an audible gap, generate additional audio cues strictly for those gaps.
> If the current cues fully support the story—including all significant SFX, AMBIENCE, and MUSIC/MOVIE_BGM—do NOT create new cues. In this case, return an empty JSON array.

## 2. STRICT OUTPUT POLICY
Return **only** the newly engineered cues that are missing in the existing array. **Do not repeat or summarize any of the existing audio cues**. You must not return conversational filler or markdown—your only response is a JSON object as described below.

## 3. AUDIO CUE CONSTRUCTION & PARAMETERS

- **SFX (Sound Effects)**: Use for distinct narrative actions/impacts (e.g., footsteps, doors, objects, dramatic events). Ensure each SFX is justified by a clear textual trigger. Typical duration: 500–3000ms.
- **AMBIENCE**: Add only for persistent background environments (e.g., rain, crowd murmur, forest sounds). Duration should begin at the relevant trigger and end at the next major change or scene shift. Avoid overlapping identical types—if a new AMBIENCE begins, terminate the previous instance at its start.
- **MUSIC (Emotional Score)**: Used to indicate emotional/scenic background when MOVIE_BGM is inappropriate, unavailable, or contextually unfitting. Do not use if a valid MOVIE_BGM is present for the same time span.
- **MOVIE_BGM (Movie Background Music)**: Select and insert the single most context-appropriate BGM track from the given CSV data if the scene supports its use. The chosen track does not need to perfectly fit the narrative, but it must not distract or conflict with the story context. Its duration should match or reasonably fit the source file, and you must never play both MUSIC and MOVIE_BGM simultaneously.

All cues must specify:
- `audio_class`: Either the detailed description (“wooden door creak”, “orchestral suspense pad”) or the file path from the MOVIE_BGM data when applicable.
- `audio_type`: One of `"SFX"`, `"AMBIENCE"`, `"MUSIC"`, `"MOVIE_BGM"`.
- `word_index`: Index of the very first word in the story text that triggers this cue.
- `start_time_ms`, `duration_ms`: Exact millisecond values for cue timing and span.
- `weight_db`: Follow the strict hierarchy below.

### MIXING HIERARCHY—STRICTLY ENFORCED
- SFX (Action/Impact): -2.0 to -6.0 dB (except: huge explosions may reach +2.0 dB).
- MUSIC or MOVIE_BGM (Mood/Emotion): -10.0 to -15.0 dB. Ensure only one is present at any given moment.
- AMBIENCE: -18.0 to -24.0 dB.

## 4. MOVIE BGM SELECTION & CONFLICTS
- If using a MOVIE_BGM (selected via `audio_class` from CSV), strictly forbid any MUSIC cues at overlapping times.
- If no MOVIE_BGM contextually fits, skip it entirely.
- Movie BGM cues take precedence if their scene match is reasonable.
- When choosing MOVIE_BGM from the CSV data, copy the `audio_class` (file path) as provided and keep the duration as close to the original as practical.

## 5. NARRATOR OBJECT INTEGRATION
For every missing section that should be read aloud, include a `"narrator_description"` (defining speaker voice, delivery style, prosody, etc.) and, where helpful, suggest how to break narration into logical parts for clarity or dramatic pacing.

## 6. OUTPUT—STRICT, CLEAN JSON
- Respond ONLY with a single JSON object containing a single key `"audio_cues"` pointing to a list of new cue objects OR an empty list if nothing is missing.
- No other text, explanation, markdown, or conversational content may appear.
- All required fields must be present on every cue.
- If no cues are missing, return: `{{"audio_cues": []}}`

---

### EXAMPLE SCENARIO

_Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall."_

JSON:
{{
  "audio_cues": [
    {{
      "story": "(The part of the story to be narrated with embedded pauses and emphasis),",
      "narrator_description": "male, monotone, high-pitched, slow-paced delivery in a confined, echoey space.",
      "audio_type": "NARRATOR",
      "start_time_ms": 0,
      "duration_ms": 4000
    }},
    {{
      "audio_class": "path/to/thriller_bgm.wav",
      "audio_type": "MOVIE_BGM",
      "word_index": 8,
      "start_time_ms": 3000,
      "duration_ms": 9000,
      "weight_db": -13.0
    }},
    {{
      "audio_class": "Heavy wooden door creaking open slowly with high-frequency friction",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": -2.0
    }},
    {{
      "audio_class": "Intense rain hitting glass",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -19.0
    }}
  ]
}}
"""
    ),
)



alignment_prediction_prompt = PromptTemplate(
    input_variables=["story_prompt", "audio_classes", "whisper_json"],
    template=(
        """
        You are a helpful assistant that predicts the alignment of the audio classes with the story prompt.
        I have a Story text as : {story_prompt}
        I have generated Audio Cues: {audio_classes}
        I have generated Whisper JSON: {whisper_json}
        
        You need to predict the alignment of the audio classes with the story prompt.
        You need to return the alignment in the following format:
        
          "audio_cues": [
          {{
            "id": <same id as the audio class>,
            "audio_class": <same as the audio class>,
            "audio_type": <same as the audio type>,
            "start_time_ms": 5000, # starting time in milliseconds
            "duration_ms": 10000, # duration in milliseconds
            "weight_db": 0.0,
            "fade_ms": 500
          }}, 
          ]
          
          example
            "story_prompt": "A helmet-clad soldier cautiously navigates a grimy, dimly lit urban corridor before being brutally ambushed by a bloodied operative, who then, protecting a young boy, plunges into a chaotic close-quarters gunfight against multiple assailants."
            "whisper.json":
            [{{"word": "A", "start": 0.0, "end": 0.18}}, {{"word": "helmet,", "start": 0.18, "end": 1.42}}, {{"word": "clad", "start": 1.42, "end": 1.9}}, {{"word": "soldier", "start": 1.9, "end": 2.34}}, {{"word": "cautiously", "start": 2.34, "end": 3.02}}, {{"word": "navigates", "start": 3.02, "end": 3.6}}, {{"word": "a", "start": 3.6, "end": 3.72}}, {{"word": "grimy,", "start": 3.72, "end": 4.34}}, {{"word": "dimly", "start": 4.34, "end": 4.62}}, {{"word": "lit", "start": 4.62, "end": 4.86}}, {{"word": "urban", "start": 4.86, "end": 5.22}}, {{"word": "corridor", "start": 5.22, "end": 5.62}}, {{"word": "before", "start": 5.62, "end": 6.22}}, {{"word": "being", "start": 6.22, "end": 6.54}}, {{"word": "brutally", "start": 6.54, "end": 6.92}}, {{"word": "ambushed", "start": 6.92, "end": 7.6}}, {{"word": "by", "start": 7.6, "end": 7.7}}, {{"word": "a", "start": 7.7, "end": 7.86}}, {{"word": "bloodied", "start": 7.86, "end": 8.14}}, {{"word": "operative,", "start": 8.14, "end": 9.42}}, {{"word": "who", "start": 9.42, "end": 9.48}}, {{"word": "then", "start": 9.48, "end": 9.7}}, {{"word": "protecting", "start": 9.7, "end": 10.22}}, {{"word": "a", "start": 10.22, "end": 10.44}}, {{"word": "young", "start": 10.44, "end": 10.64}}, {{"word": "boy", "start": 10.64, "end": 10.98}}, {{"word": "plunges", "start": 10.98, "end": 11.96}}, {{"word": "into", "start": 11.96, "end": 12.2}}, {{"word": "a", "start": 12.2, "end": 12.38}}, {{"word": "chaotic", "start": 12.38, "end": 12.7}}, {{"word": "close", "start": 12.7, "end": 13.1}}, {{"word": "-quarters", "start": 13.1, "end": 13.48}}, {{"word": "gunfight", "start": 13.48, "end": 14.04}}, {{"word": "against", "start": 14.04, "end": 14.56}}, {{"word": "multiple", "start": 14.56, "end": 15.1}}, {{"word": "assailants.", "start": 15.1, "end": 16.0}}]
            "audio_cues": [
              {{ "id": 1, "audio_class": "Distant urban street ambience", "audio_type": "AMBIENCE", "starting_time": 0.0, "duration": 16.0, "weight_db": -35.0 }},
              {{ "id": 2, "audio_class": "Tense synth drone with subtle rhythmic percussion", "audio_type": "MUSIC", "starting_time": 0.0, "duration": 8.8, "weight_db": -25.0 }},
              {{ "id": 3, "audio_class": "Heavy tactical footsteps and gear rustle", "audio_type": "AMBIENCE", "starting_time": 0.0, "duration": 8.5, "weight_db": -20.0 }},
              {{ "id": 4, "audio_class": "Brutal melee combat impacts and vocal grunts", "audio_type": "SFX", "starting_time": 8.7, "duration": 6.5, "weight_db": -10.0 }},
              {{ "id": 5, "audio_class": "Pistol slide rack and reload click", "audio_type": "SFX", "starting_time": 20.0, "duration": 0.5, "weight_db": -12.0 }},
              {{ "id": 6, "audio_class": "Deep male voice (low dialogue, 'Come on')", "audio_type": "NARRATOR", "starting_time": 20.5, "duration": 0.5, "weight_db": -18.0 }},
              {{ "id": 7, "audio_class": "Rapid gunfire, body impacts, and close-quarters combat SFX", "audio_type": "SFX", "starting_time": 25.9, "duration": 4.1, "weight_db": -7.0 }},
              {{ "id": 8, "audio_class": "Aggressive percussive action music swell", "audio_type": "MUSIC", "starting_time": 25.5, "duration": 4.5, "weight_db": -10.0 }}
            ],
            the output should be something like this:
            "audio_cues": [
              {{ "id": 1, "audio_class": "Distant urban street ambience", "audio_type": "AMBIENCE", "start_time_ms": 0, "duration_ms": 8000, "weight_db": -35.0, "fade_ms": 500 }},
              {{ "id": 2, "audio_class": "Tense synth drone with subtle rhythmic percussion", "audio_type": "MUSIC", "start_time_ms": 0, "duration_ms": 4690, "weight_db": -25.0, "fade_ms": 500 }},
              {{ "id": 3, "audio_class": "Heavy tactical footsteps and gear rustle", "audio_type": "AMBIENCE", "start_time_ms": 0, "duration_ms": 4250, "weight_db": -20.0, "fade_ms": 500 }},
              {{ "id": 4, "audio_class": "Brutal melee combat impacts and vocal grunts", "audio_type": "SFX", "start_time_ms": 4690, "duration_ms": 1810, "weight_db": -10.0, "fade_ms": 500 }},
              {{ "id": 5, "audio_class": "Pistol slide rack and reload click", "audio_type": "SFX", "start_time_ms": 10600, "duration_ms": 500, "weight_db": -12.0, "fade_ms": 500 }},
              {{ "id": 6, "audio_class": "Deep male voice (low dialogue, 'Come on')", "audio_type": "NARRATOR", "start_time_ms": 11000, "duration_ms": 500, "weight_db": -18.0, "fade_ms": 500 }},
              {{ "id": 7, "audio_class": "Rapid gunfire, body impacts, and close-quarters combat SFX", "audio_type": "SFX", "start_time_ms": 13810, "duration_ms": 2190, "weight_db": -7.0, "fade_ms": 500 }},
              {{ "id": 8, "audio_class": "Aggressive percussive action music swell", "audio_type": "MUSIC", "start_time_ms": 15000, "duration_ms": 1000, "weight_db": -10.0, "fade_ms": 500 }}
            ],
        """
    ),
)