import os
import sys
import json
from typing import List
import logging

logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)

cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)

from backend.helper.audio_conversions import audio_to_base64
from model.dl_based_alignment_predictor import CinematicMixPredictor
from model.word_aligner import WordAligner
from Variable.dataclases import AudioCue
from Utils.prompts import alignment_prediction_prompt
from Utils.llm import query_llm
from Variable.configurations import model_config
from superimposition_model.llm_dsp_weight_predictor import load_predictor

cinematic_mix_predictor = CinematicMixPredictor()
word_aligner_ins = WordAligner()
_blend_alpha_predictor = None


def _get_blend_alpha_predictor():
    global _blend_alpha_predictor
    if _blend_alpha_predictor is None:
        _blend_alpha_predictor = load_predictor(train_if_missing=True)
    return _blend_alpha_predictor


def predict_from_llm(
    story_prompt: str,
    audio_classes: List[AudioCue],
    whisper_json: str,
) -> List[AudioCue]:
    """Predict alignment of audio cues using the LLM."""
    # Format audio_classes for the prompt (e.g. list of dicts or string)
    audio_classes_repr = [
        {
            "id": c.id,
            "audio_class": c.audio_class,
            "audio_type": c.audio_type,
            "start_time_ms": c.start_time_ms,
            "duration_ms": c.duration_ms,
            "weight_db": c.weight_db,
            "fade_ms": getattr(c, "fade_ms", 500),
        }
        for c in audio_classes
    ]
    
    
    logger.info(f"Audio classes repr: {audio_classes_repr}\n\n{whisper_json}")
    whisper_json_str = json.dumps(whisper_json)
    prompt = alignment_prediction_prompt.format(
        story_prompt=story_prompt,
        audio_classes=json.dumps(audio_classes_repr, indent=2),
        whisper_json=whisper_json_str,
    )
    response = query_llm(llm_name="gemini", model_name="gemini-2.5-flash", prompt=prompt)
    if not response:
        raise ValueError("LLM returned empty response for alignment prediction")
    
    logger.info("LLM audio cues response: %s", response)
    if isinstance(response, (dict, list)):
        data = response
    else:
        data = json.loads(response)
    items = data.get("audio_cues", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = [items]
    audio_cues = [AudioCue(**item) for item in items]
    logger.info(f"Audio cues: {audio_cues}")
    return audio_cues


def predict_from_dsp(
    story_prompt: str,
    audio_classes: List[AudioCue],
    whisper_json: List[dict],
) -> List[AudioCue]:
    """
    Predict alignment using the DL-based DSP model (CinematicMixPredictor.predict_from_dsp).
    """
    class_names = [c.audio_class for c in audio_classes]
    raw_results = cinematic_mix_predictor.predict_from_dsp(
        story_prompt=story_prompt,
        audio_classes=class_names,
        whisper_json=whisper_json,
    )
   
    cues: List[AudioCue] = []
    for i, cue in enumerate(audio_classes):
        row = raw_results[i] if i < len(raw_results) else [0, 0.0, 2000]
        start_time_ms = int(round(row[0]))
        weight_db = float(row[1])
        duration_ms = int(round(row[2]))
        fade_ms = getattr(cue, "fade_ms", 500)
        cues.append(
            AudioCue(
                id=cue.id,
                audio_type=cue.audio_type,
                audio_class=cue.audio_class,
                start_time_ms=start_time_ms,
                duration_ms=duration_ms,
                weight_db=weight_db,
                fade_ms=fade_ms,
            )
        )
        
    logger.info(f"Cues: {cues}")
    if not cues:
        raise ValueError("DSP returned empty response for alignment prediction")
    return cues


def predict_avg_llm_and_dsp_align(
    story_prompt: str,
    audio_classes: List[AudioCue],
    whisper_json_str: str,
    whisper_json_list: List[dict],
) -> List[AudioCue]:
    """
    Combine LLM and DSP predictions: blend according to a mix factor (e.g. 0.5 = 50% LLM, 50% DSP).
    """
    llm_cues = predict_from_llm(
        story_prompt=story_prompt,
        audio_classes=audio_classes,
        whisper_json=whisper_json_str,
    )
    dsp_cues = predict_from_dsp(
        story_prompt=story_prompt,
        audio_classes=audio_classes,
        whisper_json=whisper_json_list,
    )
    blended: List[AudioCue] = []
    alpha_predictor = _get_blend_alpha_predictor()
    for i, llm_cue in enumerate(llm_cues):
        dsp_cue = dsp_cues[i] if i < len(dsp_cues) else llm_cue
        alpha = alpha_predictor.predict_alpha(
            audio_type=getattr(llm_cue, "audio_type", ""),
            audio_class=getattr(llm_cue, "audio_class", ""),
        )
        blended.append(
            AudioCue(
                id=llm_cue.id,
                audio_type=llm_cue.audio_type,
                audio_class=llm_cue.audio_class,
                start_time_ms=int((1 - alpha) * llm_cue.start_time_ms + alpha * dsp_cue.start_time_ms),
                duration_ms=int((1 - alpha) * llm_cue.duration_ms + alpha * dsp_cue.duration_ms),
                weight_db=(1 - alpha) * llm_cue.weight_db + alpha * dsp_cue.weight_db,
                fade_ms=int((1 - alpha) * getattr(llm_cue, "fade_ms", 500) + alpha * getattr(dsp_cue, "fade_ms", 500)),
            )
        )
    return blended


def align_predictor(
    story_prompt: str,
    audio_classes: List[AudioCue],
    narrator_audio_base64: str,
) -> List[AudioCue]:
    """
    Predict alignment of audio cues according to model config flags.
    """
    whisper_list = word_aligner_ins.get_timestamps_from_base64(narrator_audio_base64)
    whisper_json_str = json.dumps(whisper_list)

    if model_config.use_avg_llm_and_dsp_to_predict_align:
        llm_cues = predict_from_llm(
            story_prompt=story_prompt,
            audio_classes=audio_classes,
            whisper_json=whisper_json_str,
        )
        dsp_cues = predict_from_dsp(
            story_prompt=story_prompt,
            audio_classes=audio_classes,
            whisper_json=whisper_list,
        )
        logger.info("LLM cues: %s", llm_cues)
        logger.info("DSP cues: %s", dsp_cues)
        averaged: List[AudioCue] = []
        for i, llm_cue in enumerate(llm_cues):
            dsp_cue = dsp_cues[i] if i < len(dsp_cues) else llm_cue
            averaged.append(
                AudioCue(
                    id=llm_cue.id,
                    audio_type=llm_cue.audio_type,
                    audio_class=llm_cue.audio_class,
                    start_time_ms=int((llm_cue.start_time_ms + dsp_cue.start_time_ms) / 2),
                    duration_ms=int((llm_cue.duration_ms + dsp_cue.duration_ms) / 2),
                    weight_db=(llm_cue.weight_db + dsp_cue.weight_db) / 2.0,
                    fade_ms=int((getattr(llm_cue, "fade_ms", 500) + getattr(dsp_cue, "fade_ms", 500)) / 2),
                )
            )
        return averaged

    elif model_config.use_llm_to_predict_align:
        return predict_from_llm(
            story_prompt=story_prompt,
            audio_classes=audio_classes,
            whisper_json=whisper_json_str,
        )
   
    elif model_config.use_dsp_to_predict_align:
        return predict_from_dsp(
            story_prompt=story_prompt,
            audio_classes=audio_classes,
            whisper_json=whisper_list,
        )
    
    elif model_config.use_dl_based_llm_and_dsp_alignment_predictor:
        return predict_avg_llm_and_dsp_align(
            story_prompt=story_prompt,
            audio_classes=audio_classes,
            whisper_json_str=whisper_json_str,
            whisper_json_list=whisper_list,
        )
    
    raise ValueError("Invalid alignment prediction model configuration: no alignment method enabled.")




# # Testing
# if __name__ == "__main__":
#     from model.parlerTTSModel import ParlerTTSModel
#     parler_tts_model_ins = ParlerTTSModel()

    
#     story_prompt = "Deadpool, in a frenetic close-up, executes a swift, explosive strike that culminates in a flash of impact amidst a chaotic backdrop."
    
#     audio_classes = [AudioCue(id=1, audio_class="Orchestral percussive hit with brief brass accent", audio_type="SFX", start_time_ms=0, duration_ms=800, weight_db=-8, fade_ms=500), AudioCue(id=2, audio_class="Sharp, concussive impact sound effect", audio_type="SFX", start_time_ms=200, duration_ms=400, weight_db=-7, fade_ms=500), AudioCue(id=3, audio_class="Fast blade whoosh/slice", audio_type="SFX", start_time_ms=0, duration_ms=200, weight_db=-15, fade_ms=500), AudioCue(id=4, audio_class="Quick debris fall / light rumble", audio_type="SFX", start_time_ms=400, duration_ms=600, weight_db=-20, fade_ms=500)]
    
#     description = "Deadpool, in a frenetic close-up, executes a swift, explosive strike that culminates in a flash of impact amidst a chaotic backdrop."
    
#     narrator_audio_segment  = parler_tts_model_ins.generate(story_prompt, description)
#     narrator_audio_base64 = audio_to_base64(narrator_audio_segment)
#     formatted_timestamps =  word_aligner_ins.get_timestamps_from_base64(narrator_audio_base64)
#     logger.info(formatted_timestamps)
#     whisper_json_str = json.dumps(formatted_timestamps)
#     logger.info("DSP prediction:")
#     logger.info(predict_from_dsp(story_prompt, audio_classes, formatted_timestamps))
#     logger.info("Averaged prediction:")
#     logger.info(predict_avg_llm_and_dsp_align(story_prompt, audio_classes,  whisper_json_str, formatted_timestamps))
    # logger.info(predict_from_llm(story_prompt, audio_classes, whisper_json_str))
