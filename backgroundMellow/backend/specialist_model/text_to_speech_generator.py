import torch
from transformers.trainer_utils import set_seed
import logging
from helper.lib import get_model
from Variable.configurations import model_config
logger = logging.getLogger(__name__)

def text_to_speech_generator(prompt: str, description: str):
    """Generates a text to speech audio."""
    logger.info(f"Generating: '{prompt}' with description: '{description}'")

    # models = ParlerTTSModel.get_instance()
    # model = models["model"]
    # tokenizer = models["tokenizer"]
    # description_tokenizer = models["description_tokenizer"]

    # description_input_ids = description_tokenizer(description, return_tensors="pt")
    # prompt_input_ids = tokenizer(prompt, return_tensors="pt")
    # set_seed(42)
    # generation = model.generate(
    #     input_ids=description_input_ids.input_ids,
    #     attention_mask=description_input_ids.attention_mask,
    #     prompt_input_ids=prompt_input_ids.input_ids,
    #     prompt_attention_mask=prompt_input_ids.attention_mask,
    # )
    # if isinstance(generation, torch.Tensor):
    #     audio_arr = generation.cpu().numpy().squeeze()
    # else:
    #     audio_tensor = getattr(generation, "sequences", None) or getattr(
    #         generation, "audio", None
    #     )
    #     if audio_tensor is not None and isinstance(audio_tensor, torch.Tensor):
    #         audio_arr = audio_tensor.cpu().numpy().squeeze()
    #     else:
    #         audio_arr = generation.cpu().numpy().squeeze()  # type: ignore[union-attr]
    # return audio_arr
    model = get_model(model_config.narrator_model_name)
    audio_arr = model.generate(prompt, description)
    return audio_arr




# if __name__ == "__main__":
#     prompt = "Hello how are you?"
#     description = "Sita speaks at a fast pace with a slightly low-pitched voice , captured clearly in a close-sounding environment with excellent recording quality"

#     audio_arr = text_to_speech_generator(prompt, description)
#     import soundfile as sf
#     sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
#     print("Audio saved to parler_tts_out.wav")
