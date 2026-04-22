from specialist_model.sfx_generator import sfx_generator_for_batch
from specialist_model.env_generator import environment_generator_for_batch
from specialist_model.emotional_generator import emotional_music_generator_for_batch
from specialist_model.text_to_speech_generator import text_to_speech_generator
from specialist_model.movie_bgm_retriver import movie_bgm_retriver
# Audio Type mapping
SPECIALIST_MAP = {
    "SFX": sfx_generator_for_batch,
    "AMBIENCE": environment_generator_for_batch,
    "MUSIC": emotional_music_generator_for_batch ,
    "NARRATOR": text_to_speech_generator,
    "MOVIE_BGM": movie_bgm_retriver,
}
