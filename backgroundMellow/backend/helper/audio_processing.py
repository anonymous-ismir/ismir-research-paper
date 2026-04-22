# function to stretch compression and stretch expansion
from pydub import AudioSegment
from pydub.effects import speedup

def slowdown(audio_segment: AudioSegment, stretch_factor: float):
    new_frame_rate = int(audio_segment.frame_rate * stretch_factor)
    slowed = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": new_frame_rate})
    return slowed.set_frame_rate(audio_segment.frame_rate)

def stretch_compression(audio_segment: AudioSegment, stretch_factor: float):
    return speedup(audio_segment, stretch_factor)

def stretch_expansion(audio_segment: AudioSegment, stretch_factor: float):
    return slowdown(audio_segment, stretch_factor)
