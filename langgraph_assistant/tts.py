import torch
import sounddevice as sd

language = 'en'
model_id = 'v3_en'
sample_rate = 48000
speaker = 'en_0'  # en_16 en_18
device = torch.device('cpu')

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu


def text_to_speech(text: str):
    audio = model.apply_tts(text=text,
                            speaker=speaker,
                            sample_rate=sample_rate)

    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Wait until playback is finished
# text_to_speech("a bottle of water")
