import io
from typing import Callable

import numpy as np
import torch
import pyaudio
import time
import wave
from faster_whisper import WhisperModel

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

TRIGGER_WORD = "alexa"

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

model_size = "small.en"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")


# Function to validate audio chunks
def validate(model, inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


# Convert audio from int16 to float32
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


# Audio configurations
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512  # Ensure this matches the VAD model's requirement
SILENCE_THRESHOLD = 0.3  # VAD confidence below this is considered silence
SILENCE_DURATION = 2  # Seconds of silence to end recording

padding_duration = 0.5  # seconds
padding_samples = int(SAMPLE_RATE * padding_duration)  # Total samples for the padding

# Generate silence padding
padding = np.zeros(padding_samples, dtype=np.int16)

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


# method called from hui_speech.py
def process(call_back_func: Callable[[str], str]):
    print("Listening...")
    voice_data = []
    silence_start = None
    try:
        while True:
            # Read audio chunk
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = int2float(audio_int16)

            # Ensure the correct number of samples
            if len(audio_float32) != CHUNK:
                print(f"Skipping chunk with invalid size: {len(audio_float32)}")
                continue

            # VAD confidence
            confidence = model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()

            if confidence > SILENCE_THRESHOLD:
                # Voice detected
                print(f"Voice detected with confidence: {confidence:.2f}")
                voice_data.append(audio_chunk)
                silence_start = None  # Reset silence timer
            else:
                # Silence detected
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    # Save and process voice data after silence
                    if voice_data:
                        print("Voice segment captured..")
                        voice_data_with_padding = [padding.tobytes()] + voice_data

                        audio_binary = io.BytesIO()
                        # Write WAV header and data into the BytesIO object
                        with wave.open(audio_binary, "wb") as wf:
                            wf.setnchannels(CHANNELS)  # Mono audio
                            wf.setsampwidth(audio.get_sample_size(FORMAT))
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(b''.join(voice_data_with_padding))

                        # Ensure the stream position is at the start for reading
                        audio_binary.seek(0)

                        segments, info = whisper_model.transcribe(audio_binary,
                                                                  beam_size=5)
                        print("-- Detected language '%s' with probability %f" % (
                        info.language, info.language_probability))

                        trigger_word_found = False
                        full_text = ""
                        for segment in segments:
                            st = segment.start
                            end = segment.end
                            txt = segment.text
                            print("[%.2fs -> %.2fs] %s" % (st, end, txt))
                            if TRIGGER_WORD in txt.lower():
                                trigger_word_found = True
                            if trigger_word_found:
                                full_text = full_text + txt

                        if trigger_word_found:
                            print("-- Processing detected text  : " + full_text)
                            call_back_func(full_text)
                        voice_data = []  # Reset for the next segment

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
