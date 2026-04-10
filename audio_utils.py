import librosa
import numpy as np
import soundfile as sf
import os


def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    noise = noise_amp * np.random.normal(size=data.shape[0])
    return data + noise


def stretch(data, rate=0.7):
    return librosa.effects.time_stretch(y=data, rate=rate)


def shift(data):
    shift_range = int(np.random.randint(-5, 5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)


def preprocess_audio(file_path, output_folder, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, file_name)
    sf.write(output_path, audio, target_sr, subtype="PCM_16")
    return output_path
