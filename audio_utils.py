import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


def convert_audio_16k(df, output_dir="audio_16k"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_paths = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        new_path = os.path.join(output_dir, f"{i}_{os.path.basename(row['Path'])}")
        try:
            audio, sr = librosa.load(row["Path"], sr=16000)
            sf.write(new_path, audio, 16000, subtype="PCM_16")
            new_paths.append(new_path)
        except:
            new_paths.append(None)
    return new_paths


def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])


def stretch_and_pitch(data, sr, rate=0.7, pitch_factor=0.7):
    data = librosa.effects.time_stretch(y=data, rate=rate)
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)
