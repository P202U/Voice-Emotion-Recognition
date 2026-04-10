import librosa
import numpy as np


def extract_features(data, sr):
    n_fft = 2048
    hop_length = 512
    result = np.array([])

    # ZCR
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            y=data, frame_length=n_fft, hop_length=hop_length
        ).T,
        axis=0,
    )
    result = np.hstack((result, zcr))

    # Chroma
    stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))

    # RMS
    rms = np.mean(
        librosa.feature.rms(y=data, frame_length=n_fft, hop_length=hop_length).T, axis=0
    )
    result = np.hstack((result, rms))

    # Mel Spectrogram
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=data, sr=sr, n_fft=n_fft, hop_length=hop_length
        ).T,
        axis=0,
    )
    result = np.hstack((result, mel))

    # Spectral Rolloff
    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        ).T,
        axis=0,
    )
    result = np.hstack((result, rolloff))

    # Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, contrast))

    # Tonnetz
    tonnetz = np.mean(
        librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr).T, axis=0
    )
    result = np.hstack((result, tonnetz))

    return result


def get_all_features(path, sr=16000):
    data, sampling_rate = librosa.load(path, sr=sr)

    # 1. Original Data
    res1 = extract_features(data, sampling_rate)
    result = np.array(res1)

    # 2. Add Noise
    noise_data = add_noise(data)
    res2 = extract_features(noise_data, sampling_rate)
    result = np.vstack((result, res2))

    # 3. Stretch and Pitch
    str_data = stretch(data)
    pitch_data = pitch(str_data, sampling_rate)
    res3 = extract_features(pitch_data, sampling_rate)
    result = np.vstack((result, res3))

    return result
