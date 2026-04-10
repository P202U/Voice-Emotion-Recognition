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
