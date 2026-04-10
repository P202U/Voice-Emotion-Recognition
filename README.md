# Voice Emotion Recognition (SER)

A modular Speech Emotion Recognition pipeline that combines **Deep Learning (Conv1D)** and **Classical ML (Random Forest with RFE)** to classify emotions across multiple datasets.

## Features

- **Multi-Dataset Support:** Auto-downloads and maps RAVDESS, CREMA-D, TESS, and SAVEE via `data_loader.py`.
- **Advanced Augmentation:** Triples training data with noise injection, time-stretching, and pitch-shifting via `audio_utils.py`.
- **Dual-Model Architecture:** Compare a 1D Convolutional Neural Network with a Feature-Selected Random Forest via `models.py`.
- **High-Speed Processing:** Parallelized feature extraction using `feature_extraction.py`.

## Installation

You can set up this project using **uv** (recommended for speed) or **pip**.

### Option 1: Using `uv` (Fastest)

```bash
git clone https://github.com/your-username/voice-emotion-recognition.git
cd voice-emotion-recognition

uv sync

uv run main.py
```

### Option 2: Using `pip`

```bash
git clone https://github.com/your-username/voice-emotion-recognition.git
cd voice-emotion-recognition

python -m venv .venv
source .venv/bin/activate

pip install .

python main.py
```

## Project Structure

```text
├── main.py              # Orchestration script (Training & Evaluation)
├── data_loader.py       # Logic to download and map dataset labels
├── audio_utils.py       # Audio resampling (16kHz) and augmentation logic
├── feature_extraction.py # Librosa feature extraction (MFCC, Mel, etc.)
├── models.py            # CNN & Random Forest architectures
├── pyproject.toml       # Dependency management (uv/pip)
├── Emotion_recognition.ipynb # Original research notebook
└── README.md
```

## How it Works

### 1. The Pipeline

- **Ingestion:** Downloads ~4GB of audio data via `kagglehub` using `data_loader.py`.
- **Standardization:** Resamples all audio to **16,000Hz PCM_16** via `audio_utils.py`.
- **Extraction:** Computes a **162-dimensional feature vector** per sample using `feature_extraction.py`.
- **Training:** \* **CNN:** Learns spatial patterns in spectral features via a 1D Convolutional Neural Network.
  - **RF + RFE:** Selects the top 93 features via Recursive Feature Elimination for a lightweight model.

### 2. Available Emotions

The model maps all dataset labels into a unified set:
**Neutral, Happy, Sad, Angry, Fear, Disgust, Surprised.**

## Evaluation

Upon completion, the script generates:

- **Classification Report:** Precision, Recall, and F1-Score per emotion.
- **Confusion Matrix:** A heatmap showing where the model is succeeding or confusing similar vocal tones (e.g., Sad vs. Neutral).

## Troubleshooting

- **FFmpeg:** Ensure `ffmpeg` is installed on your system; `librosa` and `soundfile` require it to process various audio formats.
