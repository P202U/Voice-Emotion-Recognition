# Voice Emotion Recognition (SER)

A modular Speech Emotion Recognition pipeline that combines **Deep Learning (Conv1D)** and **Classical ML (Random Forest with RFE)** to classify emotions across multiple datasets.

## Features

- **Multi-Dataset Support:** Auto-downloads RAVDESS, CREMA-D, TESS, and SAVEE.
- **Advanced Augmentation:** Triple your training data with noise injection, time-stretching, and pitch-shifting.
- **Dual-Model Architecture:** Compare a 1D Convolutional Neural Network with a Feature-Selected Random Forest.
- **High-Speed Processing:** Parallelized feature extraction using all available CPU cores.

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

---

## Project Structure

```text
├── data_loader.py    # Logic to download and map dataset labels
├── preprocessing.py  # Audio resampling (16kHz) and augmentation
├── features.py       # Librosa feature extraction (MFCC, Mel, etc.)
├── models.py         # CNN & Random Forest architectures
├── main.py           # Orchestration script (Training & Evaluation)
├── pyproject.toml    # Dependency management
└── README.md
```

## How it Works

### 1. The Pipeline

1.  **Ingestion:** Downloads ~4GB of audio data via `kagglehub`.
2.  **Standardization:** Resamples all audio to 16,000Hz PCM_16.
3.  **Extraction:** Computes a 162-dimensional feature vector per sample.
4.  **Training:** \* **CNN:** Learns spatial patterns in spectral features.
    - **RF + RFE:** Selects the top 93 features via Recursive Feature Elimination for a lightweight, interpretable model.

### 2. Available Emotions

The model maps all dataset labels into a unified set:
`Neutral`, `Happy`, `Sad`, `Angry`, `Fear`, `Disgust`, `Surprised`.

## Evaluation

Upon completion, the script generates:

- **Classification Report:** Precision, Recall, and F1-Score per emotion.
- **Confusion Matrix:** A heatmap showing where the model is succeeding or confusing similar vocal tones (e.g., Sad vs. Calm).

---

## Troubleshooting

- **FFmpeg:** Ensure `ffmpeg` is installed on your system, as `librosa` and `soundfile` require it to process certain audio formats.
