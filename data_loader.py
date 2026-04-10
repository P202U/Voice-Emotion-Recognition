import kagglehub
import os
import pandas as pd


def download_datasets():
    paths = {
        "ravdess": kagglehub.dataset_download(
            "uwrfkaggler/ravdess-emotional-speech-audio"
        ),
        "crema": kagglehub.dataset_download("ejlok1/cremad"),
        "tess": kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess"),
        "savee": kagglehub.dataset_download(
            "ejlok1/surrey-audiovisual-expressed-emotion-savee"
        ),
    }
    return paths


def get_mapped_dataframe(paths):
    file_path, emotion = [], []

    # Nested folder logic for TESS/CREMA/SAVEE
    real_paths = {
        "rav": paths["ravdess"],
        "cre": os.path.join(paths["crema"], "AudioWAV"),
        "tes": os.path.join(paths["tess"], "TESS Toronto emotional speech set data"),
        "sav": os.path.join(paths["savee"], "ALL"),
    }


def get_data(ravdess_p, crema_p, tess_p, savee_p):
    file_path = []
    emotion = []

    # 1. RAVDESS
    rav_map = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fear",
        7: "disgust",
        8: "surprised",
    }
    for dir in os.listdir(ravdess_p):
        if not dir.startswith("Actor"):
            continue
        for file in os.listdir(os.path.join(ravdess_p, dir)):
            part = file.split(".")[0].split("-")
            emotion.append(rav_map[int(part[2])])
            file_path.append(os.path.join(ravdess_p, dir, file))

    # 2. CREMA-D
    crema_map = {
        "SAD": "sad",
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
    }
    for file in os.listdir(crema_p):
        if file.endswith(".wav"):
            part = file.split("_")[2]
            if part in crema_map:
                emotion.append(crema_map[part])
                file_path.append(os.path.join(crema_p, file))

    # 3. TESS
    for dir in os.listdir(tess_p):
        label = dir.split("_")[-1].lower()
        if label == "ps":
            label = "surprised"

        target_dir = os.path.join(tess_p, dir)
        if os.path.isdir(target_dir):
            for file in os.listdir(target_dir):
                if file.endswith(".wav"):
                    emotion.append(label)
                    file_path.append(os.path.join(target_dir, file))

    # 4. SAVEE
    savee_map = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprised",
    }
    for file in os.listdir(savee_p):
        if file.endswith(".wav"):
            code = "".join([c for c in file.split("_")[1] if not c.isdigit()])
            if code in savee_map:
                emotion.append(savee_map[code])
                file_path.append(os.path.join(savee_p, file))

    return pd.DataFrame({"Path": file_path, "Emotion": emotion})

    df["Emotion"] = df["Emotion"].replace({"surprise": "surprised", "calm": "neutral"})
    return df
