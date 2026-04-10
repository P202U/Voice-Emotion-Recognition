import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import Parallel, delayed
import tensorflow as tf

# Import your custom modules
from data_loader import download_datasets, get_data_df
from preprocessing import convert_audio_16k
from features import get_all_variants
from models import build_conv1d, build_random_forest, run_rfe_selection

# 1. Load Data
print("Downloading and loading datasets...")
paths = download_datasets()
df = get_data_df(paths)

# 2. Preprocess
print("Converting audio to 16kHz...")
df["Processed_Path"] = convert_audio_16k(df)
df = df.dropna(subset=["Processed_Path"])

# 3. Parallel Feature Extraction
print("Extracting features (including augmentations)...")


def process_row(path, emo):
    try:
        feats = get_all_variants(path)
        return [(f, emo) for f in feats]
    except Exception as e:
        return []


results = Parallel(n_jobs=-1)(
    delayed(process_row)(p, e) for p, e in zip(df.Processed_Path, df.Emotion)
)

# 4. Data Flattening & Encoding
X, Y = [], []
for sublist in results:
    for f, emotion in sublist:
        X.append(f)
        Y.append(emotion)

X = np.array(X)
Y = np.array(Y)

lb = LabelEncoder()
y_encoded = lb.fit_transform(Y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- 6. MODEL A: Conv1D (CNN) ---
print("\n--- Training Conv1D Model ---")
# Reshape for Conv1D: (samples, features, 1)
X_train_cnn = np.expand_dims(X_train, axis=2)
X_val_cnn = np.expand_dims(X_val, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

cnn_model = build_conv1d(
    input_shape=(X_train_cnn.shape[1], 1), num_classes=len(lb.classes_)
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

cnn_history = cnn_model.fit(
    X_train_cnn,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_cnn, y_val),
    callbacks=[early_stop],
)

# --- 7. MODEL B: Random Forest with RFE ---
print("\n--- Training Random Forest with RFE ---")

rfe_selector, X_train_rfe = run_rfe_selection(X_train, y_train, n_features=93)
X_test_rfe = rfe_selector.transform(X_test)

rf_final = build_random_forest()
rf_final.fit(X_train_rfe, y_train)

# 8. Evaluation (Random Forest Example)
print("\nRandom Forest Classification Report:")
y_pred_rf = rf_final.predict(X_test_rfe)
print(classification_report(y_test, y_pred_rf, target_names=lb.classes_))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=lb.classes_,
    yticklabels=lb.classes_,
    cmap="Blues",
)
plt.title("Confusion Matrix: Random Forest + RFE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Pipeline Complete!")
