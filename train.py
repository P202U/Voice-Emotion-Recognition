import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from audio_utils import add_noise, stretch, pitch
from feature_extraction import extract_features
from models import build_conv1d, build_random_forest, run_rfe_selection

# ... [Keep your Parallel processing and splitting code from before] ...

if __name__ == "__main__":
    # 1. Prepare Data
    # (Assuming X_train, X_val, X_test and y_train, y_val, y_test are ready)

    # --- MODEL 1: Random Forest with RFE ---
    print("Starting Random Forest Pipeline...")
    rfe_model, X_train_rfe = run_rfe_selection(X_train, y_train, n_features=93)
    X_test_rfe = rfe_model.transform(X_test)

    rf_classifier = build_random_forest()
    rf_classifier.fit(X_train_rfe, y_train)

    rf_preds = rf_classifier.predict(X_test_rfe)
    print("Random Forest Results:\n", classification_report(y_test, rf_preds))

    # --- MODEL 2: Conv1D CNN ---
    print("Starting Conv1D Training...")
    # Conv1D uses the full feature set (174) reshaped
    X_train_cnn = np.expand_dims(X_train, -1)
    X_val_cnn = np.expand_dims(X_val, -1)
    X_test_cnn = np.expand_dims(X_test, -1)

    cnn_model = build_conv1d((X_train_cnn.shape[1], 1), len(np.unique(y_train)))
    cnn_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    cnn_model.fit(
        X_train_cnn,
        y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=100,
        batch_size=32,
    )

    cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    print("Conv1D Results:\n", classification_report(y_test, cnn_preds))
