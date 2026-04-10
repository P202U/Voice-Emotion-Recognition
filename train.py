from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense


def build_cnn(input_shape, num_classes):
    model = Sequential(
        [
            Conv1D(128, kernel_size=5, activation="relu", input_shape=(input_shape, 1)),
            MaxPooling1D(pool_size=5, strides=2),
            Dropout(0.5),
            Conv1D(64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=5, strides=2),
            Dropout(0.5),
            Flatten(),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
