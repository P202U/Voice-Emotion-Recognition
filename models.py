from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def build_conv1d(input_shape, num_classes):
    """Conv1D architecture from Table 1"""
    model = Sequential(
        [
            Conv1D(
                128,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            MaxPooling1D(pool_size=5, strides=2, padding="same"),
            Dropout(0.5),
            Conv1D(64, kernel_size=5, strides=1, padding="same", activation="relu"),
            MaxPooling1D(pool_size=5, strides=2, padding="same"),
            Dropout(0.5),
            Flatten(),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_random_forest():
    """RF architecture: 100 trees, Gini, no max depth"""
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        bootstrap=True,
        random_state=42,
    )


def run_rfe_selection(X_train, y_train, n_features=93):
    """RFE to reduce features to the top 93"""
    selector_rf = build_random_forest()
    rfe = RFE(estimator=selector_rf, n_features_to_select=n_features, step=5)
    X_train_selected = rfe.fit_transform(X_train, y_train)
    return rfe, X_train_selected
