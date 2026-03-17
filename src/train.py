import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]


def train_model(model, X_train, y_train, X_val, y_val, batch_size=2048, epochs=40):
    class_weight = get_class_weights(y_train)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=get_callbacks(),
        verbose=1
    )

    return model, pd.DataFrame(history.history)
