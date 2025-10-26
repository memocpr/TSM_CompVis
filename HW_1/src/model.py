from typing import Tuple
import numpy as np

# Delay TensorFlow imports to runtime to allow using preprocessing without TF installed.

def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception as e:
        raise RuntimeError("TensorFlow is not installed; classification is disabled.") from e


def build_mnist_cnn(input_shape: Tuple[int,int,int]=(28,28,1), num_classes: int=10):
    tf = _require_tensorflow()
    from tensorflow.keras import layers, models  # type: ignore
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def load_or_train_default(model_path: str):
    tf = _require_tensorflow()
    from pathlib import Path
    from tensorflow.keras.datasets import mnist  # type: ignore

    p = Path(model_path)
    if p.exists():
        return tf.keras.models.load_model(str(p))
    # Train a tiny model quickly
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_mnist_cnn()
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)
    model.evaluate(x_test, y_test, verbose=0)
    model.save(str(p))
    return model


def predict_digits(model, patches: np.ndarray) -> np.ndarray:
    # patches: (N,28,28) uint8 or float
    if model is None:
        raise RuntimeError("No classification model available; TensorFlow is not installed.")
    x = patches.astype('float32')/255.0
    x = np.expand_dims(x, -1)
    probs = model.predict(x, verbose=0)
    return probs.argmax(axis=1)
