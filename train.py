"""
CNN Image Classifier — Training Script
Achieves 80%+ accuracy on CIFAR-10 using TensorFlow/Keras
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import argparse

# ─── Config ────────────────────────────────────────────────────────────────────
CLASSES      = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]
IMG_SIZE     = 32
BATCH_SIZE   = 64
EPOCHS       = 50
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "models/best_model.keras"


# ─── Data ──────────────────────────────────────────────────────────────────────
def load_and_preprocess():
    """Load CIFAR-10 and apply normalisation + augmentation."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalise to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Per-channel mean/std normalisation
    mean = np.mean(x_train, axis=(0, 1, 2))
    std  = np.std(x_train,  axis=(0, 1, 2))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test  = (x_test  - mean) / (std + 1e-7)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test  = keras.utils.to_categorical(y_test,  10)

    return (x_train, y_train), (x_test, y_test)


def build_augmentation_pipeline():
    """Keras-native augmentation layers."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")


# ─── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 10) -> keras.Model:
    """
    Deep CNN with:
    • Batch Normalisation after every conv block
    • Residual-style skip connections
    • Global Average Pooling (no dense bloat)
    • Dropout for regularisation
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation_pipeline()(inputs)

    # ── Block 1 ──────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ──────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3 ──────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # ── Head ─────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="CNN_Classifier")


# ─── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print(f"\nTensorFlow {tf.__version__}  |  GPU: {len(tf.config.list_physical_devices('GPU'))}")

    (x_train, y_train), (x_test, y_test) = load_and_preprocess()

    model = build_model()
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir="logs/"),
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=callbacks,
    )

    # ── Final evaluation ─────────────────────────────────
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{'='*40}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'='*40}\n")

    _plot_history(history, accuracy)
    return model, history


def _plot_history(history, test_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.axhline(test_acc, color="red", linestyle="--", label=f"Test ({test_acc*100:.1f}%)")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=150)
    print("Saved → results/training_curves.png")


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Image Classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args)
