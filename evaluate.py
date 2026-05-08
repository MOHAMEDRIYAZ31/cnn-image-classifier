"""
CNN Image Classifier — Evaluation
Full test-set evaluation with confusion matrix and per-class report.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import os

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]


def evaluate(model_path: str = "models/best_model.keras"):
    # Load data
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0

    # Load model & predict
    model   = keras.models.load_model(model_path)
    y_pred  = np.argmax(model.predict(x_test, verbose=1), axis=1)
    y_true  = y_test.flatten()

    # ── Metrics ─────────────────────────────────────────────
    loss_fn  = keras.losses.SparseCategoricalCrossentropy()
    test_acc = np.mean(y_pred == y_true)
    print(f"\nOverall Accuracy: {test_acc*100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── Confusion matrix ────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion Matrix  (acc={test_acc*100:.1f}%)")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved → results/confusion_matrix.png")

    return test_acc


if __name__ == "__main__":
    evaluate()
