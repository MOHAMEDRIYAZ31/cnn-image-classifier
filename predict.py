"""
CNN Image Classifier — Inference
Run predictions on a single image or a directory of images.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from PIL import Image

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]
IMG_SIZE = 32


def preprocess_image(path: str) -> np.ndarray:
    """Load, resize, and normalise an image for inference."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)          # (1, 32, 32, 3)


def predict(model_path: str, image_path: str) -> dict:
    model = keras.models.load_model(model_path)
    x     = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    top_k = np.argsort(probs)[::-1][:3]

    return {
        "predicted": CLASSES[top_k[0]],
        "confidence": float(probs[top_k[0]]),
        "top3": [(CLASSES[i], float(probs[i])) for i in top_k],
    }


def batch_predict(model_path: str, folder: str):
    model  = keras.models.load_model(model_path)
    paths  = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))
    if not paths:
        print("No .jpg/.png images found.")
        return

    results = []
    for p in paths:
        x    = preprocess_image(str(p))
        prob = model.predict(x, verbose=0)[0]
        idx  = np.argmax(prob)
        results.append((p.name, CLASSES[idx], float(prob[idx])))
        print(f"  {p.name:<30} → {CLASSES[idx]:<12} ({prob[idx]*100:.1f}%)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN inference")
    parser.add_argument("--model",  default="models/best_model.keras")
    parser.add_argument("--image",  help="Path to a single image")
    parser.add_argument("--folder", help="Path to a folder of images")
    args = parser.parse_args()

    if args.image:
        result = predict(args.model, args.image)
        print(f"\nPrediction : {result['predicted']}")
        print(f"Confidence : {result['confidence']*100:.1f}%")
        print("Top-3:")
        for cls, prob in result["top3"]:
            print(f"  {cls:<12} {prob*100:.1f}%")

    elif args.folder:
        batch_predict(args.model, args.folder)

    else:
        parser.print_help()
        sys.exit(1)
