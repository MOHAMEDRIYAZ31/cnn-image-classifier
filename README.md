# 🧠 CNN Image Classifier

> **Achieved 80%+ test accuracy on CIFAR-10 using TensorFlow/Keras** — trained from scratch with custom augmentation and a deep residual-style CNN.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-80%25%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

A custom Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset (60,000 images across 10 classes). The model was built entirely in TensorFlow/Keras without pretrained weights, demonstrating core deep learning skills from data pipeline design to model deployment.

| Metric | Value |
|---|---|
| **Test Accuracy** | **80%+** |
| Dataset | CIFAR-10 (60,000 images, 10 classes) |
| Framework | TensorFlow 2.x / Keras |
| Training time | ~45 min (GPU) / ~3 hrs (CPU) |
| Parameters | ~2.8M |

---

## 🏗️ Model Architecture

```
Input (32×32×3)
    │
    ▼
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    │
    ▼
Conv Block 1:  Conv2D(64)  → BN → ReLU  ×2  → MaxPool → Dropout(0.25)
Conv Block 2:  Conv2D(128) → BN → ReLU  ×2  → MaxPool → Dropout(0.25)
Conv Block 3:  Conv2D(256) → BN → ReLU  ×2  → MaxPool → Dropout(0.30)
    │
    ▼
Global Average Pooling
    │
    ▼
Dense(512) → Dropout(0.5) → Softmax(10)
```

**Key design choices:**
- **Batch Normalisation** after every convolution → stable, fast training
- **Increasing filter depth** (64 → 128 → 256) → captures multi-scale features
- **Global Average Pooling** instead of Flatten → fewer parameters, less overfitting
- **Inline data augmentation** → no separate preprocessing step, cleaner pipeline

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/cnn-image-classifier.git
cd cnn-image-classifier
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py                  # default 50 epochs
python train.py --epochs 30      # custom epoch count
```

Training saves the best checkpoint to `models/best_model.keras` and plots to `results/training_curves.png`.

### 3. Evaluate

```bash
python evaluate.py
```

Outputs per-class precision/recall/F1 and saves a confusion matrix to `results/confusion_matrix.png`.

### 4. Predict

```bash
# Single image
python predict.py --image path/to/image.jpg

# Batch (folder)
python predict.py --folder path/to/images/
```

---

## 📊 Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Airplane | 0.84 | 0.86 | 0.85 |
| Automobile | 0.91 | 0.89 | 0.90 |
| Bird | 0.75 | 0.71 | 0.73 |
| Cat | 0.65 | 0.63 | 0.64 |
| Deer | 0.82 | 0.83 | 0.82 |
| Dog | 0.71 | 0.70 | 0.70 |
| Frog | 0.86 | 0.88 | 0.87 |
| Horse | 0.86 | 0.88 | 0.87 |
| Ship | 0.89 | 0.91 | 0.90 |
| Truck | 0.87 | 0.89 | 0.88 |
| **Overall** | **0.82** | **0.82** | **0.82** |

> Baseline (random): 10% | Simple CNN without augmentation: ~68% | **This model: 80%+**

---

## 🛠️ Training Techniques

| Technique | Detail |
|---|---|
| Optimiser | Adam (lr=1e-3, with ReduceLROnPlateau) |
| Regularisation | Batch Norm + Dropout (0.25–0.5) |
| Augmentation | Horizontal flip, rotation ±10°, zoom ±10%, translation |
| Early stopping | Patience = 10 epochs (monitors val_accuracy) |
| LR scheduling | Halved every 5 epochs of no improvement (min 1e-6) |
| Normalisation | Per-channel mean/std subtraction |

---

## 📁 Project Structure

```
cnn-image-classifier/
├── train.py            # Training loop + callbacks
├── evaluate.py         # Full test-set evaluation + confusion matrix
├── predict.py          # Single-image & batch inference
├── requirements.txt
├── models/             # Saved checkpoints (gitignored)
└── results/            # Training curves, confusion matrix
```

---

## 💡 Possible Extensions

- [ ] Replace backbone with ResNet-50 / EfficientNet (transfer learning)
- [ ] Export to TensorFlow Lite for mobile deployment
- [ ] Wrap in a FastAPI REST endpoint
- [ ] Extend to a custom dataset via `ImageDataGenerator`

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## Training Curves
![Training Curves](results/training_curves.png)

## Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

*Built with TensorFlow 2.x · CIFAR-10 dataset courtesy of Alex Krizhevsky*
