# Wind Turbine Detection in Satellite Images

**Binary image classification using Deep Learning (PyTorch)**  
Master MSS — Université de Bordeaux | 2024–2025

---

## Objective

Detect the presence or absence of wind turbines in satellite images (128×128 pixels).  
This is a **binary supervised classification** task evaluated on an unlabeled test set.

---

## Project Structure

```
├── Projet1-Classement-Images-pytorch.ipynb   # Baseline (logistic regression — provided)
├── rapport_Fama_Pimage.ipynb                 # Full project: CNN + Transfer Learning models
└── README.md
```

---

## Models Implemented

Starting from a logistic regression baseline, four deep learning models were developed and compared:

| Model | Architecture | Training |
|---|---|---|
| **CNN (custom)** | 3 conv layers + BatchNorm + Dropout | Full dataset |
| **MobileNetV2** | Pretrained on ImageNet | Fine-tuned (binary head) |
| **EfficientNetB3** | Pretrained on ImageNet | Fine-tuned + unfrozen layers |
| **ResNet50** | Pretrained on ImageNet | Fine-tuned (progressive layer unfreezing) |

---

## Methodology

**Preprocessing & Augmentation (training set)**
- Resize to 128×128
- Random horizontal flip + random rotation (20°)
- Normalization (ImageNet stats)

**Training strategy**
- Adam / AdamW / RMSprop optimizers depending on model
- Learning rate schedulers (StepLR, ReduceLROnPlateau, CosineAnnealing)
- Early stopping via best validation accuracy checkpoint
- Two-phase fine-tuning for transfer learning models (frozen → progressive unfreezing)

---

## Results

EfficientNetB3 and ResNet50 outperformed MobileNetV2 in terms of both stability and validation loss.  
The custom CNN showed competitive performance and was the only model trained on the full dataset due to hardware constraints (no GPU on Mac).

> Note: Full dataset training was limited to the custom CNN model due to GPU unavailability — other models were evaluated on the small dataset only.

---

## Tech Stack

- Python 3
- PyTorch & torchvision
- Google Colab (GPU training)
- Matplotlib

---

## Authors

Academic project — Master MSS, Université de Bordeaux (group work)
