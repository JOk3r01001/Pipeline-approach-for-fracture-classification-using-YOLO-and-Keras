# Pipeline-approach-for-fracture-classification-using-YOLO-and-Keras
A deep learning pipeline for automated bone fracture detection from X-ray images, combining a YOLO11 segmentation model with a custom EfficientNetB1-based classifier in Keras.

---
## Overview
This project implements a cascade pipeline where:
1. **YOLO11s-seg** localizes the fracture region using segmentation masks
2. **EfficientNetB1 (Keras)** classifies the cropped region as fractured or healthy

The pipeline was trained and evaluated on the **FracAtlas** dataset and developed in **Google Colab**.

---
## Results

### Binary Pipeline Results (Main Approach)
| Model | Accuracy | Precision (Fracture) | Recall (Fracture) | F1 Score (Fracture) |
|---|---|---|---|---|
| YOLO11s-seg (1 class) | - | 68.80% | 52.17% | 59.59% |
| Keras Classifier | 79% | 45% | 81% | 57% |
| **Pipeline (YOLO → Keras)** | **91%** | **79%** | **68%** | **73%** |

### Multi-Class YOLO Segmentation Results
| Class | Precision | Recall | mAP50 (Mask) | mAP50-95 (Mask) |
|---|---|---|---|---|
| No Fracture | 95.22% | 97.51% | 98.58% | - |
| Fracture | 64.72% | 45.16% | 52.12% | - |
| **Overall** | - | - | **75.35%** | **58.38%** |

> ⚠️ **Note:** Multi-class YOLO results are skewed due to the absence of real polygon annotations
> for healthy bones in FracAtlas. Synthetic full-image masks were generated for the No Fracture
> class, which caused the model to learn mask shape differences rather than actual fracture patterns.
> These results are therefore not suitable for direct comparison with the pipeline approach.

---
## Requirements
```bash
pip install ultralytics
pip install tensorflow
pip install keras
pip install scikit-learn
pip install pandas
pip install pillow
pip install opencv-python
```

---
## Project Structure
```
├── notebooks/
│   ├── 1_data_preparation.ipynb
│   ├── 2_yolo_training.ipynb
│   ├── 3_keras_training.ipynb
│   └── 4_pipeline_evaluation.ipynb
├── splits/
│   ├── universal_train_split.csv
│   ├── universal_val_split.csv
│   └── universal_test_split.csv
└── README.md
```

---
## Dataset
> Iftekharul Abedeen et al.
> *FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation.*
> Scientific Data, 10(1), 2023.
> https://doi.org/10.1038/s41597-023-02432-4

---
## Limitations
- Free version of Google Colab (T4 GPU, 15GB VRAM)
- FracAtlas originates exclusively from Bangladeshi medical centers
- Multi-class YOLO limited by absence of real polygon annotations for healthy bones
- Class imbalance (1:4.69 fracture to healthy) addressed via controlled undersampling

---
## Author
Radek Bártl — Bachelor's Thesis, Czech University of Life Sciences Prague (ČZU), 2026
