# Culturally-Adaptive Helmet Detection for Indian Traffic

## 📌 The Problem
Standard Helmet Detection Systems (HDS) fail in the Indian context. Current generic AI models are primarily trained on Western datasets and cannot process the unique cultural nuances of Indian traffic. This creates two massive operational failures:

1. **False Positives (Wrongful Ticketing):** Penalizing Sikh riders wearing legally exempt Pagdis (Turbans).
2. **False Negatives (Missed Violations):** Misclassifying thick cultural fabrics (Dupattas, Burqas, Gamchas) as helmets due to their bulky silhouette. 

## 🎯 Our Approach
We developed a culturally-adaptive, 3-class "Head ROI" classifier (Helmet, Pagdi, No-Helmet) robust enough to handle these Indian visual distractors. 

* **Strict Taxonomy:** We decoupled 'Pagdi' and 'Helmet' into separate classes to prevent feature pollution.
* **Distractor Integration:** We intentionally included hijabs and dupattas into our No-Helmet class; giving this class high variance actually helps the model's accuracy by teaching it to recognize soft fabrics vs. hard plastics.
* **Smart Filtering:** The ultimate downstream application of this model is to act as a highly efficient gateway, triggering computationally heavy License Plate Recognition (OCR) only when a true "No-Helmet" violation is confirmed.

## 📂 Repository Structure
```text
├── VLM-Pipeline/             # Vision-Language Model scripts for automated curation
├── Models/                   # Core training directory
│   ├── runs/                 # Output directory for weights, metrics, and PR curves
│   ├── Datasets/             # Active training data (split into train/val)
│   ├── Models.ipynb          # Master notebook for comparative training execution
│   ├── dataset.yaml          # Class mapping and path configurations
│   ├── yolo11m.pt            # YOLOv11 Medium baseline weights
│   ├── yolo11n.pt            # YOLOv11 Nano classifier weights
│   └── rtdetr-l.pt           # RT-DETR Large baseline weights
├── Pre-Processed-Dataset/    # Intermediate dataset storage post-Area Filtering
├── New-Datasets/             # Data scraped from Kaggle & external sources
├── Auto-Annotation.ipynb     # Pipeline combining Grounding DINO + LLaVA
├── Preprocessing.ipynb       # Scripts for Area Filtering and standardization
├── Subsampling.ipynb         # Logic for visualising minority class ratios
├── fiftyone.ipynb            # Voxel FiftyOne integration for dataset visualization
├── README.md
└── LICENSE
```

## ⚙️ Methodology & Pipeline

Our pipeline leverages a combination of automated annotation and advanced data augmentation to handle severe class imbalances.

1. Data Curation & Preprocessing

    - Automated Annotation: We initially utilized Grounding DINO and LLaVA to generate high-accuracy segmentation masks and bounding boxes, separating the subject from the background.

    - Automated Area Filter: Bounding boxes below a specific pixel threshold are automatically discarded to eliminate tiny, blurry pixel blobs from the training set.

    - Contextual Augmentation: We utilize Mosaic augmentation to stitch images into a 2x2 grid, forcing the model to learn small-object distance detection. We also use MixUp (50% opacity blends) to simulate complex occlusions and visual noise.

2. Model Architecture

    We evaluated distinct architectural philosophies to ensure real-time edge processing capability:

    - YOLOv11-Medium: Selected for its C2PSA (Cross-Stage Partial Spatial Attention) module, which forces the network to focus on high-frequency textural differences (fabric folds vs. smooth polycarbonate).

    - RT-DETR-Large: A Transformer-based architecture utilized for its self-attention mechanisms, excelling at complex, overlapping visual boundaries.

    - Resnet-50 (Classifier): A 50-layer deep convolutional neural network that utilizes "shortcut" or "skip" connections to jump over blocks of layers.

    - Distribution Focal Loss: Implemented to heavily penalize the model when it misclassifies the rare "Pagdi" class, forcing the network to pay closer attention to it despite the lack of volume.

## 🚀 Hardware & Training Strategy

This pipeline was successfully compiled and trained locally on an 8GB NVIDIA RTX 4060 GPU. To ensure mathematical stability within strict VRAM limits:

- Gradient Accumulation: Physical batch sizes were dropped to safety thresholds (e.g., batch=2 or 4), with gradients accumulated (nbs=16) to maintain effective learning rates.

- FP32 for Transformers: Automatic Mixed Precision (amp=True) is disabled exclusively for RT-DETR to prevent 16-bit floating-point NaN overflows during heavy self-attention matrix multiplications.

### 🔮 Future Work & Deployment

To transition this model from academic evaluation to a live municipal traffic network:

- Primary Gating Model: Implementing a precursor model that strictly isolates active motorcyclists from the scene, mitigating the false positive error of flagging a pedestrian walking near a bike as a rider.

- DeepSORT Tracking: YOLO is memory-less and will flag one violator 30 times a second. We have identified DeepSORT (Object Tracking) as the required algorithmic solution to assign unique IDs to riders, ensuring a single violation log per person.