# 😀 Multi-Task Facial Affect Recognition

### 🎭 Expression Classification + Valence-Arousal Regression with Transfer Learning

A deep learning pipeline that predicts **three things at once** from a single face image: 🗂️ which of 8 discrete emotions is being expressed, 📈 how positive/negative the emotion feels (**valence**), and ⚡ how calm/excited it is (**arousal**). Built with **TensorFlow/Keras**, using transfer learning on top of **MobileNetV2** and **DenseNet121**, and benchmarked with a rigorous, research-grade evaluation suite.

---

## 📚 Table of Contents

- [🌟 Overview](#-overview)
- [🧠 The Science: Categorical + Dimensional Emotion](#-the-science-categorical--dimensional-emotion)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Expected Project Structure](#-expected-project-structure)
- [📦 Dataset Format](#-dataset-format)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🔬 Pipeline Deep Dive](#-pipeline-deep-dive)
  - [1. Loading Annotations](#1--loading-annotations)
  - [2. Loading & Normalizing Images](#2--loading--normalizing-images)
  - [3. Model Architecture](#3--model-architecture)
  - [4. Data Augmentation](#4--data-augmentation)
  - [5. Training](#5--training)
  - [6. Evaluation Metrics](#6--evaluation-metrics)
  - [7. Qualitative Results](#7--qualitative-results)
- [📊 Results](#-results)
- [🔎 Interpreting the Results](#-interpreting-the-results)
- [🚧 Known Limitations](#-known-limitations)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Overview

Human emotion is more than a single label like *"happy"* or *"sad"* — it also has **intensity** and **direction**. This project models emotion two ways simultaneously, in a single multi-task neural network:

| Task | Type | What it Captures |
|---|---|---|
| 🗂️ **Expression** | Classification (8 classes) | Discrete emotion category (e.g., neutral, happy, sad, surprise, fear, disgust, anger, contempt) |
| 📈 **Valence** | Regression (-1 to 1) | How *positive* or *negative* the emotion is |
| ⚡ **Arousal** | Regression (-1 to 1) | How *calm* or *intense/excited* the emotion is |

Rather than training three separate models, a **single shared CNN backbone** (MobileNetV2 or DenseNet121) feeds into **three specialized output heads** — one per task — allowing the network to learn shared visual features (facial geometry, muscle activation, expression texture) that benefit all three predictions jointly. 🧩

The project also includes a **comprehensive evaluation harness** — going well beyond simple accuracy — using metrics standard in affective computing research: Cohen's Kappa, Krippendorff's Alpha, Concordance Correlation Coefficient (CCC), Sign Agreement Ratio (SAGR), AUC-ROC, and AUC-PR. 📐

---

## 🧠 The Science: Categorical + Dimensional Emotion

This project bridges two classic models of emotion from affective computing:

- 🗂️ **Categorical model** — emotions belong to discrete classes (happy, sad, angry, etc.), typically framed as **Ekman's basic emotions** plus neutral/contempt.
- 🌐 **Dimensional model (Circumplex model of affect)** — emotions are points on a continuous 2D plane defined by:
  - **Valence** (x-axis): unpleasant ↔️ pleasant
  - **Arousal** (y-axis): calm ↔️ excited

```
                          High Arousal ⚡
                                │
                Angry 😠        │        Surprised 😲
                  Fear 😨       │       Excited 🤩
                                │
   Negative ─────────────────── ┼ ─────────────────── Positive
   Valence   😐 Neutral          │           Happy 😄     Valence
                Sad 😢          │
              Disgust 🤢        │
                                │
                          Low Arousal 😴
```

By predicting **both** representations from the same image, the model captures richer emotional nuance than a classification-only or regression-only approach would.

---

## 🏗️ Architecture

```
                       ┌─────────────────────────┐
                       │      Input Image        │
                       │      224 × 224 × 3      │
                       └────────────┬────────────┘
                                    v
                       ┌─────────────────────────────┐
                       │   Pretrained CNN Backbone   │
                       │  (MobileNetV2 / DenseNet121)│
                       │          Frozen (ImageNet)  │
                       └────────────┬────────────────┘
                                    v
                       ┌─────────────────────────┐
                       │  Global Average Pooling │
                       └────────────┬────────────┘
                                    v
                       ┌─────────────────────────┐
                       │   Dense(512) + ReLU     │
                       │   Dropout(0.5)          │
                       └────────────┬────────────┘
                       ┌────────────┼────────────┐
                       v            v            v
              ┌────────────┐ ┌────────────┐ ┌────────────┐
              │ Expression │ │  Valence   │ │  Arousal   │
              │ Dense(8)   │ │ Dense(1)   │ │ Dense(1)   │
              │ Softmax    │ │ Tanh       │ │ Tanh       │
              └────────────┘ └────────────┘ └────────────┘
```

**Key design choices:**

- 🧊 **Frozen backbone** — the pretrained CNN layers are frozen so only the new head layers are trained, making this a classic **transfer learning / feature extraction** setup (fast to train, resistant to overfitting on limited data).
- 🎯 **Softmax head** for the 8-way expression classification task.
- ↔️ **Tanh heads** for valence and arousal, since both are bounded in **[-1, 1]** — a perfect match for tanh's output range.
- 🔀 **Multi-task loss** — sparse categorical crossentropy for expression + MSE for valence and arousal, optimized jointly with Adam.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| 🧠 Deep Learning Framework | **TensorFlow / Keras** |
| 🏛️ Pretrained Backbones | **MobileNetV2**, **DenseNet121** (ImageNet weights) |
| 🔢 Numerical Computing | **NumPy** |
| 📊 Data Handling | **Pandas** |
| 🤖 Classical ML Utilities | **scikit-learn** (`train_test_split`, metrics) |
| 📈 Statistics | **SciPy** (`pearsonr`) |
| 🤝 Inter-Rater Reliability | **Krippendorff** (`krippendorff.alpha`) |
| 🎨 Visualization | **Matplotlib** |
| 🎲 Utilities | `os`, `random` |

---

## 📁 Expected Project Structure

```
facial-affect-recognition/
│
├── main.py                     # 🧠 Full pipeline: loading, training, evaluation, plotting
│
├── Dataset/
│   ├── images/                  # 🖼️ Face images (.jpg / .png)
│   └── annotations/             # 🏷️ Per-image .npy label files
│       ├── <id>_exp.npy          #     Discrete expression label (0–7)
│       ├── <id>_val.npy          #     Valence value (-1 to 1, or -2 = invalid)
│       └── <id>_aro.npy          #     Arousal value (-1 to 1, or -2 = invalid)
│
└── README.md
```

> 💡 This structure matches the **AffectNet**-style annotation format (8-class expressions + continuous valence/arousal, with `-2` reserved as a "no valid annotation" sentinel value).

---

## 📦 Dataset Format

The pipeline expects, for every image, three companion `.npy` annotation files sharing the same base filename:

| Suffix | Contents | Range |
|---|---|---|
| `_exp.npy` | Expression class label | `0`–`7` (8 discrete emotions) |
| `_val.npy` | Valence score | `-1.0` to `1.0` (or `-2` = invalid/missing) |
| `_aro.npy` | Arousal score | `-1.0` to `1.0` (or `-2` = invalid/missing) |

⚠️ Any sample where valence **or** arousal equals `-2` is automatically **filtered out** during preprocessing, since `-2` is used as a sentinel for missing/invalid annotations.

---

## ⚙️ Installation

### ✅ Prerequisites

- 🐍 Python 3.8+
- 🖥️ GPU strongly recommended (training two CNN-based multi-output models is compute-intensive)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/facial-affect-recognition.git
cd facial-affect-recognition
```

### 2️⃣ Create a Virtual Environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate      # 🐧 macOS/Linux
venv\Scripts\activate         # 🪟 Windows
```

### 3️⃣ Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib krippendorff
```

| Package | Why it's needed |
|---|---|
| 🧠 `tensorflow` | Model building, training, and inference (Keras API) |
| 🔢 `numpy` | Array operations, image tensors, annotation loading |
| 📊 `pandas` | Tabulating final results comparison |
| 🤖 `scikit-learn` | Train/val/test splitting + accuracy, F1, Cohen's Kappa, ROC/PR AUC |
| 📈 `scipy` | Pearson correlation for valence/arousal |
| 🎨 `matplotlib` | Training curves & qualitative prediction plots |
| 🤝 `krippendorff` | Krippendorff's Alpha inter-rater reliability metric |

### 4️⃣ Set Your Dataset Paths

Update the path constants near the top of the script to point at your local dataset:

```python
image_dir = r"path/to/Dataset/images"
annotation_dir = r"path/to/Dataset/annotations"
```

---

## 🚀 Usage

Run the full pipeline — data loading, training both backbones, evaluation, and plotting — with a single command:

```bash
python main.py
```

This will, in order:

1. 📥 Load and filter annotations
2. ✂️ Split data into **train (70%)** / **validation (~20%)** / **test (~10%)**
3. 🖼️ Load and normalize all images to `224×224×3`
4. 🏋️ Train **MobileNetV2** for 10 epochs (with data augmentation)
5. 📈 Plot its accuracy/loss curves
6. 🧪 Evaluate it on the held-out test set
7. 🏋️ Repeat steps 4–6 for **DenseNet121**
8. 🖼️ Display qualitative "correct vs. incorrect" prediction samples for each model
9. 📋 Print a final side-by-side results comparison table

---

## 🔬 Pipeline Deep Dive

### 1. 📥 Loading Annotations

`load_annotations()` scans the annotation directory for files ending in `_exp.npy`, `_val.npy`, and `_aro.npy`, **sorts them** for consistent ordering across the three label types, then loads and concatenates each into a single NumPy array — producing aligned `expressions`, `valence`, and `arousal` arrays.

### 2. 🖼️ Loading & Normalizing Images

`load_images()` loads each image at a fixed `224×224` resolution (matching the CNN backbones' expected input size), converts it to an array, and **normalizes pixel values to `[0, 1]`** by dividing by 255 — a standard preprocessing step for CNNs.

### 3. 🧠 Model Architecture

The `FacialExpressionModel` class wraps the entire model lifecycle:

- `build_model()` — instantiates either `MobileNetV2` or `DenseNet121` pretrained on ImageNet (with the classification top removed), **freezes all its layers**, then attaches a custom head: `GlobalAveragePooling2D → Dense(512, ReLU) → Dropout(0.5)` → three parallel output branches for expression, valence, and arousal.
- `compile_model()` — compiles with the **Adam** optimizer, `sparse_categorical_crossentropy` for expression and `mse` for valence/arousal, tracking `accuracy` and `mse` respectively.

### 4. 🎨 Data Augmentation

When `augment=True`, `train_model()` builds a Keras `ImageDataGenerator` with:

- 🔄 Random rotations (±20°)
- ↔️ Width/height shifts (±10%)
- 🪞 Horizontal flips
- 🔍 Random zoom (±20%)

A **custom generator function** batches images, applies augmentation, and yields them alongside their three-headed label dictionary — keeping expression, valence, and arousal labels correctly synced to their augmented images.

### 5. 🏋️ Training

`train_model()` fits the model for a configurable number of `epochs` and `batch_size`, using either the plain in-memory arrays or the augmented generator, with validation performed on a held-out validation set after every epoch.

### 6. 📐 Evaluation Metrics

`evaluate_model()` computes a **comprehensive battery of metrics** — because a single accuracy number rarely tells the whole story for emotion recognition:

**🗂️ Classification (Expression) Metrics:**
| Metric | What it Measures |
|---|---|
| ✅ Accuracy | Overall proportion of correct predictions |
| ⚖️ F1-Score (weighted) | Balance of precision & recall across all 8 classes |
| 🤝 Cohen's Kappa | Agreement between predicted & true labels, adjusted for chance |
| 📏 Krippendorff's Alpha | Inter-rater reliability between predictions and ground truth |
| 📈 AUC-ROC | Discriminative power across all classes (one-vs-rest) |
| 📉 AUC-PR | Precision-recall trade-off, especially useful for imbalanced classes |

**📈⚡ Regression (Valence & Arousal) Metrics:**
| Metric | What it Measures |
|---|---|
| 📉 RMSE | Average magnitude of prediction error |
| 🔗 Pearson Correlation | Linear relationship strength between predicted & true values |
| ➕➖ Sign Agreement Ratio (SAGR) | How often the predicted sign (positive/negative) matches the true sign |
| 🎯 Concordance Correlation Coefficient (CCC) | Combines correlation *and* mean-difference — the gold-standard agreement metric in affective computing |

### 7. 🖼️ Qualitative Results

`qualitative_results()` randomly samples correctly and incorrectly classified test images and displays them side-by-side with their predicted vs. true expression labels — a quick visual sanity check that complements the numeric metrics. 👀

---

## 📊 Results

Both backbones were trained for **10 epochs** with augmentation enabled. Here's how they compared on the held-out test set:

| Metric | 📱 MobileNetV2 | 🌐 DenseNet121 |
|---|---:|---:|
| ✅ Accuracy | 0.1010 | **0.1212** |
| ⚖️ F1-Score | 0.0653 | **0.0860** |
| 🤝 Cohen's Kappa | -0.0189 | **0.0018** |
| 📏 Krippendorff's Alpha | -0.1401 | **-0.0952** |
| 📈 AUC-ROC | **0.4732** | 0.4674 |
| 📉 AUC-PR | **0.1156** | 0.1138 |
| 📉 Valence RMSE | 0.4662 | **0.4659** |
| 🔗 Valence Correlation | -0.0242 | **0.0293** |
| ➕➖ Valence SAGR | 0.7071 | 0.7071 |
| 🎯 Valence CCC | -0.0021 | **0.0037** |
| 📉 Arousal RMSE | **0.3957** | 0.3993 |
| 🔗 Arousal Correlation | **-0.0677** | -0.1125 |
| ➕➖ Arousal SAGR | 0.7475 | 0.7475 |
| 🎯 Arousal CCC | **-0.0072** | -0.0202 |

*(Bold = better score for that metric across the two models.)*

### 📈 Training Snapshot

- 📱 **MobileNetV2**: ~70–90s/epoch, final train loss ≈ 2.44, val expression accuracy hovering around **12–13%**
- 🌐 **DenseNet121**: ~280–310s/epoch (much slower due to a deeper architecture), final train loss ≈ 2.44, val expression accuracy hovering around **11–14%**

---

## 🔎 Interpreting the Results

⚠️ **Honest read**: with a frozen backbone, only 10 training epochs, and expression accuracy hovering close to what random guessing across 8 classes would produce (~12.5%), both models are currently **underfitting** the expression classification task, and correlation-based valence/arousal metrics are close to zero — meaning the regression heads haven't yet learned a meaningfully predictive signal either. This is a **useful and expected starting point** for a from-scratch multi-task setup, and points directly at where to focus next:

- 🔓 **Unfreezing** some of the backbone's later layers for fine-tuning (rather than pure feature extraction) is likely to help the most.
- ⏳ **More epochs** — 10 is a very short training run for an 8-class + 2-regression multi-task problem.
- ⚖️ **Loss weighting** — right now, expression, valence, and arousal losses are summed with equal weight; tuning `loss_weights` in `compile()` can help balance the multi-task objective.
- 📊 **Class imbalance** — if certain expressions are underrepresented, techniques like class weighting or focal loss could help.
- 🧪 **Learning rate tuning** — the default Adam learning rate may not be optimal for this frozen-backbone regime.

---

## 🚧 Known Limitations

- 🧊 Backbones are fully frozen — the model can only recombine pretrained ImageNet features, not adapt them to facial affect specifically.
- ⏱️ Only 10 training epochs were run — likely insufficient for convergence on this multi-task objective.
- 🎯 Expression accuracy near chance level suggests the current configuration needs further tuning before being production-ready.
- 💻 Hardcoded local file paths (`C:\Users\...`) need to be generalized/parameterized for portability.
- 🖼️ No explicit class-imbalance handling for the 8 expression categories.

---

## 🔮 Future Improvements

- 🔓 Fine-tune (unfreeze) the top layers of the backbone after initial head training
- 📈 Increase training epochs with early stopping and learning rate scheduling
- ⚖️ Introduce class weighting or focal loss for the expression head
- 🎚️ Tune multi-task loss weights to balance classification vs. regression objectives
- 🧪 Experiment with additional backbones (EfficientNet, ResNet, ViT)
- 📊 Add k-fold cross-validation for more robust performance estimates
- 🗃️ Package the pipeline into a `requirements.txt` + CLI arguments for reproducibility
- 📉 Add TensorBoard logging for richer training diagnostics

---

## 🤝 Contributing

Contributions and suggestions are welcome! 🙌

1. 🍴 Fork the repository
2. 🌿 Create a new branch (`git checkout -b feature/amazing-improvement`)
3. 💾 Commit your changes
4. 📤 Push to the branch
5. 🔁 Open a Pull Request

---
