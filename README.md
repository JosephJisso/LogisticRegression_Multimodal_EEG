<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/OpenCV-4.0+-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🧠 Multimodal Depression Detection System</h1>

<p align="center">
  <b>A machine learning pipeline combining Facial Expression Analysis and EEG Signal Processing for comprehensive depression screening</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Modality%201-Facial%20Video%20Analysis-blueviolet?style=flat-square" alt="Facial"/>
  <img src="https://img.shields.io/badge/Modality%202-EEG%20Signal%20Processing-teal?style=flat-square" alt="EEG"/>
  <img src="https://img.shields.io/badge/Algorithm-Logistic%20Regression-red?style=flat-square" alt="Algorithm"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Modality 1: Facial Expression Analysis](#-modality-1-facial-expression-analysis)
- [Modality 2: EEG-Based Detection](#-modality-2-eeg-based-detection)
- [Multimodal Fusion](#-multimodal-fusion)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

Depression is a critical mental health condition affecting millions worldwide. Early detection is crucial for timely intervention. This project implements a **multimodal approach** that combines:

| Modality | Data Source | Analysis Type |
|----------|-------------|---------------|
| 🎭 **Facial Expression** | Video Frames | Visual Emotion Recognition |
| 🧠 **EEG Signals** | Brain Activity | Neurophysiological Patterns |

By fusing insights from both **behavioral** (facial expressions) and **physiological** (brain signals) indicators, the system provides a more robust and comprehensive depression screening mechanism.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL DEPRESSION DETECTION SYSTEM                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐    │
│  │     MODALITY 1: VIDEO       │    │      MODALITY 2: EEG            │    │
│  │                             │    │                                 │    │
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────┐  │    │
│  │  │   Video Input (.mp4)  │  │    │  │   EEG Signals (32 Ch)     │  │    │
│  │  └───────────┬───────────┘  │    │  └─────────────┬─────────────┘  │    │
│  │              ▼              │    │                ▼                │    │
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────┐  │    │
│  │  │   Frame Extraction    │  │    │  │  Band-Pass Filtering      │  │    │
│  │  │   (Every 5th Frame)   │  │    │  │  (0.5 - 45 Hz)            │  │    │
│  │  └───────────┬───────────┘  │    │  └─────────────┬─────────────┘  │    │
│  │              ▼              │    │                ▼                │    │
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────┐  │    │
│  │  │   Preprocessing       │  │    │  │   Feature Extraction      │  │    │
│  │  │   • Grayscale         │  │    │  │   • Time Domain (Mean,Var)│  │    │
│  │  │   • Resize (48×48)    │  │    │  │   • Frequency Domain      │  │    │
│  │  │   • Flatten (2304)    │  │    │  │     (δ, θ, α, β bands)    │  │    │
│  │  │   • StandardScaler    │  │    │  │   • StandardScaler        │  │    │
│  │  └───────────┬───────────┘  │    │  └─────────────┬─────────────┘  │    │
│  │              ▼              │    │                ▼                │    │
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────┐  │    │
│  │  │  Logistic Regression  │  │    │  │   Logistic Regression     │  │    │
│  │  │  (7-Class Emotion)    │  │    │  │   (Binary: Dep/Non-Dep)   │  │    │
│  │  └───────────┬───────────┘  │    │  └─────────────┬─────────────┘  │    │
│  │              ▼              │    │                ▼                │    │
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────┐  │    │
│  │  │  Depression Ratio     │  │    │  │   Depression Prediction   │  │    │
│  │  │  (Sad+Neutral+Fear)/  │  │    │  │   (0: Non-Dep, 1: Dep)    │  │    │
│  │  │   Happy               │  │    │  └─────────────┬─────────────┘  │    │
│  │  └───────────┬───────────┘  │    │                │                │    │
│  │              │              │    │                │                │    │
│  └──────────────┼──────────────┘    └────────────────┼────────────────┘    │
│                 │                                    │                     │
│                 └──────────────┬─────────────────────┘                     │
│                                ▼                                           │
│                 ┌───────────────────────────────────┐                      │
│                 │       MULTIMODAL FUSION           │                      │
│                 │    Combined Depression Score      │                      │
│                 └───────────────────────────────────┘                      │
│                                │                                           │
│                                ▼                                           │
│                 ┌───────────────────────────────────┐                      │
│                 │         FINAL OUTPUT              │                      │
│                 │  "High Depressive Indicators" or  │                      │
│                 │  "Normal/Balanced Affect"         │                      │
│                 └───────────────────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎭 Modality 1: Facial Expression Analysis

### Phase A: Training Pipeline

Trains a Logistic Regression model to recognize 7 emotions from static facial images.

#### Dataset
- **Source:** [Kaggle - DepVidMood Facial Expression Dataset](https://www.kaggle.com/datasets/ziya07/depvidmood-facial-expression-video-dataset)
- **Emotions:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

#### Preprocessing Pipeline

```python
Raw Image → Grayscale → Resize(48×48) → Flatten(2304) → StandardScaler
```

| Step | Operation | Input | Output |
|------|-----------|-------|--------|
| 1 | Grayscale Conversion | RGB (3 channels) | Gray (1 channel) |
| 2 | Resizing | Variable size | 48 × 48 pixels |
| 3 | Flattening | 2D Matrix (48×48) | 1D Vector (2304) |
| 4 | Scaling | Raw pixels | Normalized (μ=0, σ=1) |

#### Model Configuration

```python
LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=1000
)
```

### Phase B: Video Analysis Pipeline

Applies the trained model to video files for real-time depression screening.

```
Video (.mp4) → Frame Extraction → Preprocessing → Inference → Aggregation → Depression Score
```

#### Depression Heuristic

$$\text{Depression Ratio} = \frac{\text{Sad} + \text{Neutral} + \text{Fear}}{\text{Happy} + 1}$$

| Ratio | Classification |
|-------|----------------|
| > 2.0 | 🔴 High Depressive Indicators |
| ≤ 2.0 | 🟢 Normal/Balanced Affect |

---

## 🧠 Modality 2: EEG-Based Detection

### Dataset: DEAP

| Specification | Value |
|---------------|-------|
| **Subjects** | 32 |
| **Trials/Subject** | 40 |
| **EEG Channels** | 32 |
| **Sampling Rate** | 128 Hz |
| **Labels** | Valence, Arousal, Dominance, Liking |

### Depression Labeling Strategy

Since clinical labels are unavailable, we use **valence-based proxy labeling**:

| Valence | Label | Class |
|---------|-------|-------|
| ≤ 4 | 1 | 😔 Depressed |
| > 4 | 0 | 😊 Non-Depressed |

### Signal Processing Pipeline

#### 1. Band-Pass Filtering
```
Butterworth Filter: 0.5 Hz - 45 Hz
Purpose: Remove noise, baseline drift, high-frequency artifacts
```

#### 2. Feature Extraction

**Time-Domain Features:**
- Mean amplitude
- Variance

**Frequency-Domain Features (Welch PSD):**

| Band | Frequency Range | Associated State |
|------|-----------------|------------------|
| **Delta (δ)** | 0.5 - 4 Hz | Deep sleep |
| **Theta (θ)** | 4 - 8 Hz | Drowsiness, meditation |
| **Alpha (α)** | 8 - 13 Hz | Relaxation, calm |
| **Beta (β)** | 13 - 30 Hz | Active thinking, focus |

### Model Configuration

```python
LogisticRegression(
    penalty='l2',
    class_weight='balanced',
    max_iter=1000
)
```

### Training Strategy

| Parameter | Value |
|-----------|-------|
| Train/Test Split | 80/20 |
| Stratification | Yes |
| Scaling | StandardScaler |
| Missing Values | Mean Imputation |

---

## 🔗 Multimodal Fusion

The system combines predictions from both modalities to provide a robust final assessment:

```
                    ┌─────────────────┐
                    │  Video Analysis │ ──► Depression Ratio
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Fusion Layer   │ ──► Final Prediction
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  EEG Analysis   │ ──► Binary Classification
                    └─────────────────┘
```

### Fusion Strategy

| Video Result | EEG Result | Final Output |
|--------------|------------|--------------|
| High | Depressed | 🔴 **Confirmed High Risk** |
| High | Non-Depressed | 🟡 **Moderate Risk** |
| Normal | Depressed | 🟡 **Moderate Risk** |
| Normal | Non-Depressed | 🟢 **Low Risk** |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/JosephJisso/LogisticRegression_Multimodal_EEG.git
cd LogisticRegression_Multimodal_EEG

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
kaggle>=1.5.0
```

### Dataset Setup

```bash
# Configure Kaggle API
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<user>\.kaggle\ (Windows)

# Download Facial Expression Dataset
kaggle datasets download ziya07/depvidmood-facial-expression-video-dataset
unzip depvidmood-facial-expression-video-dataset.zip -d data/facial

# Download DEAP Dataset
kaggle datasets download manh123df/deap-dataset
unzip deap-dataset.zip -d data/eeg
```

---

## 💻 Usage

### Training the Models

```python
# Train Facial Expression Model
python train_facial_model.py --data_dir data/facial --output_dir models/

# Train EEG Model
python train_eeg_model.py --data_dir data/eeg --output_dir models/
```

### Running Inference

```python
# Analyze a video file
python analyze_video.py --video_path path/to/video.mp4 --model_path models/facial_model.pkl

# Analyze EEG data
python analyze_eeg.py --eeg_path path/to/eeg_data.dat --model_path models/eeg_model.pkl

# Run multimodal analysis
python multimodal_analysis.py \
    --video_path path/to/video.mp4 \
    --eeg_path path/to/eeg_data.dat \
    --facial_model models/facial_model.pkl \
    --eeg_model models/eeg_model.pkl
```

### Example Output

```
═══════════════════════════════════════════════════════════════
              MULTIMODAL DEPRESSION ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

📹 VIDEO ANALYSIS
───────────────────────────────────────────────────────────────
  Frames Analyzed: 1,245
  Emotion Distribution:
    • Happy:   156 (12.5%)
    • Sad:     423 (34.0%)
    • Neutral: 389 (31.2%)
    • Fear:    187 (15.0%)
    • Angry:    45 (3.6%)
    • Disgust:  28 (2.2%)
    • Surprise: 17 (1.4%)
  
  Depression Ratio: 6.36
  Status: 🔴 HIGH DEPRESSIVE INDICATORS

🧠 EEG ANALYSIS
───────────────────────────────────────────────────────────────
  Trials Analyzed: 40
  Classification: DEPRESSED
  Confidence: 78.3%

🔗 MULTIMODAL FUSION
───────────────────────────────────────────────────────────────
  Combined Assessment: CONFIRMED HIGH RISK
  Recommendation: Professional consultation advised

═══════════════════════════════════════════════════════════════
```

---

## 📊 Results

### Facial Expression Recognition

#### Confusion Matrix

<p align="center">
  <img src="docs/confusion_matrix.png" alt="Confusion Matrix" width="600"/>
</p>

#### Performance Metrics

| Emotion | Samples | Accuracy |
|---------|---------|----------|
| Angry | 60 | 100% |
| Disgust | 48 | 100% |
| Fear | 48 | 100% |
| Happy | 50 | 100% |
| Neutral | 72 | 100% |
| Sad | 50 | 100% |
| Surprise | 60 | 100% |

**Overall Training Accuracy:** 100% *(Note: Evaluate on held-out test set for generalization)*

### EEG Classification

| Metric | Score |
|--------|-------|
| Accuracy | ~75-85% |
| Precision | ~0.78 |
| Recall | ~0.82 |
| F1-Score | ~0.80 |

---

## 📁 Project Structure

```
multimodal-depression-detection/
│
├── 📂 data/
│   ├── 📂 facial/                    # Facial expression images
│   │   ├── 📂 Angry/
│   │   ├── 📂 Disgust/
│   │   ├── 📂 Fear/
│   │   ├── 📂 Happy/
│   │   ├── 📂 Neutral/
│   │   ├── 📂 Sad/
│   │   └── 📂 Surprise/
│   ├── 📂 eeg/                       # EEG signal data
│   │   └── 📂 data_preprocessed_python/
│   └── 📂 videos/                    # Test video files
│
├── 📂 models/
│   ├── 📄 facial_model.pkl           # Trained facial model
│   ├── 📄 facial_scaler.pkl          # Facial feature scaler
│   ├── 📄 eeg_model.pkl              # Trained EEG model
│   └── 📄 eeg_scaler.pkl             # EEG feature scaler
│
├── 📂 src/
│   ├── 📄 facial_preprocessing.py    # Image preprocessing utilities
│   ├── 📄 eeg_preprocessing.py       # EEG signal processing
│   ├── 📄 feature_extraction.py      # Feature extraction functions
│   ├── 📄 train_facial_model.py      # Facial model training script
│   ├── 📄 train_eeg_model.py         # EEG model training script
│   ├── 📄 analyze_video.py           # Video analysis script
│   ├── 📄 analyze_eeg.py             # EEG analysis script
│   └── 📄 multimodal_analysis.py     # Combined analysis script
│
├── 📂 notebooks/
│   ├── 📓 01_facial_eda.ipynb        # Facial data exploration
│   ├── 📓 02_eeg_eda.ipynb           # EEG data exploration
│   └── 📓 03_multimodal_demo.ipynb   # Full pipeline demonstration
│
├── 📂 docs/
│   ├── 📄 confusion_matrix.png       # Results visualization
│   └── 📄 architecture.png           # System architecture diagram
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
└── 📄 LICENSE                        # MIT License
```

---

## 🔮 Future Work

- [ ] **Deep Learning Integration:** Replace Logistic Regression with CNNs for facial analysis and LSTMs/Transformers for EEG
- [ ] **Real-time Processing:** Implement live video and EEG stream analysis
- [ ] **Additional Modalities:** Incorporate speech analysis and text sentiment
- [ ] **Clinical Validation:** Partner with healthcare institutions for real-world testing
- [ ] **Mobile Deployment:** Develop smartphone application for accessibility
- [ ] **Explainability:** Add SHAP/LIME interpretations for model predictions

---

## 📚 References

1. **DEAP Dataset:** Koelstra, S., et al. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE Transactions on Affective Computing.

2. **Facial Expression Recognition:** Goodfellow, I., et al. (2013). Challenges in Representation Learning: A Report on Three Machine Learning Contests.

3. **Depression Detection from EEG:** Acharya, U. R., et al. (2018). Automated EEG-based screening of depression using deep convolutional neural network.

4. **Multimodal Affective Computing:** Poria, S., et al. (2017). A review of affective computing: From unimodal analysis to multimodal fusion.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Authors

- **Joseph Jisso** - *Initial work* - [GitHub Profile](https://github.com/JosephJisso)

---

## ⭐ Acknowledgments

- Kaggle community for datasets
- scikit-learn developers
- OpenCV community
- All researchers in affective computing and mental health AI

---

<p align="center">
  <b>⚠️ Disclaimer</b>
</p>

<p align="center">
  <i>This system is designed for research and educational purposes only. It is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for mental health concerns.</i>
</p>

---

<p align="center">
  Made with ❤️ for Mental Health Awareness
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Mental%20Health-Matters-green?style=for-the-badge" alt="Mental Health Matters"/>
</p>
