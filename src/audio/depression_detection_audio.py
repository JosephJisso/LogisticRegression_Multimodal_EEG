# %% [markdown]
# # 🎙️ Depression Detection from Audio
# ## Binary Classification using Logistic Regression
# 
# **Objective:** Build a pipeline that classifies audio samples as **Depressed (1)** or **Healthy (0)** using acoustic features.
# 
# ---
# 
# ### Pipeline Overview:
# 1. **Data Loading** - DAIC-WOZ or RAVDESS fallback
# 2. **Feature Extraction** - MFCCs, Pitch, Jitter, Shimmer, Energy
# 3. **Preprocessing** - Handle NaN, SMOTE for imbalance
# 4. **Training** - Logistic Regression with StandardScaler
# 5. **Evaluation** - Metrics & Feature Importance

# %% [markdown]
# ---
# ## Step A: Environment Setup

# %%
# Install required packages (run once)
# !pip install librosa scikit-learn imbalanced-learn pandas matplotlib seaborn

# %%
import os
import warnings
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Audio
import librosa

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve

# Imbalance handling
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

print("✓ All imports successful!")

# %% [markdown]
# ---
# ## Step B: Download Sample Data (RAVDESS Fallback)
# 
# Since DAIC-WOZ requires access permissions, we use **RAVDESS** as a proxy:
# - Sad/Fearful emotions → **Depressed (1)**
# - Neutral/Calm/Happy → **Healthy (0)**

# %%
def download_sample_data(target_dir="./sample_audio_data"):
    """Downloads RAVDESS dataset for testing."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    extract_path = target_path / "ravdess"
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"✓ Dataset already exists at: {extract_path}")
        return str(extract_path)
    
    ravdess_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = target_path / "ravdess_speech.zip"
    
    print("Downloading RAVDESS (~200MB)...")
    
    def progress(block_num, block_size, total_size):
        pct = min(100, (block_num * block_size / total_size) * 100)
        print(f"\rProgress: {pct:.1f}%", end='')
    
    urllib.request.urlretrieve(ravdess_url, zip_path, progress)
    print("\n✓ Download complete!")
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    os.remove(zip_path)
    
    print(f"✓ Extracted to: {extract_path}")
    return str(extract_path)

# %%
def create_labels_from_ravdess(dataset_path):
    """Maps RAVDESS emotions to binary depression labels."""
    data = []
    depressed_emotions = ['04', '06']  # sad, fearful
    healthy_emotions = ['01', '02', '03']  # neutral, calm, happy
    
    for wav_file in Path(dataset_path).rglob("*.wav"):
        parts = wav_file.name.replace('.wav', '').split('-')
        if len(parts) >= 3:
            emotion = parts[2]
            if emotion in depressed_emotions:
                label = 1
            elif emotion in healthy_emotions:
                label = 0
            else:
                continue
            data.append({'file_path': str(wav_file), 'label': label})
    
    df = pd.DataFrame(data)
    print(f"\n✓ {len(df)} audio files")
    print(f"  Healthy: {len(df[df['label']==0])} | Depressed: {len(df[df['label']==1])}")
    return df

# %%
# Download and create labels
dataset_path = download_sample_data()
dataset_df = create_labels_from_ravdess(dataset_path)
dataset_df.head()

# %% [markdown]
# ---
# ## Step C: Feature Extraction Engine
# 
# We extract **42 acoustic features** per audio file:
# 
# | Category | Features | Why? |
# |----------|----------|------|
# | **Spectral** | 13 MFCCs × 2 (mean/std) | Vocal tract shape |
# | **Prosodic** | Pitch mean, std, range | Emotional intonation |
# | **Voice Quality** | Jitter, Shimmer | Laryngeal perturbation |
# | **Temporal** | ZCR, RMS Energy | Speech dynamics |

# %%
# Configuration
N_MFCC = 13           # Standard for speech recognition
FRAME_LENGTH = 2048   # ~93ms at 22050Hz
HOP_LENGTH = 512      # ~23ms hop

# %%
def calculate_jitter(f0_values):
    """Jitter = pitch period variation (higher in voice disorders)."""
    voiced = f0_values[~np.isnan(f0_values) & (f0_values > 0)]
    if len(voiced) < 2:
        return np.nan
    periods = 1.0 / voiced
    return np.mean(np.abs(np.diff(periods))) / np.mean(periods)


def calculate_shimmer(audio, sr, f0_values):
    """Shimmer = amplitude variation (proxy using RMS energy)."""
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    if len(rms) < 2:
        return np.nan
    return np.mean(np.abs(np.diff(rms))) / np.mean(rms)

# %%
def extract_features(file_path):
    """Extract comprehensive acoustic features from one audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        
        # Validate audio
        if len(audio) < sr * 0.5 or np.max(np.abs(audio)) < 0.001:
            return None
        
        features = {}
        
        # === 1. MFCCs (26 features) ===
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, 
                                      n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        for i in range(N_MFCC):
            features[f'MFCC_{i+1}_Mean'] = np.mean(mfccs[i])
            features[f'MFCC_{i+1}_Std'] = np.std(mfccs[i])
        
        # === 2. Pitch / F0 (3 features) ===
        f0, _, _ = librosa.pyin(audio, fmin=65, fmax=2093, sr=sr,
                                 frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        voiced_f0 = f0[~np.isnan(f0)]
        features['Pitch_Mean'] = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
        features['Pitch_Std'] = np.std(voiced_f0) if len(voiced_f0) > 0 else 0
        features['Pitch_Range'] = (np.max(voiced_f0) - np.min(voiced_f0)) if len(voiced_f0) > 0 else 0
        
        # === 3. Jitter & Shimmer (2 features) ===
        features['Jitter'] = calculate_jitter(f0)
        features['Shimmer'] = calculate_shimmer(audio, sr, f0)
        
        # === 4. ZCR & RMS Energy (4 features) ===
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        features['ZCR_Mean'] = np.mean(zcr)
        features['ZCR_Std'] = np.std(zcr)
        
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        features['RMS_Mean'] = np.mean(rms)
        features['RMS_Std'] = np.std(rms)
        
        # === 5. Additional Spectral (6 features) ===
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
        features['Spectral_Centroid_Mean'] = np.mean(cent)
        features['Spectral_Centroid_Std'] = np.std(cent)
        
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
        features['Spectral_Bandwidth_Mean'] = np.mean(bw)
        features['Spectral_Bandwidth_Std'] = np.std(bw)
        
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
        features['Spectral_Rolloff_Mean'] = np.mean(rolloff)
        features['Spectral_Rolloff_Std'] = np.std(rolloff)
        
        return features
        
    except Exception as e:
        print(f"Error: {file_path} - {e}")
        return None

# %%
# Extract features from all files
print("Extracting features from audio files...\n")

all_features = []
for idx, row in dataset_df.iterrows():
    if (idx + 1) % 100 == 0:
        print(f"Processing {idx + 1}/{len(dataset_df)}...")
    
    feats = extract_features(row['file_path'])
    if feats:
        feats['label'] = row['label']
        all_features.append(feats)

features_df = pd.DataFrame(all_features)
print(f"\n✓ Extracted features from {len(features_df)} files")
print(f"  Feature count: {len(features_df.columns) - 1}")

# %%
# Preview
features_df.head()

# %% [markdown]
# ---
# ## Step D: Data Processing
# 
# 1. Handle NaN/Inf values
# 2. Check class imbalance
# 3. Apply SMOTE if needed

# %%
# Separate features and labels
feature_cols = [c for c in features_df.columns if c != 'label']
X = features_df[feature_cols].copy()
y = features_df['label'].values

print(f"Shape: {X.shape}")
print(f"\nNaN counts per feature:")
nan_counts = X.isna().sum()
print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "  No NaN values!")

# %%
# Handle NaN and Inf
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
print("✓ Replaced NaN/Inf with median")

# %%
# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class Distribution:")
for u, c in zip(unique, counts):
    print(f"  {'Healthy' if u==0 else 'Depressed'} ({u}): {c} ({c/len(y)*100:.1f}%)")

imbalance_ratio = max(counts) / min(counts)
print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

# %%
# Visualize class distribution
plt.figure(figsize=(6, 4))
colors = ['#3498db', '#e74c3c']
plt.bar(['Healthy (0)', 'Depressed (1)'], counts, color=colors)
plt.title('Class Distribution', fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step E: Train Logistic Regression Model

# %%
# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# %%
# Apply SMOTE to training data only
minority_count = min(np.unique(y_train, return_counts=True)[1])
k_neighbors = min(5, minority_count - 1)

if k_neighbors >= 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_bal)} samples")
else:
    X_train_bal, y_train_bal = X_train, y_train
    print("Skipping SMOTE (not enough samples)")

# %%
# StandardScaler - CRUCIAL for Logistic Regression!
# LR uses gradient descent; unscaled features cause slow convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print("✓ Features standardized (mean=0, std=1)")

# %%
# Train Logistic Regression
# solver='liblinear': Best for small datasets
# class_weight='balanced': Additional imbalance handling
model = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train_bal)
print("✓ Model trained!")

# Training accuracy
train_acc = accuracy_score(y_train_bal, model.predict(X_train_scaled))
print(f"  Training Accuracy: {train_acc:.3f}")

# %% [markdown]
# ---
# ## Step F: Evaluation

# %%
# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# %%
# Metrics
print("="*50)
print("TEST SET METRICS")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

# %%
# Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Depressed']))

# %%
# Confusion Matrix
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Depressed'],
            yticklabels=['Healthy', 'Depressed'],
            annot_kws={'size': 16})
plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %%
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontweight='bold', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step G: Feature Importance (Clinical Insight)
# 
# **This is the most important part!** Shows which acoustic features contribute most to depression prediction.
# 
# - **Positive coefficient**: Feature increases → More likely depressed
# - **Negative coefficient**: Feature increases → More likely healthy

# %%
# Feature importance from coefficients
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0],
    'abs_coef': np.abs(model.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("Top 15 Features:")
print("-" * 45)
for _, row in importance_df.head(15).iterrows():
    arrow = "↑" if row['coefficient'] > 0 else "↓"
    print(f"{arrow} {row['feature']:30} {row['coefficient']:+.4f}")

# %%
# Feature Importance Plot
top_n = 20
top_feats = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in top_feats['coefficient']]

plt.barh(range(len(top_feats)), top_feats['coefficient'].values, color=colors)
plt.yticks(range(len(top_feats)), top_feats['feature'].values)
plt.xlabel('Coefficient Value')
plt.ylabel('Acoustic Feature')
plt.title('Feature Importance for Depression Detection\n(Logistic Regression Coefficients)', 
          fontweight='bold', fontsize=14)

# Legend
from matplotlib.patches import Patch
legend = [Patch(color='#e74c3c', label='↑ Depression likelihood'),
          Patch(color='#3498db', label='↓ Depression likelihood')]
plt.legend(handles=legend, loc='lower right')
plt.axvline(x=0, color='gray', linewidth=0.8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%
# Feature Distributions: Healthy vs Depressed
top_6 = importance_df['feature'].head(6).tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(top_6):
    healthy = features_df[features_df['label'] == 0][feat]
    depressed = features_df[features_df['label'] == 1][feat]
    
    axes[i].hist(healthy, bins=20, alpha=0.6, label='Healthy', color='#3498db', density=True)
    axes[i].hist(depressed, bins=20, alpha=0.6, label='Depressed', color='#e74c3c', density=True)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)
    axes[i].set_title(f'{feat}')

plt.suptitle('Top Feature Distributions by Class', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary
# 
# ### Key Results:
# - Trained a Logistic Regression model on acoustic features
# - Applied StandardScaler and SMOTE for proper preprocessing
# - Generated feature importance for clinical interpretation
# 
# ### Key Features (Depression Indicators):
# | Feature | Direction | Clinical Meaning |
# |---------|-----------|------------------|
# | MFCCs | Varies | Vocal tract characteristics |
# | Pitch_Std | ↓ | Reduced pitch variation (monotone) |
# | RMS_Mean | ↓ | Lower vocal energy |
# | Jitter | ↑ | Voice instability |

# %% [markdown]
# ---
# ## Using with DAIC-WOZ Dataset
# 
# If you have access to DAIC-WOZ, replace the data loading section with:

# %%
def load_daic_woz(daic_path, labels_csv):
    """Load DAIC-WOZ with PHQ-8 based labels."""
    labels = pd.read_csv(labels_csv)
    data = []
    
    for _, row in labels.iterrows():
        audio_file = Path(daic_path) / f"{row['Participant_ID']}_AUDIO.wav"
        if audio_file.exists():
            # Depressed if PHQ-8 >= 10 (clinical threshold)
            label = 1 if row['PHQ8_Score'] >= 10 else 0
            data.append({'file_path': str(audio_file), 'label': label})
    
    return pd.DataFrame(data)

# Usage:
# dataset_df = load_daic_woz('./DAIC-WOZ', './labels.csv')


