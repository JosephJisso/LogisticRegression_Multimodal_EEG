# Audio-Based Depression Detection

Upload your audio feature extraction code here.

## Expected Files
- `preprocessing.py` - Audio loading and preprocessing
- `feature_extraction.py` - Extract 41 acoustic features
- `train_model.py` - Model training with SMOTE
- `inference.py` - Audio analysis and prediction

## Features (41 total)
- MFCCs (1-13): Mean and Std
- Prosodic: Pitch Mean, Std, Range
- Voice Quality: Jitter, Shimmer
- Temporal: ZCR, RMS Energy
- Spectral: Centroid, Bandwidth, Rolloff

## Dataset
- Primary: DAIC-WOZ
- Fallback: RAVDESS
