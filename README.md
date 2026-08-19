# Multimodal Early Autism Screening: Disentangling Environmental Delays via Multimodal Deep Learning and Explainable AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0-007acc.svg)](https://developers.google.com/mediapipe)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Domain](https://img.shields.io/badge/Domain-AI%20for%20Health%20%7C%20Neurodevelopment-success.svg)](#)

---

## 1. Project Overview

This repository houses the research codebase for **Multimodal Early Autism Screening**, an advanced machine learning framework designed for pediatric behavioral screening. The system integrates three synchronized physiological and behavioral streams:

- **Computer Vision (Video)**: Extracts 3D spatial dynamics of facial landmarks focused on eye gaze, joint attention, and facial affect using **MediaPipe Face Mesh**.
- **Acoustic Signal Processing (Audio)**: Analyzes pediatric vocalizations and prosodic speech patterns through standardized **Mel-Frequency Cepstral Coefficients (MFCCs)**.
- **Clinical Natural Language Processing (Text)**: Encodes structured clinical narratives from the **M-CHAT-R (Modified Checklist for Autism in Toddlers, Revised)** questionnaire into high-dimensional semantic representations using **DistilBERT / ClinicalBERT**.

By capturing cross-modal correlations, the framework provides an objective, computational paradigm for early pediatric neurodevelopmental risk assessment.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 RAW MULTIMODAL STREAMS                 │
                  └───────┬───────────────────┬───────────────────┬────────┘
                          │                   │                   │
                          ▼                   ▼                   ▼
                  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
                  │ Pediatric     │   │ Audio Clips   │   │ M-CHAT-R      │
                  │ Videos (.mp4) │   │ (.wav)        │   │ Questionnaires│
                  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                          │                   │                   │
    [Phase 1]             ▼ (5.0 FPS)         ▼ (16 kHz, 10s)     ▼ (Tokenizer)
  Preprocessing   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  & Alignment     │ MediaPipe     │   │ Librosa       │   │ HuggingFace   │
                  │ Face Mesh     │   │ MFCC (40x313) │   │ DistilBERT    │
                  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                          │                   │                   │
                          └─────────────┬─────┴───────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   MultimodalAutismDataset     │
                        │    (171 Aligned Samples)      │
                        └───────────────┬───────────────┘
                                        │
    [Phase 2-3]                         ▼
    Downstream          ┌───────────────────────────────┐
    Modeling            │  Modular Feature Encoders     │
    & Fusion            │   (Vision + Audio + Text)     │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Late Fusion & GA Weights    │
                        │   + XAI (SHAP & LIME)         │
                        └───────────────────────────────┘
```

---

## 2. The Problem & Clinical Motivation

Early diagnosis of Autism Spectrum Disorder (ASD) before age 3 is vital for optimizing long-term developmental outcomes. However, the post-COVID-19 landscape has introduced significant clinical ambiguity:

- **Pandemic-Induced Environmental Delays**: Toddlers raised during lockdowns experienced prolonged social isolation, disrupted peer interactions, and reduced linguistic exposure due to caregiver masking. Consequently, many present with non-ASD speech delays and reduced joint attention.
- **Diagnostic Bottlenecks**: Standard screening tools (e.g., M-CHAT questionnaires alone) suffer from high false-positive rates when evaluating environmentally isolated children, overwhelming specialized clinical triage.
- **Our Proposed Solution**: Differentiating true neurodevelopmental ASD from environmental developmental delays by combining objective micro-behavioral video landmarks, acoustic prosodic features, and clinical narratives. Leveraging **Explainable AI (XAI)** enables clinicians to isolate feature attributions (e.g., separating auditory inattention from speech latency).

---

## 3. Current Progress: Phase 1 (Complete)

Phase 1 data ingestion, feature standardization, and multimodal synchronization are **100% complete**:

| Modality | Raw Input Source | Preprocessing Method | Output Shape / Dimension | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Video** | AV-ASD `.mp4` Clips | MediaPipe Face Mesh at 5.0 FPS (92 Social Engagement Landmarks: Eyes, Mouth, Eyebrows) | `[T, 92, 3]` / `[50, 92, 3]` (`.npy` / `.pt`) | **Complete** |
| **Audio** | AV-ASD `.wav` Audio | Librosa Resampling (16 kHz), Fixed 10s Window (Padding/Truncation), 40 MFCCs | `[40, 313]` (`.npy` / `.pt`) | **Complete** |
| **Text** | M-CHAT-R Responses | Clinical Narrative Synthesis, Casing Normalization, DistilBERT Pooled Embeddings | `[768]` (`.pt` / `.csv`) | **Complete** |
| **Dataset** | Multi-Stream Merging | `MultimodalAutismDataset` with strict unaligned-sample dropping | **171 Aligned Samples** | **Complete** |

### Verified Dataset Summary
- Total video files processed: **171**
- Total audio files processed: **172** (1 unaligned record successfully filtered out)
- Total M-CHAT text records: **6,075** + per-clip behavioral narratives
- Aligned multimodal dataset size: **171 1-to-1 matched multimodal samples**

---

## 4. Research Roadmap

```
├── Phase 1: Data Ingestion, Extraction & PyTorch Dataset Alignment  [✓ COMPLETED]
├── Phase 2: Unimodal Modular Encoders & Late Fusion Integration      [IN PROGRESS]
├── Phase 3: Soft Computing (Genetic Algorithm) & Explainable AI     [PLANNED]
└── Phase 4: Benchmark Evaluation & Conference Manuscript Drafting   [PLANNED]
```

### Phase 2: Modular Architectures & Late Fusion Integration
- **Vision Sub-Network**: Spatio-temporal modeling over 3D landmark trajectories using a 1D Temporal Convolutional Network (TCN) or Bidirectional LSTM with self-attention.
- **Audio Sub-Network**: 2D ResNet / 1D-CNN over MFCC spectrograms to capture prosodic pitch variations and acoustic inflection.
- **Text Sub-Network**: Dense projection layers fine-tuned on clinical semantic embeddings.
- **Fusion Layer**: Multimodal late-fusion network fusing unimodal latent representations with cross-modal attention.

### Phase 3: Soft Computing Optimization & Explainability (XAI)
- **Genetic Algorithm (GA) Weight Optimization**: Employing evolutionary algorithms to optimize fusion layer weights and loss penalty coefficients, combating class imbalance.
- **Model Interpretability (SHAP & LIME)**: Computing Shapley values and local surrogate models to highlight which behavioral modalities contributed to screening predictions, explicitly isolating environmental delay markers.

### Phase 4: Empirical Evaluation & Manuscript Preparation
- Stratified 5-Fold Cross-Validation, sensitivity, specificity, and Area Under the ROC Curve (AUC-ROC).
- Ablation studies (Vision-only vs. Audio-only vs. Text-only vs. Multimodal Fusion).
- Submission to top-tier health informatics / AI in medicine conference.

---

## 5. Repository Structure

```
early_autism_screening/
├── data/
│   ├── raw/                           # Raw datasets (excluded from git)
│   │   ├── video/                     # Raw .mp4 video clips
│   │   ├── audio/                     # Raw .wav audio clips
│   │   ├── text/                      # mchat_results.csv & clinical files
│   │   └── AV-ASD_repo/               # AV-ASD annotations and metadata
│   └── processed/                     # Standardized feature tensors (excluded)
│       ├── video_landmarks/           # Extracted 3D landmark arrays (.npy/.pt)
│       ├── audio_features/            # Standardized 40x313 MFCC matrices (.npy/.pt)
│       └── text_embeddings/           # DistilBERT pooled embeddings (.pt/.csv)
├── models/                            # MediaPipe models and saved weights (.task)
├── notebooks/                         # Exploratory and diagnostic Jupyter notebooks
├── src/
│   ├── data_processing/
│   │   ├── __init__.py                # Package exports
│   │   ├── extract_video_landmarks.py # MediaPipe 3D face mesh extractor
│   │   ├── extract_audio_features.py  # Librosa 16kHz MFCC extractor
│   │   ├── extract_text_embeddings.py # DistilBERT clinical text embedding pipeline
│   │   └── multimodal_dataset.py      # Custom PyTorch Multimodal Dataset
│   └── models/                        # Downstream neural architectures (Phase 2)
├── .gitignore                         # Comprehensive ignore rules
├── requirements.txt                   # Project Python dependencies
└── README.md                          # Project documentation
```

---

## 6. Setup & Installation

### 1. Conda Environment Setup
```powershell
# Create Conda virtual environment with Python 3.10
conda create -n autism_screening python=3.10 -y

# Activate the environment
conda activate autism_screening

# Install project dependencies
pip install -r requirements.txt
```

### 2. Running Preprocessing Pipelines

Each pipeline is modular and can be executed independently:

```powershell
# 1. Extract 3D facial landmarks from video (5.0 FPS, 92 social landmarks)
python src/data_processing/extract_video_landmarks.py

# 2. Extract standardized MFCC audio matrices (16 kHz, 10s window, 40 coeffs)
python src/data_processing/extract_audio_features.py

# 3. Extract dense clinical text embeddings using DistilBERT
python src/data_processing/extract_text_embeddings.py

# 4. Verify multimodal alignment and test DataLoader batch iteration
python src/data_processing/multimodal_dataset.py
```

---

## 7. Citation & Academic Inquiries

If you find this research codebase useful in your work, please cite:

```bibtex
@inproceedings{early_autism_multimodal2026,
  title={Multimodal Early Autism Screening: Disentangling Environmental Delays via Multimodal Deep Learning and Explainable AI},
  author={Research Team},
  booktitle={Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```
