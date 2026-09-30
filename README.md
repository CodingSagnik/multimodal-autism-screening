# Multimodal Early Autism Screening: Disentangling Environmental Delays via Multimodal Deep Learning and Explainable AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0-007acc.svg)](https://developers.google.com/mediapipe)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Domain](https://img.shields.io/badge/Domain-AI%20for%20Health%20%7C%20Neurodevelopment-success.svg)](#)

---

## 1. Project Overview

This repository houses the research codebase for **Multimodal Early Autism Screening**, an advanced machine learning framework designed for pediatric behavioral screening. The system integrates synchronized physiological, behavioral, and clinical streams:

- **Computer Vision (Video)**: Extracts 3D spatial dynamics of facial landmarks focused on eye gaze, joint attention, and facial affect using **MediaPipe Face Mesh**, with **temporal attention padding masks** to handle variable-length sequences.
- **Acoustic Signal Processing (Audio)**: Analyzes pediatric vocalizations and prosodic speech patterns through standardized **Mel-Frequency Cepstral Coefficients (MFCCs)** via 2D Convolutional neural networks.
- **Clinical Natural Language Processing (Text)**: Encodes **Whisper ASR speech transcriptions** of raw audio clips using **DistilBERT**, providing genuine multimodal text signal from vocalization and speech content with zero diagnostic data leakage.

The architecture operates in **dual-mode**:
- **3-Modality Screening**: Vision (128D) + Audio (128D) + Text (64D) $\rightarrow$ **320D** Fused Feature Space.
- **2-Modality Screening**: Vision (128D) + Audio (128D) $\rightarrow$ **256D** Fused Feature Space.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 RAW MULTIMODAL STREAMS                 │
                  └───────┬───────────────────┬───────────────────┬────────┘
                          │                   │                   │
                          ▼                   ▼                   ▼
                  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
                  │ Pediatric     │   │ Audio Clips   │   │ Whisper ASR   │
                  │ Videos (.mp4) │   │ (.wav)        │   │ Transcripts   │
                  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                          │                   │                   │
    [Phase 1]             ▼ (5.0 FPS)         ▼ (16 kHz, 10s)     ▼ (Tokenizer)
  Preprocessing   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  & Alignment     │ MediaPipe     │   │ Librosa       │   │ Whisper ASR + │
                  │ Face Mesh     │   │ MFCC (40x313) │   │ DistilBERT    │
                  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                          │                   │                   │
                          └─────────────┬─────┴───────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   MultimodalAutismDataset     │
                        │    (171 Aligned Samples)      │
                        │  + Temporal Attention Masks   │
                        └───────────────┬───────────────┘
                                        │
    [Phase 2-3]                         ▼
    Downstream          ┌───────────────────────────────┐
    Modeling            │  Modular Feature Encoders     │
    & Fusion            │  (Vision + Audio + [Text])    │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │ Late Fusion (320D or 256D)    │
                        │   + Dynamic Modality Weights  │
                        │   + XAI (SHAP & LIME) Hooks   │
                        └───────────────────────────────┘
```

---

## 2. Dataset Architecture & Leakage-Free Design

The repository cleanly delineates two distinct datasets to avoid cross-cohort contamination and data leakage:

### 1. AV-ASD Multimodal Benchmark (171 Aligned Clips)
- **171 Aligned Tri-Stream Samples**: Exactly 171 video landmark sequences, 171 audio MFCC matrices, and 171 dense text embeddings with 1-to-1 sample correspondence.
- **Leakage-Free Clinical Narratives**: Generated via `src/data_processing/generate_video_text_embeddings.py`. Text records combine neutral recording metadata with **Whisper ASR speech transcriptions** of the raw audio clips, providing genuine multimodal text signal from vocalization and speech patterns without any diagnostic label injection. Verified through linear probes demonstrating sub-$0.95$ cross-validation accuracy.
- **Temporal Attention Masking**: All video sequences pad/truncate to $T=50$ frames, generating a boolean `video_mask` (`True` for valid frames, `False` for padding) so zero-padding does not dilute attention pooling.

### 2. M-CHAT-R Standalone Clinical Cohort (6,075 Patients)
- Located in `data/processed/mchat_standalone/` (`mchat_embedded.pt` and `mchat_embedded.csv`).
- Evaluated as a standalone pediatric NLP/tabular baseline, completely decoupled from the AV-ASD video fusion pipeline.

---

## 3. Modular Architecture Summary (Phase 2)

| Component / Sub-Network | Architecture Description | Latent Representation | Trainable Parameters |
| :--- | :--- | :--- | :---: |
| **Vision Sub-Network** (`VisionLandmarkModel`) | Conv1d (GroupNorm) + 2-layer BiLSTM + Temporal Attention Pooling with sequence validity mask + LayerNorm | 128D Embedding | 330,113 |
| **Audio Sub-Network** (`AcousticCNNModel`) | 3-stage 2D CNN (GroupNorm) Spectrogram Extractor + Adaptive Pooling + LayerNorm | 128D Embedding | 109,793 |
| **Text Sub-Network** (`ClinicalTextMLP`) | 3-stage regularized LayerNorm MLP compressing DistilBERT embeddings | 64D Embedding | 238,977 |
| **Late Fusion Head** (`MultimodalAutismClassifier`) | Multi-stage MLP with LayerNorm, dropout, and dynamic modality scaling | 320D $\rightarrow$ 1 Logit (3M) <br> 256D $\rightarrow$ 1 Logit (2M) | 45,569 |
| **Total Model (3-Modality)** | End-to-end tri-stream classifier (Vision + Audio + Text) | 320D Fused Vector | **724,452** |
| **Total Model (2-Modality)** | Dual-mode audiovisual classifier (Vision + Audio) | 256D Fused Vector | **477,283** |

### Robustness & Normalization
All CNN backbone layers use **GroupNorm** (batch-size-independent), and all projection layers and the fusion classifier use **LayerNorm**, guaranteeing seamless, error-free operation on single-sample inputs (`batch_size=1`) during both training and real-time clinical screening inference.

### Class Imbalance Handling
The dataset provides built-in class imbalance utilities via `compute_pos_weight()` (for `BCEWithLogitsLoss`) and `get_sampler_weights()` (for `WeightedRandomSampler`), addressing the 83%/17% ASD/Control split.

---

## 4. Repository Structure

```
early_autism_screening/
├── data/
│   ├── raw/                                 # Raw source data (excluded from git)
│   │   ├── video/                           # Raw .mp4 video clips
│   │   ├── audio/                           # Raw .wav audio clips
│   │   ├── text/                            # Raw mchat_results.csv
│   │   └── AV-ASD_repo/                     # AV-ASD annotations and metadata
│   └── processed/                           # Aligned feature tensors (excluded)
│       ├── video_landmarks/                 # 3D landmark arrays (.npy, 171 files)
│       ├── audio_features/                  # Standardized 40x313 MFCC matrices (171 files)
│       ├── text_embeddings/                 # Leak-free DistilBERT embeddings (171 files)
│       └── mchat_standalone/                # Standalone M-CHAT cohort (6,075 records)
├── models/                                  # Pre-trained models and detector assets
├── notebooks/                               # Exploratory analysis notebooks
├── src/
│   ├── data_processing/
│   │   ├── __init__.py                      # Package exports
│   │   ├── extract_video_landmarks.py       # MediaPipe 3D face mesh extractor
│   │   ├── extract_audio_features.py        # Librosa 16kHz MFCC extractor
│   │   ├── extract_text_embeddings.py       # Standalone M-CHAT text pipeline
│   │   ├── generate_video_text_embeddings.py# Leak-free video text narrative generator
│   │   └── multimodal_dataset.py            # Dual-mode PyTorch Multimodal Dataset
│   └── models/
│       ├── __init__.py                      # Package exports
│       ├── vision_model.py                  # VisionLandmarkModel (BiLSTM + Attention + Mask)
│       ├── audio_model.py                   # AcousticCNNModel (3-stage 2D CNN)
│       ├── text_model.py                    # ClinicalTextMLP (LayerNorm MLP)
│       └── fusion_model.py                  # MultimodalAutismClassifier (Dual-mode Late Fusion)
├── tests/
│   ├── check_phase2.py                      # Integration audit script
│   ├── check_fusion.py                      # Multimodal late fusion audit script
│   ├── check_subnetwork.py                  # Unimodal branch verification audit
│   └── test_leakage_free.py                 # Automated text leakage test suite
├── pyproject.toml                           # Package configuration (pip install -e .)
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 5. Setup & Installation

### 1. Environment Setup
```powershell
# Create Conda virtual environment
conda create -n autism_screening python=3.10 -y
conda activate autism_screening

# Install dependencies and editable project package
pip install -r requirements.txt
pip install -e .
```

### 2. Running Data Generation & Verification
```powershell
# Generate leak-free clinical narratives and text embeddings for AV-ASD clips
python src/data_processing/generate_video_text_embeddings.py

# Verify multimodal dataset alignment & temporal attention padding masks
python src/data_processing/multimodal_dataset.py

# Run the complete test suite
python tests/test_leakage_free.py
python tests/check_subnetwork.py
python tests/check_fusion.py
python tests/check_phase2.py
```

---

## 6. Citation & Academic Inquiries

If you find this research codebase useful in your work, please cite:

```bibtex
@inproceedings{early_autism_multimodal2026,
  title={Multimodal Early Autism Screening: Disentangling Environmental Delays via Multimodal Deep Learning and Explainable AI},
  author={Research Team},
  booktitle={Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```
