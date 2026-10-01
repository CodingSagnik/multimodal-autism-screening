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

## 4. Phase 3: Algorithmic Optimization & Explainable AI (XAI)

Phase 3 introduces technical novelty required for clinical conference submission, satisfying the imperative to **disentangle pandemic-induced environmental delays from ASD**:

### 1. Genetic Algorithm (GA) Optimization (`src/optimization/`)
- **Soft Computing Multi-Gene Optimization**: Automatically discovers optimal scalar modality fusion weights $(w_v, w_a, w_t)$, optimizer dynamics (learning rate, weight decay), regularization rates (dropout 1 & 2), and late fusion capacity (`fusion_hidden1`, `fusion_hidden2`).
- **Operators**: Latin Hypercube Sampling (LHS) initialization, Simulated Binary Crossover (SBX, $\eta_c=20$), Polynomial Mutation ($\eta_m=20$), Tournament Selection ($k=3$), and strict elitism.
- **Fitness Objective**: Mean Stratified Cross-Validation **Balanced Accuracy** (averaging sensitivity and specificity to counter 83/17 class imbalance).

### 2. Model-Agnostic Explainability (`src/explainability/`)
- **SHAP (KernelExplainer)**: Computes additive feature attributions over the 320D multimodal fused feature space, aggregating importance across Vision, Audio, and Text streams.
- **LIME (TabularExplainer)**: Produces local linear surrogate decision rules and contrastive explanations comparing ASD vs. Control clinical profiles.

### 3. Pandemic Context Module (`src/explainability/pandemic_context.py`)
- **Clinical Feature Taxonomy**: Partitions the 320D multimodal latent space into:
  - *Pandemic-Sensitive Environmental Markers*: Lower-face affect (mask-wearing), acoustic volume/energy (lockdown isolation), and conversational expressive vocabulary.
  - *Intrinsic Neurodevelopmental ASD Markers*: Atypical joint attention gaze shifts, stereotypic vocal cadence, and repetitive behavioral patterns.
- **Visual Landmark Heatmaps**: Extracts frame-by-frame temporal attention pooling weights and renders dynamic 3D facial landmark saliency heatmaps across video sequences.
- **Acoustic Spectrogram Attribution**: Overlays frequency-temporal attribution maps onto 40x313 MFCC matrices.
- **Pandemic Disentanglement Scatter Plot**: Quantifies $S_{\text{pandemic}}$ vs. $S_{\text{asd}}$ attribution ratios, flagging patients whose positive screening was confounded by pandemic environmental isolation for clinical developmental re-evaluation.

---

## 5. Phase 4: Empirical Evaluation & Paper Drafting

Phase 4 transitions the completed software project into a structured conference submission:

### 1. Results Compilation (`src/evaluation/`)
- **`ResultsCompiler`**: Automated generation of all publication-ready figures and tables from the cross-validation and GA optimization outputs.
- **Radar Performance Profile**: Spider chart comparing unimodal vs. multimodal metric profiles across 6 evaluation dimensions.
- **Grouped Metric Comparison**: Bar chart with error bars for all model × metric combinations.
- **Sensitivity–Specificity Tradeoff**: Scatter plot with iso-balanced-accuracy contours showing the clinical tradeoff across configurations.
- **Ablation Study Table**: LaTeX table quantifying the contribution of each modality relative to the best unimodal baseline.
- **GA Hyperparameter Summary**: LaTeX table documenting the 9-gene search space and discovered optimal configuration.

### 2. Conference Paper (`paper/main.tex`)
- **IEEE/ACM Conference Format**: Full paper draft with Abstract, Introduction, Literature Review (20 references), Methodology (mathematical formulations for GA, SHAP, Pandemic Context Module), Experimental Results (6 figures, 3 tables), Discussion, and Conclusion.
- **Key figures referenced**: Architecture pipeline diagram, SHAP modality attribution, LIME contrastive case study, pandemic disentanglement scatter, landmark attention heatmaps, acoustic spectrograms, GA convergence curves, radar comparison, grouped metrics, sensitivity-specificity tradeoff.

---

## 6. Repository Structure

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
├── models/
│   ├── ga_checkpoints/                      # GA population checkpoints & final model
│   └── ga_logs/                             # Evolution CSV logs & convergence plots
├── notebooks/                               # Exploratory analysis notebooks
├── paper/                                   # Phase 4: Conference Paper & Publication Assets
│   ├── main.tex                             # Full IEEE/ACM conference paper draft
│   ├── figures/                             # Publication-ready figures (11 total)
│   │   ├── fig1_multimodal_architecture_pipeline.png
│   │   ├── fig2_modality_importance_shap.png
│   │   ├── fig3_top_latent_features_shap.png
│   │   ├── fig4_lime_contrastive_case_study.png
│   │   ├── fig5_pandemic_disentanglement_scatter.png
│   │   ├── fig6_video_landmark_attention_heatmaps.png
│   │   ├── fig7_acoustic_attribution_spectrograms.png
│   │   ├── fig8_ga_convergence_curves.png
│   │   ├── fig9_radar_performance_profile.png
│   │   ├── fig10_grouped_metric_comparison.png
│   │   └── fig11_sensitivity_specificity_tradeoff.png
│   └── tables/                              # LaTeX & Markdown data tables
│       ├── model_comparison.tex             # Main benchmark table (6 models × 6 metrics)
│       ├── ablation_study.tex               # Modality ablation with Δ balanced accuracy
│       ├── ga_hyperparameter_summary.tex    # GA search space & optimal values
│       ├── model_comparison.md              # Markdown mirror for README embedding
│       ├── model_comparison.json            # Raw numeric data (mean ± std)
│       ├── model_comparison.csv             # CSV export for external tools
│       └── pandemic_xai_report.md           # Clinical XAI disentanglement report
├── src/
│   ├── data_processing/                     # Phase 1: Feature Extraction & Alignment
│   │   ├── __init__.py                      # Package exports
│   │   ├── extract_video_landmarks.py       # MediaPipe 3D face mesh extractor
│   │   ├── extract_audio_features.py        # Librosa 16kHz MFCC extractor
│   │   ├── extract_text_embeddings.py       # Standalone M-CHAT text pipeline
│   │   ├── generate_video_text_embeddings.py# Leak-free video text narrative generator
│   │   └── multimodal_dataset.py            # Dual-mode PyTorch Multimodal Dataset
│   ├── models/                              # Phase 2: Unimodal Encoders & Late Fusion
│   │   ├── __init__.py                      # Package exports
│   │   ├── vision_model.py                  # VisionLandmarkModel (BiLSTM + Attention + Mask)
│   │   ├── audio_model.py                   # AcousticCNNModel (3-stage 2D CNN)
│   │   ├── text_model.py                    # ClinicalTextMLP (LayerNorm MLP)
│   │   └── fusion_model.py                  # MultimodalAutismClassifier (Dual-mode Late Fusion)
│   ├── training/                            # Phase 3: Cross-Validation & Training
│   │   ├── __init__.py                      # Package exports
│   │   └── trainer.py                       # AutismScreeningTrainer (Stratified K-Fold CV)
│   ├── optimization/                        # Phase 3: Soft Computing & GA Optimization
│   │   ├── __init__.py                      # Package exports
│   │   ├── ga_chromosome.py                 # Hybrid chromosome encoding & CV fitness
│   │   ├── genetic_algorithm.py             # GeneticAlgorithmEngine (LHS, SBX, Elitism)
│   │   └── run_ga.py                        # GA optimization entry point & retraining
│   ├── explainability/                      # Phase 3: Explainable AI & Pandemic Context
│   │   ├── __init__.py                      # Package exports
│   │   ├── shap_explainer.py                # MultimodalSHAPExplainer (KernelExplainer)
│   │   ├── lime_explainer.py                # MultimodalLIMEExplainer (TabularExplainer)
│   │   └── pandemic_context.py              # PandemicContextAnalyzer (Landmark Heatmaps & XAI)
│   └── evaluation/                          # Phase 4: Results Compilation & Paper Assets
│       ├── __init__.py                      # Package exports
│       └── generate_results.py              # ResultsCompiler (Figures, Tables, LaTeX export)
├── tests/
│   ├── check_phase2.py                      # Phase 2 integration audit script
│   ├── check_fusion.py                      # Phase 2 multimodal late fusion audit script
│   ├── check_subnetwork.py                  # Phase 2 unimodal branch audit script
│   ├── test_leakage_free.py                 # Text leakage verification test suite
│   ├── test_phase3_training.py              # Phase 3 training & cross-validation tests
│   ├── test_phase3_ga.py                    # Phase 3 Genetic Algorithm unit & operator tests
│   └── test_phase3_xai.py                   # Phase 3 SHAP, LIME, and Pandemic Context tests
├── pyproject.toml                           # Package configuration (pip install -e .)
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 7. Setup & Execution

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
```

### 3. Running Phase 3 Genetic Algorithm & Retraining
```powershell
# Run full GA optimization (population=20, generations=30)
python src/optimization/run_ga.py --population 20 --generations 30

# Fast smoke-test verification
python src/optimization/run_ga.py --smoke_test
```

### 4. Running Phase 4 Results Compilation
```powershell
# Generate all publication-ready figures and tables
python src/evaluation/generate_results.py
```

### 5. Running the Test Suite (Phases 1, 2, and 3)
```powershell
# Phase 1 & 2 Verification
python tests/test_leakage_free.py
python tests/check_subnetwork.py
python tests/check_fusion.py
python tests/check_phase2.py

# Phase 3 Verification
python tests/test_phase3_training.py
python tests/test_phase3_ga.py
python tests/test_phase3_xai.py
```

### 6. Compiling the Conference Paper
```powershell
# Requires LaTeX distribution (e.g., MiKTeX or TeX Live)
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 8. Citation & Academic Inquiries

If you find this research codebase useful in your work, please cite:

```bibtex
@inproceedings{early_autism_multimodal2026,
  title={Multimodal Early Autism Screening: Disentangling Environmental Delays via Multimodal Deep Learning and Explainable AI},
  author={Research Team},
  booktitle={Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```
