# Empirical Benchmark: Unimodal vs. Multimodal Screening Models

| Model Architecture | Bal. Acc | Macro F1 | ROC-AUC | PR-AUC | Sensitivity (ASD) | Specificity (Ctrl) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Unimodal Vision (3D Landmarks) | 0.632 ± 0.045 | 0.431 ± 0.104 | 0.701 ± 0.039 | 0.930 ± 0.016 | 0.364 ± 0.176 | 0.900 ± 0.133 |
| Unimodal Audio (Acoustic CNN) | 0.840 ± 0.061 | 0.721 ± 0.077 | 0.879 ± 0.093 | 0.965 ± 0.033 | 0.747 ± 0.082 | 0.933 ± 0.082 |
| Unimodal Text (Whisper ASR) | 0.697 ± 0.124 | 0.503 ± 0.196 | 0.726 ± 0.105 | 0.923 ± 0.040 | 0.461 ± 0.266 | 0.933 ± 0.082 |
| Bimodal Late Fusion (Vision + Audio) | 0.817 ± 0.093 | 0.728 ± 0.115 | 0.862 ± 0.088 | 0.963 ± 0.029 | 0.775 ± 0.102 | 0.860 ± 0.127 |
| Trimodal Late Fusion (Equal Weights) | 0.726 ± 0.168 | 0.604 ± 0.282 | 0.829 ± 0.087 | 0.952 ± 0.032 | 0.618 ± 0.354 | 0.833 ± 0.149 |
| Trimodal Late Fusion (GA-Optimized) | 0.806 ± 0.041 | 0.680 ± 0.033 | 0.863 ± 0.080 | 0.965 ± 0.025 | 0.711 ± 0.068 | 0.900 ± 0.133 |

*All models evaluated via 5-Fold Stratified Cross-Validation on the AV-ASD cohort (N=171).*
