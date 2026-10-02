# Empirical Benchmark: Unimodal vs. Multimodal Screening Models

| Model Architecture | Bal. Acc | Macro F1 | ROC-AUC | PR-AUC | Sensitivity (ASD) | Specificity (Ctrl) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Unimodal Vision (3D Landmarks) | 0.630 ± 0.060 | 0.436 ± 0.145 | 0.708 ± 0.039 | 0.921 ± 0.028 | 0.393 ± 0.236 | 0.867 ± 0.163 |
| Unimodal Audio (Acoustic CNN) | 0.829 ± 0.049 | 0.703 ± 0.059 | 0.885 ± 0.092 | 0.969 ± 0.029 | 0.726 ± 0.066 | 0.933 ± 0.082 |
| Unimodal Text (Whisper ASR) | 0.697 ± 0.124 | 0.503 ± 0.196 | 0.726 ± 0.105 | 0.923 ± 0.040 | 0.461 ± 0.266 | 0.933 ± 0.082 |
| Bimodal Late Fusion (Vision + Audio) | 0.782 ± 0.059 | 0.650 ± 0.088 | 0.861 ± 0.080 | 0.968 ± 0.020 | 0.664 ± 0.117 | 0.900 ± 0.082 |
| Trimodal Late Fusion (Equal Weights) | 0.723 ± 0.161 | 0.559 ± 0.245 | 0.842 ± 0.101 | 0.950 ± 0.042 | 0.547 ± 0.298 | 0.900 ± 0.133 |
| Trimodal Late Fusion (GA-Optimized) | 0.841 ± 0.079 | 0.741 ± 0.095 | 0.859 ± 0.083 | 0.964 ± 0.026 | 0.782 ± 0.094 | 0.900 ± 0.133 |

*All models evaluated via Stratified Cross-Validation on the AV-ASD cohort ($N=171$).*
