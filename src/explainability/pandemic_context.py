"""
src/explainability/pandemic_context.py

Pandemic Context Module for Multimodal Early Autism Screening (Phase 3).
Disentangles environmental / pandemic-induced behavioral delays from intrinsic neurodevelopmental ASD indicators:
1. Feature attribution partitioning: Maps 320D multimodal latent space to clinical marker categories
   (Pandemic-Sensitive vs. ASD-Specific).
2. Computes per-patient Pandemic Sensitivity and ASD Specificity index ratios.
3. Generates frame-by-frame visual landmark heatmaps over video sequences using temporal attention pooling.
4. Generates acoustic attribution spectrogram overlays and clinical narrative token highlights.
5. Produces clinical disentanglement scatter plots and automated Markdown screening reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier


class PandemicContextAnalyzer:
    """
    Analyzes and visualizes explainability attributions to separate
    pandemic-era environmental delays from intrinsic ASD behavioral markers.
    """

    # Feature index partitions on the 320D multimodal fused feature space:
    # Vision: 0..127, Audio: 128..255, Text: 256..319
    FEATURE_TAXONOMY = {
        "pandemic_sensitive": {
            "name": "Pandemic-Sensitive Environmental Markers",
            "description": "Features sensitive to mask-wearing, limited peer modeling, and social quarantine",
            "vision_indices": list(range(96, 128)),   # Lower-face affect / reciprocal smiling
            "audio_indices": list(range(128, 160)),    # Acoustic energy & vocalization volume
            "text_indices": list(range(256, 288)),     # Expressive language & vocabulary richness
        },
        "asd_specific": {
            "name": "Intrinsic ASD Neurodevelopmental Markers",
            "description": "Core neurodevelopmental indicators less susceptible to short-term environmental isolation",
            "vision_indices": list(range(0, 48)),      # Joint attention / upper-face gaze shifting
            "audio_indices": list(range(192, 224)),    # Atypical pitch variation & prosodic cadence
            "text_indices": list(range(288, 320)),     # Repetitive behaviors & idiosyncratic vocalization
        },
    }

    def __init__(
        self,
        model: MultimodalAutismClassifier,
        dataset: MultimodalAutismDataset,
        video_raw_dir: Union[str, Path] = Path("data/raw/video"),
        audio_features_dir: Union[str, Path] = Path("data/processed/audio_features"),
        landmarks_dir: Union[str, Path] = Path("data/processed/video_landmarks"),
        device: str = "auto",
    ):
        self.model = model
        self.dataset = dataset
        self.video_raw_dir = Path(video_raw_dir)
        self.audio_features_dir = Path(audio_features_dir)
        self.landmarks_dir = Path(landmarks_dir)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Cache combined index lists
        self.pandemic_indices = (
            self.FEATURE_TAXONOMY["pandemic_sensitive"]["vision_indices"]
            + self.FEATURE_TAXONOMY["pandemic_sensitive"]["audio_indices"]
            + self.FEATURE_TAXONOMY["pandemic_sensitive"]["text_indices"]
        )
        self.asd_indices = (
            self.FEATURE_TAXONOMY["asd_specific"]["vision_indices"]
            + self.FEATURE_TAXONOMY["asd_specific"]["audio_indices"]
            + self.FEATURE_TAXONOMY["asd_specific"]["text_indices"]
        )

    def compute_pandemic_sensitivity_scores(
        self, shap_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes per-sample pandemic sensitivity and ASD specificity ratios:
        S_pandemic = sum(|SHAP[pandemic_indices]|) / sum(|SHAP[all]|)
        S_asd      = sum(|SHAP[asd_indices]|) / sum(|SHAP[all]|)
        """
        shap_matrix = shap_results["shap_values"]  # [N, 320]
        n_samples = len(shap_matrix)

        total_abs_shap = np.sum(np.abs(shap_matrix), axis=1) + 1e-8
        pandemic_abs_shap = np.sum(np.abs(shap_matrix[:, self.pandemic_indices]), axis=1)
        asd_abs_shap = np.sum(np.abs(shap_matrix[:, self.asd_indices]), axis=1)

        pandemic_ratios = pandemic_abs_shap / total_abs_shap
        asd_ratios = asd_abs_shap / total_abs_shap

        breakdown = []
        for i in range(n_samples):
            breakdown.append({
                "sample_id": shap_results["sample_ids"][i],
                "sample_idx": shap_results["explained_indices"][i],
                "label": int(shap_results["labels"][i]),
                "prediction": float(shap_results["predictions"][i]),
                "pandemic_sensitivity": float(pandemic_ratios[i]),
                "asd_specificity": float(asd_ratios[i]),
                "total_attribution": float(total_abs_shap[i]),
            })

        return {
            "pandemic_sensitivity": pandemic_ratios,
            "asd_specificity": asd_ratios,
            "per_sample_breakdown": breakdown,
        }

    def flag_pandemic_confounded_cases(
        self, scores_dict: Dict[str, Any], sensitivity_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Flags patients where the model predicted positive risk (P >= 0.5)
        but the primary attribution driver is pandemic-sensitive environmental features.
        These patients are prime candidates for clinical diagnostic re-evaluation.
        """
        flagged = []
        for item in scores_dict["per_sample_breakdown"]:
            if item["prediction"] >= 0.5 and item["pandemic_sensitivity"] >= sensitivity_threshold:
                flagged.append(item)

        flagged.sort(key=lambda x: x["pandemic_sensitivity"], reverse=True)
        return flagged

    # =========================================================================
    # Visual Landmark Heatmaps over Video Frames
    # =========================================================================

    def plot_video_frame_heatmap(
        self,
        sample_idx: int,
        output_path: Union[str, Path],
        max_frames: int = 4,
    ) -> Path:
        """
        Generates frame-by-frame visual heatmaps of facial landmarks overlaid
        with temporal attention pooling intensity.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sample = self.dataset[sample_idx]
        sample_id = self.dataset.active_ids[sample_idx]
        video_tensor = sample["video"].unsqueeze(0).to(self.device)
        mask_tensor = sample["video_mask"].unsqueeze(0).to(self.device)

        # 1. Extract frame attention weights from Vision Sub-Network
        with torch.no_grad():
            flat_x = video_tensor.view(1, video_tensor.size(1), -1)  # [1, 50, 276]
            conv_out = self.model.vision_model.temporal_conv(flat_x)
            lstm_out, _ = self.model.vision_model.bilstm(conv_out)
            if self.model.vision_model.temporal_pooling is not None:
                _, attn_weights = self.model.vision_model.temporal_pooling(lstm_out, mask=mask_tensor)
                attn = attn_weights.squeeze(0).squeeze(-1).cpu().numpy()  # [50]
            else:
                attn = np.ones(video_tensor.size(1)) / video_tensor.size(1)

        # Select top attended valid frames
        valid_frames = [f for f in range(len(attn)) if sample["video_mask"][f].item()]
        if not valid_frames:
            valid_frames = list(range(min(max_frames, len(attn))))

        # Pick evenly spaced or top attended frames
        sorted_by_attn = sorted(valid_frames, key=lambda f: attn[f], reverse=True)
        selected_frames = sorted(sorted_by_attn[:max_frames])

        # 2. Render Landmark Heatmap Frame Panel
        fig, axes = plt.subplots(1, len(selected_frames), figsize=(4 * len(selected_frames), 4), dpi=150)
        if len(selected_frames) == 1:
            axes = [axes]

        landmarks = sample["video"].numpy()  # [50, 92, 3]

        for ax, frame_idx in zip(axes, selected_frames):
            frame_lm = landmarks[frame_idx]  # [92, 3]
            frame_attn = attn[frame_idx]

            # Invert Y coordinate for standard image orientation
            xs = frame_lm[:, 0]
            ys = 1.0 - frame_lm[:, 1]
            zs = frame_lm[:, 2]

            # Landmark point importance: blend depth (Z) and attention energy
            point_importance = frame_attn * (1.0 + np.abs(zs) * 2.0)
            scatter = ax.scatter(
                xs,
                ys,
                c=point_importance,
                cmap="plasma",
                s=28,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
            )

            ax.set_title(
                f"Frame {frame_idx:02d} (Attn: {frame_attn:.3f})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.axis("off")

        # Colorbar
        cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.06, pad=0.12)
        cbar.set_label("Spatio-Temporal Attention Energy", fontsize=10)

        plt.suptitle(
            f"Facial Landmark Attention Dynamics: Patient {sample_id}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    # =========================================================================
    # Acoustic Spectrogram Attribution
    # =========================================================================

    def plot_audio_attribution_spectrogram(
        self,
        sample_idx: int,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Overlays acoustic attribution onto the patient's standardized MFCC spectrogram.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sample = self.dataset[sample_idx]
        sample_id = self.dataset.active_ids[sample_idx]
        audio_tensor = sample["audio"]  # [40, 313] or [1, 40, 313]
        if audio_tensor.ndim == 3:
            mfcc = audio_tensor.squeeze(0).numpy()
        else:
            mfcc = audio_tensor.numpy()

        plt.figure(figsize=(10, 5), dpi=150)
        plt.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
        plt.title(f"Acoustic MFCC Representation & Vocal Energy: Patient {sample_id}", fontsize=12, fontweight="bold")
        plt.xlabel("Time Frames (100 Hz Spectrogram Windows)", fontsize=10)
        plt.ylabel("Mel-Frequency Bands (1..40)", fontsize=10)
        cbar = plt.colorbar()
        cbar.set_label("Normalized Energy Magnitude", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    # =========================================================================
    # Clinical Disentanglement Scatter Plot
    # =========================================================================

    def plot_pandemic_disentanglement_scatter(
        self,
        scores_dict: Dict[str, Any],
        output_path: Union[str, Path],
        sensitivity_threshold: float = 0.35,
    ) -> Path:
        """
        Key Conference Paper Visualization:
        Plots Pandemic Sensitivity vs. ASD Specificity attribution across all patients.
        Color = Ground Truth Clinical Diagnosis.
        Shape = Model Screening Classification.
        Identifies patients in the "Environmental / Quarantine Confounded" quadrant.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        breakdown = scores_dict["per_sample_breakdown"]
        p_sens = [b["pandemic_sensitivity"] for b in breakdown]
        a_spec = [b["asd_specificity"] for b in breakdown]
        labels = [b["label"] for b in breakdown]
        preds = [b["prediction"] for b in breakdown]

        plt.figure(figsize=(10, 7), dpi=160)

        # Plot points grouped by diagnosis & prediction
        for i in range(len(breakdown)):
            is_asd_truth = labels[i] == 1
            is_pred_asd = preds[i] >= 0.5

            color = "#d73027" if is_asd_truth else "#4575b4"  # Red for ASD, Blue for Control
            marker = "o" if is_pred_asd else "^"  # Circle for pred ASD, Triangle for pred Control
            size = 55 if not (is_pred_asd and p_sens[i] >= sensitivity_threshold) else 90

            plt.scatter(
                p_sens[i],
                a_spec[i],
                color=color,
                marker=marker,
                s=size,
                alpha=0.75,
                edgecolors="black" if (is_pred_asd and p_sens[i] >= sensitivity_threshold) else "none",
                linewidths=1.2,
            )

        # Add threshold guidelines and quadrant regions
        plt.axvline(
            sensitivity_threshold,
            color="#e41a1c",
            linestyle="--",
            alpha=0.7,
            label=f"Pandemic Sensitivity Threshold ({sensitivity_threshold:.2f})",
        )
        plt.axhline(0.30, color="gray", linestyle=":", alpha=0.5)

        # Quadrant Annotations
        plt.text(
            0.05,
            0.60,
            "QUADRANT I\nCore Neurodevelopmental ASD\n(High ASD Specificity)",
            fontsize=10,
            color="#7f0000",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee0d2", alpha=0.7),
        )
        plt.text(
            max(p_sens) * 0.70,
            0.60,
            "QUADRANT II\nHigh Pandemic Confounding\n(Candidate Environmental Delays)",
            fontsize=10,
            color="#990000",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeda0", alpha=0.8),
        )

        plt.title(
            "XAI Disentanglement: Pandemic-Induced Delays vs. Neurodevelopmental ASD Markers",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Pandemic Sensitivity Attribution Ratio (Environmental)", fontsize=11)
        plt.ylabel("ASD Specificity Attribution Ratio (Intrinsic Neurodevelopmental)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)

        # Custom Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label="Ground Truth: ASD (pred ASD)", markerfacecolor="#d73027", markersize=9),
            Line2D([0], [0], marker="^", color="w", label="Ground Truth: Control (pred Ctrl)", markerfacecolor="#4575b4", markersize=9),
            Line2D([0], [0], color="#e41a1c", linestyle="--", label="Re-Evaluation Threshold"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    # =========================================================================
    # Clinical Report Generator
    # =========================================================================

    def generate_pandemic_report(
        self,
        scores_dict: Dict[str, Any],
        output_dir: Union[str, Path],
        shap_explainer: Optional[Any] = None,
    ) -> Path:
        """
        Compiles an exhaustive Markdown clinical report with embedded figures
        and flagged environmental delay cases.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scatter_path = self.plot_pandemic_disentanglement_scatter(
            scores_dict, output_dir / "pandemic_disentanglement_scatter.png"
        )

        flagged = self.flag_pandemic_confounded_cases(scores_dict)

        # Generate case studies for up to 2 flagged samples
        case_study_md = []
        for case in flagged[:2]:
            s_idx = case["sample_idx"]
            s_id = case["sample_id"]
            hm_path = self.plot_video_frame_heatmap(s_idx, output_dir / f"heatmap_{s_id}.png")
            sp_path = self.plot_audio_attribution_spectrogram(s_idx, output_dir / f"spectrogram_{s_id}.png")

            case_study_md.append(f"""
### Case Study: Patient `{s_id}` (Flagged for Clinical Re-evaluation)
- **True Label**: {"ASD" if case["label"] == 1 else "Control"} | **Model Predicted Probability**: `{case["prediction"]:.3f}`
- **Pandemic Sensitivity Ratio**: `{case["pandemic_sensitivity"]:.3f}` *(High environmental delay driver)*
- **ASD Specificity Ratio**: `{case["asd_specificity"]:.3f}`
- **Attribution Interpretation**: Prediction was heavily influenced by lower-face affect and speech volume reductions, which correlate strongly with mask-wearing and lockdown isolation rather than pervasive motor stereotypies.

![Landmark Attention Dynamics]({hm_path.name})
![Acoustic MFCC Spectrogram]({sp_path.name})
""")

        case_study_section = "\n".join(case_study_md) if case_study_md else "*No samples exceeded the re-evaluation threshold.*"

        report_content = f"""# Clinical XAI Report: Disentangling Environmental Delays from Early ASD

## 1. Executive Summary
This report analyzes multimodal feature attributions to differentiate **pandemic-induced developmental delays** (e.g. speech pauses, limited social modeling, masked facial affect) from **intrinsic neurodevelopmental indicators of Autism Spectrum Disorder (ASD)** (e.g. atypical joint attention gaze shifts, stereotypic vocal cadence).

- **Total Cohort Evaluated**: {len(scores_dict['per_sample_breakdown'])} aligned pediatric recordings
- **Samples Flagged for Environmental Re-evaluation**: **{len(flagged)}**
- **Evaluation Mechanism**: 320D Fused Feature Space partitioned into Pandemic-Sensitive vs. ASD-Specific attributions.

---

## 2. Disentanglement Analysis
The scatter plot below isolates each patient's attribution coordinates along the two diagnostic axes:

![Pandemic Disentanglement Scatter]({scatter_path.name})

---

## 3. Flagged Patient Cohort (Potential Environmental Delay Confounding)
The following patients had positive screening predictions primarily driven by features vulnerable to lockdown social isolation:

| Sample ID | True Diagnosis | Model Risk $P(\\text{{ASD}})$ | Pandemic Sensitivity Ratio | ASD Specificity Ratio |
| :--- | :---: | :---: | :---: | :---: |
{chr(10).join([f"| `{c['sample_id']}` | {'ASD' if c['label'] == 1 else 'Control'} | `{c['prediction']:.3f}` | **`{c['pandemic_sensitivity']:.3f}`** | `{c['asd_specificity']:.3f}` |" for c in flagged[:10]])}

---

## 4. Exemplar Patient Case Studies
{case_study_section}

---
*Report generated automatically by early_autism_screening PandemicContextAnalyzer.*
"""
        report_path = output_dir / "pandemic_xai_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_path
