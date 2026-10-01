"""
src/explainability/lime_explainer.py

LIME (Local Interpretable Model-agnostic Explanations) Module for Multimodal Autism Late Fusion (Phase 3).
Provides:
- Tabular LIME explanations over the 320D multimodal fused latent space.
- Modality-level attribution breakdown for local decisions.
- Direct side-by-side contrastive explanations between ASD and Control clinical profiles.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier


class MultimodalLIMEExplainer:
    """
    Computes local linear surrogate explanations (LIME) on the multimodal
    fused feature space of MultimodalAutismClassifier.
    """

    def __init__(
        self,
        model: MultimodalAutismClassifier,
        dataset: MultimodalAutismDataset,
        modality_weights: Optional[Sequence[float]] = None,
        n_background_samples: int = 50,
        random_seed: int = 42,
        device: str = "auto",
    ):
        self.model = model
        self.dataset = dataset
        self.modality_weights = tuple(modality_weights) if modality_weights is not None else None
        self.n_background_samples = min(n_background_samples, len(dataset))
        self.random_seed = random_seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.use_text = model.use_text
        self.v_dim = model.vision_dim
        self.a_dim = model.audio_dim
        self.t_dim = model.text_dim
        self.fused_dim = model.fused_dim

        # Feature names
        self.feature_names = (
            [f"vision_{i:03d}" for i in range(self.v_dim)]
            + [f"audio_{i:03d}" for i in range(self.a_dim)]
            + ([f"text_{i:02d}" for i in range(self.t_dim)] if self.use_text else [])
        )

        # Background feature matrix for LIME distribution modeling
        self.background_features, _ = self._extract_latent_features(
            list(range(min(self.n_background_samples, len(dataset))))
        )

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.background_features,
            feature_names=self.feature_names,
            class_names=["Control", "ASD"],
            mode="classification",
            random_state=self.random_seed,
            discretize_continuous=True,
        )

    def _extract_latent_features(
        self, indices: Optional[Sequence[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        sample_indices = (
            list(indices) if indices is not None else list(range(len(self.dataset)))
        )
        loader = DataLoader(
            torch.utils.data.Subset(self.dataset, sample_indices),
            batch_size=16,
            shuffle=False,
        )

        all_fused: List[np.ndarray] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in loader:
                video = batch["video"].to(self.device)
                audio = batch["audio"].to(self.device)
                text = batch.get("text", None)
                if text is not None:
                    text = text.to(self.device)
                video_mask = batch.get("video_mask", None)
                if video_mask is not None:
                    video_mask = video_mask.to(self.device)
                labels = batch["label"]

                _, sub = self.model(
                    video,
                    audio,
                    text,
                    video_mask=video_mask,
                    modality_weights=self.modality_weights,
                    return_sub_features=True,
                )
                fused = sub["fused"].cpu().numpy()
                all_fused.append(fused)
                all_labels.extend(labels.numpy().astype(int).tolist())

        return np.vstack(all_fused), np.array(all_labels)

    def _predict_proba_from_fused(self, fused_np: np.ndarray) -> np.ndarray:
        """
        Predicts 2-class probability distribution [P(Control), P(ASD)] for LIME.
        """
        self.model.eval()
        fused_tensor = torch.from_numpy(fused_np).float().to(self.device)
        with torch.no_grad():
            logits = self.model.fusion_head(fused_tensor)
            p_asd = torch.sigmoid(logits.squeeze(1)).cpu().numpy()

        if p_asd.ndim == 0:
            p_asd = np.array([p_asd])
        p_ctrl = 1.0 - p_asd
        return np.column_stack([p_ctrl, p_asd])

    def explain_sample(
        self, sample_idx: int, num_features: int = 15
    ) -> Dict[str, Any]:
        """
        Computes LIME explanation for a single pediatric sample.
        """
        fused_vec, labels = self._extract_latent_features([sample_idx])
        vec = fused_vec[0]
        true_label = labels[0]
        sample_id = self.dataset.active_ids[sample_idx]

        probs = self._predict_proba_from_fused(fused_vec)[0]

        exp = self.explainer.explain_instance(
            data_row=vec,
            predict_fn=self._predict_proba_from_fused,
            num_features=num_features,
            labels=(1,),  # Explain ASD class (1)
        )

        feature_weights = exp.as_list(label=1)

        # Modality contribution sums
        v_contrib = sum(abs(w) for name, w in feature_weights if "vision" in name)
        a_contrib = sum(abs(w) for name, w in feature_weights if "audio" in name)
        t_contrib = sum(abs(w) for name, w in feature_weights if "text" in name)

        return {
            "sample_idx": sample_idx,
            "sample_id": sample_id,
            "true_label": int(true_label),
            "label_name": "ASD" if true_label == 1 else "Control",
            "prob_control": float(probs[0]),
            "prob_asd": float(probs[1]),
            "feature_weights": feature_weights,
            "modality_contributions": {
                "vision": float(v_contrib),
                "audio": float(a_contrib),
                "text": float(t_contrib),
            },
            "lime_explanation_obj": exp,
        }

    def explain_batch(
        self, sample_indices: Sequence[int], num_features: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Generates LIME explanations across a batch of sample indices.
        """
        return [self.explain_sample(idx, num_features=num_features) for idx in sample_indices]

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_sample_explanation(
        self, explanation: Dict[str, Any], save_path: Union[str, Path]
    ) -> None:
        """
        Plots horizontal bar chart of local LIME decision rules.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        features = [f[0] for f in explanation["feature_weights"]][::-1]
        weights = [f[1] for f in explanation["feature_weights"]][::-1]
        colors = ["#d73027" if w > 0 else "#4575b4" for w in weights]

        plt.figure(figsize=(10, 6), dpi=150)
        plt.barh(features, weights, color=colors, edgecolor="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=1.0)
        plt.title(
            f"LIME Explanation: {explanation['sample_id']} "
            f"(True: {explanation['label_name']} | P(ASD)={explanation['prob_asd']:.3f})",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("LIME Feature Weight (Red=Supports ASD, Blue=Supports Control)", fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def compare_asd_vs_control(
        self, asd_idx: int, control_idx: int, save_path: Union[str, Path]
    ) -> None:
        """
        Produces a side-by-side comparative diagnostic explanation between
        an ASD patient and a Control patient.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        exp_asd = self.explain_sample(asd_idx, num_features=10)
        exp_ctrl = self.explain_sample(control_idx, num_features=10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Plot ASD
        feats_asd = [f[0] for f in exp_asd["feature_weights"]][::-1]
        w_asd = [f[1] for f in exp_asd["feature_weights"]][::-1]
        c_asd = ["#d73027" if w > 0 else "#4575b4" for w in w_asd]
        ax1.barh(feats_asd, w_asd, color=c_asd, edgecolor="black", linewidth=0.8)
        ax1.axvline(0, color="black", linewidth=0.8)
        ax1.set_title(
            f"ASD Patient: {exp_asd['sample_id']}\nP(ASD) = {exp_asd['prob_asd']:.3f}",
            fontweight="bold",
            fontsize=11,
        )
        ax1.set_xlabel("Feature Weight", fontsize=10)
        ax1.grid(axis="x", linestyle="--", alpha=0.5)

        # Plot Control
        feats_ctrl = [f[0] for f in exp_ctrl["feature_weights"]][::-1]
        w_ctrl = [f[1] for f in exp_ctrl["feature_weights"]][::-1]
        c_ctrl = ["#d73027" if w > 0 else "#4575b4" for w in w_ctrl]
        ax2.barh(feats_ctrl, w_ctrl, color=c_ctrl, edgecolor="black", linewidth=0.8)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_title(
            f"Control Patient: {exp_ctrl['sample_id']}\nP(ASD) = {exp_ctrl['prob_asd']:.3f}",
            fontweight="bold",
            fontsize=11,
        )
        ax2.set_xlabel("Feature Weight", fontsize=10)
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        plt.suptitle("Multimodal LIME Contrastive Analysis: ASD vs. Control", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
