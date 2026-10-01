"""
src/explainability/shap_explainer.py

SHAP (SHapley Additive exPlanations) Module for Multimodal Autism Late Fusion Classifier (Phase 3).
Operates at the fused latent representation level (320D):
- Extracts pre-fusion latent representations (Vision=128D, Audio=128D, Text=64D).
- Wraps the late fusion classification head for model-agnostic KernelExplainer.
- Computes additive feature attributions and aggregates importance across modalities.
- Generates publication-grade explainability plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier


class MultimodalSHAPExplainer:
    """
    Computes SHAP values on the multimodal fused latent space to explain
    predictions of the MultimodalAutismClassifier.
    """

    def __init__(
        self,
        model: MultimodalAutismClassifier,
        dataset: MultimodalAutismDataset,
        modality_weights: Optional[Sequence[float]] = None,
        n_background_samples: int = 40,
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

        # Generate feature names
        self.feature_names: List[str] = (
            [f"vision_{i:03d}" for i in range(self.v_dim)]
            + [f"audio_{i:03d}" for i in range(self.a_dim)]
            + ([f"text_{i:02d}" for i in range(self.t_dim)] if self.use_text else [])
        )

        # Extract background samples
        self.background_features = self._extract_background_features()
        self.explainer = shap.KernelExplainer(
            self._predict_from_fused, self.background_features
        )

    def _extract_latent_features(
        self, indices: Optional[Sequence[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts concatenated fused feature vectors [N, fused_dim] and labels [N].
        """
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

    def _extract_background_features(self) -> np.ndarray:
        """Extracts background distribution using stratified sampling."""
        labels = [self.dataset.labels_map.get(sid, 0) for sid in self.dataset.active_ids]
        rng = np.random.default_rng(self.random_seed)

        pos_indices = [i for i, l in enumerate(labels) if l == 1]
        neg_indices = [i for i, l in enumerate(labels) if l == 0]

        n_pos_bg = max(1, int(self.n_background_samples * (len(pos_indices) / len(labels))))
        n_neg_bg = max(1, self.n_background_samples - n_pos_bg)

        bg_pos = rng.choice(pos_indices, size=min(n_pos_bg, len(pos_indices)), replace=False)
        bg_neg = rng.choice(neg_indices, size=min(n_neg_bg, len(neg_indices)), replace=False)
        bg_indices = list(bg_pos) + list(bg_neg)

        features, _ = self._extract_latent_features(bg_indices)
        return features

    def _predict_from_fused(self, fused_np: np.ndarray) -> np.ndarray:
        """
        Prediction function wrapper for SHAP KernelExplainer:
        takes [N, fused_dim] numpy array -> returns [N] predicted risk probabilities.
        """
        self.model.eval()
        fused_tensor = torch.from_numpy(fused_np).float().to(self.device)
        with torch.no_grad():
            logits = self.model.fusion_head(fused_tensor)
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs])
        return probs

    def compute_shap_values(
        self,
        indices: Optional[Sequence[int]] = None,
        n_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Computes SHAP attributions for the specified sample indices.

        Returns structured dictionary containing per-feature and per-modality attributions.
        """
        explain_indices = (
            list(indices) if indices is not None else list(range(len(self.dataset)))
        )
        fused_vecs, labels = self._extract_latent_features(explain_indices)

        # KernelExplainer computes SHAP
        shap_vals = self.explainer.shap_values(fused_vecs, nsamples=n_samples)
        if isinstance(shap_vals, list):
            shap_matrix = np.array(shap_vals[0])
        else:
            shap_matrix = np.array(shap_vals)

        # Modality slicing
        v_slice = slice(0, self.v_dim)
        a_slice = slice(self.v_dim, self.v_dim + self.a_dim)
        t_slice = slice(self.v_dim + self.a_dim, self.fused_dim)

        v_shap = shap_matrix[:, v_slice]
        a_shap = shap_matrix[:, a_slice]
        t_shap = shap_matrix[:, t_slice] if self.use_text else np.zeros((len(explain_indices), 1))

        # Modality-level aggregates (mean absolute SHAP per sample)
        modality_shap = {
            "vision": np.sum(np.abs(v_shap), axis=1),
            "audio": np.sum(np.abs(a_shap), axis=1),
            "text": np.sum(np.abs(t_shap), axis=1) if self.use_text else np.zeros(len(explain_indices)),
        }

        mean_abs_modality = {
            "vision": float(np.mean(modality_shap["vision"])),
            "audio": float(np.mean(modality_shap["audio"])),
            "text": float(np.mean(modality_shap["text"])) if self.use_text else 0.0,
        }

        # Calculate model predictions for explained samples
        preds = self._predict_from_fused(fused_vecs)

        base_val = (
            float(self.explainer.expected_value)
            if np.isscalar(self.explainer.expected_value)
            else float(self.explainer.expected_value[0])
        )

        return {
            "shap_values": shap_matrix,
            "fused_features": fused_vecs,
            "labels": labels,
            "predictions": preds,
            "base_value": base_val,
            "feature_names": self.feature_names,
            "modality_shap": modality_shap,
            "mean_abs_modality": mean_abs_modality,
            "explained_indices": explain_indices,
            "sample_ids": [self.dataset.active_ids[i] for i in explain_indices],
        }

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_modality_importance(
        self, results: Dict[str, Any], save_path: Union[str, Path]
    ) -> None:
        """
        Generates bar chart of global modality importance (Mean |SHAP| value).
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        mean_imp = results["mean_abs_modality"]
        modalities = ["Vision\n(Landmarks)", "Audio\n(Acoustic CNN)", "Text\n(Clinical NLP)"]
        values = [mean_imp["vision"], mean_imp["audio"], mean_imp["text"]]
        colors = ["#2b5c8f", "#d95f02", "#7570b3"]

        plt.figure(figsize=(8, 5), dpi=150)
        bars = plt.bar(modalities, values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.title("Global Modality Importance (Mean |SHAP| Attribution)", fontsize=13, fontweight="bold")
        plt.ylabel("Mean Absolute SHAP Value", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_shap_summary(
        self, results: Dict[str, Any], save_path: Union[str, Path], top_k: int = 15
    ) -> None:
        """
        Generates summary bar plot for top-K influential fusion latent dimensions.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        shap_vals = results["shap_values"]
        names = results["feature_names"]
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:top_k]

        top_names = [names[i] for i in top_indices][::-1]
        top_scores = [mean_abs[i] for i in top_indices][::-1]

        # Color-code by modality
        bar_colors = []
        for n in top_names:
            if n.startswith("vision"):
                bar_colors.append("#2b5c8f")
            elif n.startswith("audio"):
                bar_colors.append("#d95f02")
            else:
                bar_colors.append("#7570b3")

        plt.figure(figsize=(9, 7), dpi=150)
        plt.barh(top_names, top_scores, color=bar_colors, edgecolor="black", linewidth=0.8)
        plt.title(f"Top-{top_k} Discriminative Latent Dimensions (SHAP)", fontsize=13, fontweight="bold")
        plt.xlabel("Mean |SHAP Value|", fontsize=11)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_sample_waterfall(
        self,
        results: Dict[str, Any],
        sample_idx: int,
        save_path: Union[str, Path],
        top_k: int = 10,
    ) -> None:
        """
        Generates individual waterfall/bar explanation for a single pediatric sample.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        idx = results["explained_indices"].index(sample_idx)
        sample_shap = results["shap_values"][idx]
        names = results["feature_names"]
        pred = results["predictions"][idx]
        true_label = results["labels"][idx]
        sample_id = results["sample_ids"][idx]

        top_indices = np.argsort(np.abs(sample_shap))[::-1][:top_k]
        top_names = [names[i] for i in top_indices][::-1]
        top_vals = [sample_shap[i] for i in top_indices][::-1]
        colors = ["#d73027" if v > 0 else "#4575b4" for v in top_vals]

        plt.figure(figsize=(9, 6), dpi=150)
        plt.barh(top_names, top_vals, color=colors, edgecolor="black", linewidth=0.8)
        label_text = "ASD" if true_label == 1 else "Control"
        plt.title(
            f"Local Attribution: {sample_id} (True: {label_text}, Pred: {pred:.3f})",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("SHAP Attribution (Red=Pushes toward ASD, Blue=Toward Control)", fontsize=10)
        plt.axvline(0, color="black", linewidth=1.0)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
