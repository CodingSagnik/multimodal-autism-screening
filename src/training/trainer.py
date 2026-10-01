"""
src/training/trainer.py

Stratified K-Fold Cross-Validation Trainer for Multimodal Early Autism Screening (Phase 3).
Features:
- Stratified splitting preserving 83/17 ASD vs Control class distribution across folds.
- Class-imbalance mitigation: per-fold pos_weight for BCEWithLogitsLoss and WeightedRandomSampler.
- Comprehensive metric tracking: Balanced Accuracy, Macro F1, PR-AUC, ROC-AUC, Sensitivity, Specificity.
- Early stopping with best checkpoint restoration.
- Supports dynamic scalar modality weighting for GA evaluation.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes clinically relevant evaluation metrics for imbalanced screening:
    - Balanced Accuracy: (Sensitivity + Specificity) / 2
    - Macro-F1: Unweighted average of per-class F1
    - PR-AUC: Area under Precision-Recall curve
    - ROC-AUC: Area under Receiver Operating Characteristic curve
    - Sensitivity (Recall of positive class / ASD)
    - Specificity (Recall of negative class / Control)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # Balanced accuracy & Macro F1
    bal_acc = float(metrics.balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Sensitivity & Specificity
    if n_pos > 0:
        sensitivity = float(metrics.recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        sensitivity = 0.0

    if n_neg > 0:
        specificity = float(metrics.recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    else:
        specificity = 0.0

    # ROC-AUC (safe check for single-class cases)
    if n_pos > 0 and n_neg > 0:
        try:
            roc_auc = float(metrics.roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.5
        try:
            precision, recall, _ = metrics.precision_recall_curve(y_true, y_probs)
            pr_auc = float(metrics.auc(recall, precision))
        except Exception:
            pr_auc = 0.5
    else:
        roc_auc = 0.5
        pr_auc = 0.5

    return {
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


class AutismScreeningTrainer:
    """
    Orchestrates training and cross-validation of the multimodal late fusion classifier.
    """

    def __init__(
        self,
        model: MultimodalAutismClassifier,
        dataset: MultimodalAutismDataset,
        n_folds: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = "auto",
        modality_weights: Optional[Sequence[float]] = None,
        use_class_balanced_loss: bool = True,
        use_weighted_sampler: bool = True,
        random_seed: int = 42,
        verbose: bool = False,
    ):
        self.base_model = model
        self.dataset = dataset
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.modality_weights = tuple(modality_weights) if modality_weights is not None else None
        self.use_class_balanced_loss = use_class_balanced_loss
        self.use_weighted_sampler = use_weighted_sampler
        self.random_seed = random_seed
        self.verbose = verbose

        # Determine target device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Cache initial state for clean fold re-initialization
        self._initial_state_dict = copy.deepcopy(model.state_dict())

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_dataset_labels(self) -> List[int]:
        return [self.dataset.labels_map.get(sid, 0) for sid in self.dataset.active_ids]

    def _evaluate_epoch(
        self, model: nn.Module, loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluates model on validation loader, returning (loss, metrics_dict)."""
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        n_batches = 0
        all_probs: List[float] = []
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
                labels = batch["label"].to(self.device).float()

                logits = model(
                    video,
                    audio,
                    text,
                    video_mask=video_mask,
                    modality_weights=self.modality_weights,
                )
                loss = criterion(logits.squeeze(1), labels)
                total_loss += loss.item()
                n_batches += 1

                probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy().astype(int).tolist())

        mean_loss = total_loss / max(1, n_batches)
        metrics_dict = compute_metrics(np.array(all_labels), np.array(all_probs))
        metrics_dict["loss"] = mean_loss
        return mean_loss, metrics_dict

    def train_one_fold(
        self, fold_idx: int, train_indices: Sequence[int], val_indices: Sequence[int]
    ) -> Dict[str, Any]:
        """
        Trains and validates a single cross-validation fold.
        """
        self._set_seed(self.random_seed + fold_idx)

        # Clone fresh model instance for this fold
        model = copy.deepcopy(self.base_model).to(self.device)
        model.load_state_dict(self._initial_state_dict)

        labels = self._get_dataset_labels()
        train_labels = [labels[i] for i in train_indices]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos

        # Configure loss pos_weight
        if self.use_class_balanced_loss and n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Configure training sampler
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)

        if self.use_weighted_sampler and n_pos > 0 and n_neg > 0:
            w_pos = len(train_labels) / (2.0 * n_pos)
            w_neg = len(train_labels) / (2.0 * n_neg)
            sample_weights = torch.tensor(
                [w_pos if l == 1 else w_neg for l in train_labels], dtype=torch.float32
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(train_indices), replacement=True
            )
            train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True
            )

        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_score = -1.0  # Track balanced accuracy
        best_state = None
        best_metrics: Dict[str, float] = {}
        epochs_no_improve = 0
        train_losses: List[float] = []
        val_scores: List[float] = []

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                video = batch["video"].to(self.device)
                audio = batch["audio"].to(self.device)
                text = batch.get("text", None)
                if text is not None:
                    text = text.to(self.device)
                video_mask = batch.get("video_mask", None)
                if video_mask is not None:
                    video_mask = video_mask.to(self.device)
                batch_labels = batch["label"].to(self.device).float()

                optimizer.zero_grad()
                logits = model(
                    video,
                    audio,
                    text,
                    video_mask=video_mask,
                    modality_weights=self.modality_weights,
                )
                loss = criterion(logits.squeeze(1), batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            epoch_train_loss = running_loss / max(1, n_batches)
            train_losses.append(epoch_train_loss)

            # Evaluate on validation fold
            _, val_metrics = self._evaluate_epoch(model, val_loader)
            current_score = val_metrics["balanced_accuracy"]
            val_scores.append(current_score)

            if self.verbose:
                logger.info(
                    f"Fold {fold_idx+1} | Epoch {epoch:02d}/{self.num_epochs:02d} | "
                    f"Train Loss: {epoch_train_loss:.4f} | "
                    f"Val Bal Acc: {current_score:.4f} | "
                    f"Macro F1: {val_metrics['macro_f1']:.4f}"
                )

            # Check for improvement
            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = copy.deepcopy(val_metrics)
                best_metrics["best_epoch"] = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    if self.verbose:
                        logger.info(f"Fold {fold_idx+1}: Early stopping triggered at epoch {epoch}.")
                    break

        return {
            "fold_idx": fold_idx,
            "best_model_state_dict": best_state,
            "best_metrics": best_metrics,
            "train_losses": train_losses,
            "val_scores": val_scores,
        }

    def run_cv(self) -> Dict[str, Any]:
        """
        Executes full Stratified K-Fold cross validation.
        """
        labels = self._get_dataset_labels()
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_seed
        )

        fold_results = []
        best_cv_score = -1.0
        best_overall_state = None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(labels)), labels)):
            res = self.train_one_fold(fold_idx, train_idx, val_idx)
            fold_results.append(res)
            fold_score = res["best_metrics"].get("balanced_accuracy", 0.0)
            if fold_score > best_cv_score:
                best_cv_score = fold_score
                best_overall_state = res["best_model_state_dict"]

        # Aggregate metric distributions
        metric_keys = ["balanced_accuracy", "macro_f1", "roc_auc", "pr_auc", "sensitivity", "specificity"]
        summary: Dict[str, Any] = {
            "n_folds": self.n_folds,
            "per_fold_results": fold_results,
            "best_fold_model_state_dict": best_overall_state,
        }

        for key in metric_keys:
            vals = [f["best_metrics"].get(key, 0.0) for f in fold_results]
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))

        return summary

    def train_full(
        self, num_epochs: Optional[int] = None
    ) -> Tuple[MultimodalAutismClassifier, List[float]]:
        """
        Trains model on 100% of the dataset (e.g. for final artifact deployment).
        Returns (trained_model, train_loss_history).
        """
        self._set_seed(self.random_seed)
        model = copy.deepcopy(self.base_model).to(self.device)
        model.load_state_dict(self._initial_state_dict)

        labels = self._get_dataset_labels()
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        if self.use_class_balanced_loss and n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        if self.use_weighted_sampler and n_pos > 0 and n_neg > 0:
            sampler_weights = self.dataset.get_sampler_weights()
            sampler = WeightedRandomSampler(
                weights=sampler_weights, num_samples=len(self.dataset), replacement=True
            )
            loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        epochs = num_epochs if num_epochs is not None else self.num_epochs
        train_losses: List[float] = []

        model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            n_batches = 0
            for batch in loader:
                video = batch["video"].to(self.device)
                audio = batch["audio"].to(self.device)
                text = batch.get("text", None)
                if text is not None:
                    text = text.to(self.device)
                video_mask = batch.get("video_mask", None)
                if video_mask is not None:
                    video_mask = video_mask.to(self.device)
                batch_labels = batch["label"].to(self.device).float()

                optimizer.zero_grad()
                logits = model(
                    video,
                    audio,
                    text,
                    video_mask=video_mask,
                    modality_weights=self.modality_weights,
                )
                loss = criterion(logits.squeeze(1), batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(1, n_batches)
            train_losses.append(epoch_loss)

        return model, train_losses
