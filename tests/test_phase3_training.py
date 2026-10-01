"""
tests/test_phase3_training.py

Unit and integration tests for Phase 3 Training Pipeline and Cross-Validation.
Verifies:
1. compute_metrics correctness on synthetic predictions.
2. AutismScreeningTrainer initialization and parameter caching.
3. Single-fold execution with class-imbalanced pos_weight and weighted sampling.
4. Stratified K-fold cross-validation aggregation and metric structure.
5. train_full training on 100% of dataset.
"""

import numpy as np
import torch

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier
from src.training.trainer import AutismScreeningTrainer, compute_metrics


def test_compute_metrics_perfect():
    """Verify metrics for perfect classification."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_probs = np.array([0.9, 0.85, 0.95, 0.1, 0.05])
    m = compute_metrics(y_true, y_probs)

    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_inverted():
    """Verify metrics for completely inverted classification."""
    y_true = np.array([1, 1, 0, 0])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_probs)

    assert m["balanced_accuracy"] == 0.0
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 0.0


def test_trainer_initialization():
    """Verify AutismScreeningTrainer initialization on real dataset."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    trainer = AutismScreeningTrainer(
        model=model,
        dataset=dataset,
        n_folds=3,
        batch_size=16,
        learning_rate=5e-4,
        weight_decay=1e-4,
        num_epochs=2,
        modality_weights=(1.2, 0.9, 1.0),
    )

    assert trainer.n_folds == 3
    assert trainer.batch_size == 16
    assert trainer.modality_weights == (1.2, 0.9, 1.0)
    assert trainer.device is not None


def test_single_fold_training():
    """Test training and validating a single fold."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    trainer = AutismScreeningTrainer(
        model=model,
        dataset=dataset,
        batch_size=32,
        num_epochs=1,
    )

    labels = trainer._get_dataset_labels()
    train_idx = list(range(0, 100))
    val_idx = list(range(100, len(labels)))

    res = trainer.train_one_fold(fold_idx=0, train_indices=train_idx, val_indices=val_idx)

    assert "fold_idx" in res
    assert "best_model_state_dict" in res
    assert "best_metrics" in res
    assert "train_losses" in res
    assert "val_scores" in res
    assert len(res["train_losses"]) == 1


def test_cv_execution():
    """Test running 2-fold cross-validation."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    trainer = AutismScreeningTrainer(
        model=model,
        dataset=dataset,
        n_folds=2,
        batch_size=32,
        num_epochs=1,
    )

    cv_summary = trainer.run_cv()

    assert cv_summary["n_folds"] == 2
    assert len(cv_summary["per_fold_results"]) == 2
    assert "mean_balanced_accuracy" in cv_summary
    assert "std_balanced_accuracy" in cv_summary
    assert "mean_macro_f1" in cv_summary
    assert "mean_roc_auc" in cv_summary
    assert "mean_pr_auc" in cv_summary
    assert 0.0 <= cv_summary["mean_balanced_accuracy"] <= 1.0


def test_train_full():
    """Test training full model on 100% dataset."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    trainer = AutismScreeningTrainer(
        model=model,
        dataset=dataset,
        batch_size=32,
    )

    trained_model, losses = trainer.train_full(num_epochs=1)
    assert len(losses) == 1
    assert isinstance(trained_model, MultimodalAutismClassifier)


if __name__ == "__main__":
    print("Running Phase 3 Training Pipeline Tests...")
    test_compute_metrics_perfect()
    print("  • test_compute_metrics_perfect: PASSED")
    test_compute_metrics_inverted()
    print("  • test_compute_metrics_inverted: PASSED")
    test_trainer_initialization()
    print("  • test_trainer_initialization: PASSED")
    test_single_fold_training()
    print("  • test_single_fold_training: PASSED")
    test_cv_execution()
    print("  • test_cv_execution: PASSED")
    test_train_full()
    print("  • test_train_full: PASSED")
    print("All Phase 3 Training Pipeline Tests PASSED!")
