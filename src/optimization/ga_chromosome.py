"""
src/optimization/ga_chromosome.py

Genetic Algorithm Chromosome Representation and Fitness Evaluation for Multimodal Late Fusion.
Encodes:
1. Modality weights (w_vision, w_audio, w_text) in [0.0, 2.0]
2. Optimizer parameters: learning_rate (log-scale), weight_decay (log-scale)
3. Regularization parameters: dropout1, dropout2
4. Late fusion architecture: fusion_hidden1, fusion_hidden2
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier
from src.training.trainer import AutismScreeningTrainer

# Discrete choice definitions
HIDDEN1_CHOICES = [64, 128, 256]
HIDDEN2_CHOICES = [16, 32, 64]


@dataclass
class GAChromosome:
    """
    Candidate individual encoding modality fusion weights, training hyperparameters,
    and late fusion architectural capacity.
    """

    # 1. Modality Weights
    w_vision: float = 1.0
    w_audio: float = 1.0
    w_text: float = 1.0

    # 2. Training Hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout1: float = 0.4
    dropout2: float = 0.2

    # 3. Architecture Hyperparameters
    fusion_hidden1: int = 128
    fusion_hidden2: int = 32

    # Evaluated Fitness (Mean Validation Balanced Accuracy)
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_modality_weights(self) -> Tuple[float, float, float]:
        """Returns tuple of scalar weights (w_v, w_a, w_t)."""
        return (float(self.w_vision), float(self.w_audio), float(self.w_text))

    def build_model(self, use_text: bool = True) -> MultimodalAutismClassifier:
        """
        Instantiates a MultimodalAutismClassifier with the architecture
        and dropout parameters encoded in this chromosome.
        """
        return MultimodalAutismClassifier(
            use_text=use_text,
            fusion_hidden1=int(self.fusion_hidden1),
            fusion_hidden2=int(self.fusion_hidden2),
            dropout1=float(self.dropout1),
            dropout2=float(self.dropout2),
        )

    def to_gene_vector(self) -> np.ndarray:
        """
        Encodes the chromosome into a normalized real-valued vector in [0.0, 1.0]^9.
        """
        g0 = np.clip(self.w_vision / 2.0, 0.0, 1.0)
        g1 = np.clip(self.w_audio / 2.0, 0.0, 1.0)
        g2 = np.clip(self.w_text / 2.0, 0.0, 1.0)

        # log10(1e-5) = -5.0, log10(1e-2) = -2.0 -> span = 3.0
        log_lr = math.log10(max(1e-5, min(1e-2, self.learning_rate)))
        g3 = np.clip((log_lr - (-5.0)) / 3.0, 0.0, 1.0)

        # log10(1e-6) = -6.0, log10(1e-2) = -2.0 -> span = 4.0
        log_wd = math.log10(max(1e-6, min(1e-2, self.weight_decay)))
        g4 = np.clip((log_wd - (-6.0)) / 4.0, 0.0, 1.0)

        # dropout1 in [0.1, 0.6] -> span = 0.5
        g5 = np.clip((self.dropout1 - 0.1) / 0.5, 0.0, 1.0)

        # dropout2 in [0.05, 0.40] -> span = 0.35
        g6 = np.clip((self.dropout2 - 0.05) / 0.35, 0.0, 1.0)

        # Discrete choices mapped to [0, 1) bins
        idx1 = (
            HIDDEN1_CHOICES.index(self.fusion_hidden1)
            if self.fusion_hidden1 in HIDDEN1_CHOICES
            else 1
        )
        g7 = (idx1 + 0.5) / len(HIDDEN1_CHOICES)

        idx2 = (
            HIDDEN2_CHOICES.index(self.fusion_hidden2)
            if self.fusion_hidden2 in HIDDEN2_CHOICES
            else 1
        )
        g8 = (idx2 + 0.5) / len(HIDDEN2_CHOICES)

        return np.array([g0, g1, g2, g3, g4, g5, g6, g7, g8], dtype=np.float64)

    @classmethod
    def from_gene_vector(
        cls, genes: Union[Sequence[float], np.ndarray]
    ) -> "GAChromosome":
        """
        Decodes a normalized real-valued vector in [0.0, 1.0]^9 back into a GAChromosome.
        """
        g = np.clip(np.asarray(genes, dtype=np.float64), 0.0, 1.0)
        assert len(g) >= 9, f"Expected gene vector of length >= 9, got {len(g)}"

        w_v = float(g[0] * 2.0)
        w_a = float(g[1] * 2.0)
        w_t = float(g[2] * 2.0)

        lr = float(10.0 ** (-5.0 + g[3] * 3.0))
        wd = float(10.0 ** (-6.0 + g[4] * 4.0))

        do1 = float(0.1 + g[5] * 0.5)
        do2 = float(0.05 + g[6] * 0.35)

        idx1 = min(len(HIDDEN1_CHOICES) - 1, int(g[7] * len(HIDDEN1_CHOICES)))
        hidden1 = HIDDEN1_CHOICES[idx1]

        idx2 = min(len(HIDDEN2_CHOICES) - 1, int(g[8] * len(HIDDEN2_CHOICES)))
        hidden2 = HIDDEN2_CHOICES[idx2]

        return cls(
            w_vision=w_v,
            w_audio=w_a,
            w_text=w_t,
            learning_rate=lr,
            weight_decay=wd,
            dropout1=do1,
            dropout2=do2,
            fusion_hidden1=hidden1,
            fusion_hidden2=hidden2,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes chromosome parameters to dictionary."""
        def _sanitize(val):
            if isinstance(val, (int, float, str, bool)) or val is None:
                return val
            elif isinstance(val, (np.floating, np.integer)):
                return float(val)
            elif isinstance(val, dict):
                return {
                    k: _sanitize(v)
                    for k, v in val.items()
                    if k not in ("best_fold_model_state_dict", "best_model_state_dict")
                }
            elif isinstance(val, (list, tuple)):
                return [_sanitize(x) for x in val]
            return str(val)

        return {
            "w_vision": float(self.w_vision),
            "w_audio": float(self.w_audio),
            "w_text": float(self.w_text),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "dropout1": float(self.dropout1),
            "dropout2": float(self.dropout2),
            "fusion_hidden1": int(self.fusion_hidden1),
            "fusion_hidden2": int(self.fusion_hidden2),
            "fitness": float(self.fitness) if self.fitness is not None else None,
            "metadata": _sanitize(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAChromosome":
        """Reconstructs chromosome from dictionary."""
        return cls(
            w_vision=float(data.get("w_vision", 1.0)),
            w_audio=float(data.get("w_audio", 1.0)),
            w_text=float(data.get("w_text", 1.0)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            dropout1=float(data.get("dropout1", 0.4)),
            dropout2=float(data.get("dropout2", 0.2)),
            fusion_hidden1=int(data.get("fusion_hidden1", 128)),
            fusion_hidden2=int(data.get("fusion_hidden2", 32)),
            fitness=float(data["fitness"]) if data.get("fitness") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


def evaluate_chromosome_fitness(
    chromosome: GAChromosome,
    dataset: MultimodalAutismDataset,
    n_folds: int = 3,
    num_epochs: int = 25,
    batch_size: int = 16,
    random_seed: int = 42,
    device: str = "auto",
    early_stopping_patience: int = 6,
    verbose: bool = False,
) -> float:
    """
    Evaluates fitness of a chromosome using stratified K-fold cross-validation.
    Fitness = Mean Balanced Accuracy across folds.
    """
    model = chromosome.build_model()
    trainer = AutismScreeningTrainer(
        model=model,
        dataset=dataset,
        n_folds=n_folds,
        batch_size=batch_size,
        learning_rate=chromosome.learning_rate,
        weight_decay=chromosome.weight_decay,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        modality_weights=chromosome.get_modality_weights(),
        random_seed=random_seed,
        verbose=verbose,
    )

    cv_results = trainer.run_cv()
    fitness = float(cv_results.get("mean_balanced_accuracy", 0.0))
    chromosome.fitness = fitness
    chromosome.metadata = {
        k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
        for k, v in cv_results.items()
        if k not in ("best_fold_model_state_dict", "per_fold_results")
    }
    return fitness
