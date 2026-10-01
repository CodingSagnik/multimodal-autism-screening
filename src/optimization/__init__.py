"""
src/optimization/__init__.py

Optimization package for Multimodal Early Autism Screening (Phase 3).
Provides Genetic Algorithm (Soft Computing) hyperparameter and modality fusion weight optimization.
"""

from .ga_chromosome import (
    GAChromosome,
    evaluate_chromosome_fitness,
)
from .genetic_algorithm import (
    GeneticAlgorithmEngine,
    latin_hypercube_sampling,
)
from .run_ga import run_pipeline

__all__ = [
    "GAChromosome",
    "evaluate_chromosome_fitness",
    "GeneticAlgorithmEngine",
    "latin_hypercube_sampling",
    "run_pipeline",
]
