"""
src/training/__init__.py

Training pipeline package for Multimodal Early Autism Screening (Phase 3).
Provides stratified cross-validation, class-balanced loss, and comprehensive evaluation metrics.
"""

from .trainer import AutismScreeningTrainer

__all__ = ["AutismScreeningTrainer"]
