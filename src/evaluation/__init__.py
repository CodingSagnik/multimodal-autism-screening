"""
src/evaluation/__init__.py

Evaluation and benchmarking package for Multimodal Early Autism Screening (Phase 4).
Provides:
- Rigorous unimodal vs. multimodal cross-validation benchmarking.
- Publication-quality LaTeX and CSV table generation.
- High-resolution XAI figure generation and paper artifact exports.
"""

from src.evaluation.benchmark_models import ModelBenchmarkEvaluator
from src.evaluation.export_paper_assets import export_all_paper_assets
from src.evaluation.generate_results import ResultsCompiler

__all__ = [
    "ModelBenchmarkEvaluator",
    "ResultsCompiler",
    "export_all_paper_assets",
]
