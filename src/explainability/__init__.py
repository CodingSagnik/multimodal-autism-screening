"""
src/explainability/__init__.py

Explainable Artificial Intelligence (XAI) package for Multimodal Early Autism Screening (Phase 3).
Provides:
- SHAP (KernelExplainer) and LIME (TabularExplainer) multimodal attribution frameworks.
- Pandemic Context Module disentangling environmental delay from ASD via clinical feature mappings.
"""

from .shap_explainer import MultimodalSHAPExplainer
from .lime_explainer import MultimodalLIMEExplainer
from .pandemic_context import PandemicContextAnalyzer

__all__ = [
    "MultimodalSHAPExplainer",
    "MultimodalLIMEExplainer",
    "PandemicContextAnalyzer",
]
