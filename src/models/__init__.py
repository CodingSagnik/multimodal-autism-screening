"""
src/models/__init__.py

Models package containing unimodal encoders and multimodal fusion architectures
for Early Autism Screening (Phase 2).
"""

from .vision_model import VisionLandmarkModel, VisionModel, TemporalAttentionPooling
from .audio_model import AcousticCNNModel, AudioModel
from .text_model import ClinicalTextMLP, TextModel
from .fusion_model import MultimodalAutismClassifier, MultimodalFusionModel

__all__ = [
    "VisionLandmarkModel",
    "VisionModel",
    "TemporalAttentionPooling",
    "AcousticCNNModel",
    "AudioModel",
    "ClinicalTextMLP",
    "TextModel",
    "MultimodalAutismClassifier",
    "MultimodalFusionModel",
]
