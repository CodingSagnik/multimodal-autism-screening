"""
src/models/text_model.py

Dedicated Multi-Layer Perceptron (MLP) Deep Learning Module for processing clinical
narrative embeddings derived from M-CHAT-R questionnaires in the Multimodal Early
Autism Screening system (Phase 2).

Architecture:
1. Block 1: Linear(768 -> 256) -> LayerNorm(256) -> ReLU() -> Dropout(0.4)
2. Block 2: Linear(256 -> 128) -> LayerNorm(128) -> ReLU() -> Dropout(0.3)
3. Block 3: Linear(128 -> 64) -> LayerNorm(64) -> ReLU()
4. Optional Classification Head: Linear(64 -> 1) controlled by return_features flag
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ClinicalTextMLP(nn.Module):
    """
    Clinical Text MLP Sub-Network for regularizing high-dimensional DistilBERT
    narrative representations and projecting them into a compact latent space.

    Can operate in:
    1. Feature Extractor Mode (return_features=True): Returns [batch_size, 64] dense embedding
       for multimodal late fusion integration.
    2. Classification Mode (return_features=False): Returns [batch_size, 1] raw logit for
       unimodal standalone training/evaluation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        embedding_dim: int = 64,
        dropout1: float = 0.4,
        dropout2: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # ----------------------------------------------------------------------
        # Stage 1: Dimensionality reduction & initial regularization
        # Input: [B, 768] -> Output: [B, 256]
        # ----------------------------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
        )

        # ----------------------------------------------------------------------
        # Stage 2: Intermediate non-linear compression
        # Input: [B, 256] -> Output: [B, 128]
        # ----------------------------------------------------------------------
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
        )

        # ----------------------------------------------------------------------
        # Stage 3: Latent text embedding space
        # Input: [B, 128] -> Output: [B, 64]
        # ----------------------------------------------------------------------
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------------
        # Stage 4: Standalone binary classification head
        # Input: [B, 64] -> Output: [B, 1]
        # ----------------------------------------------------------------------
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for ClinicalTextMLP.

        Args:
            x: Input tensor of shape [batch_size, 768] (or [batch_size, 1, 768]).
            return_features: If True, returns [batch_size, 64] embedding for multimodal fusion.
                             If False, returns [batch_size, 1] logit for standalone classification.

        Returns:
            torch.Tensor of shape [batch_size, 64] if return_features=True
            torch.Tensor of shape [batch_size, 1] if return_features=False
        """
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor [batch_size, {self.input_dim}], received tensor of shape {list(x.shape)}"
            )

        # Pass through regularized projection backbone
        h1 = self.block1(x)
        h2 = self.block2(h1)
        embedding = self.block3(h2)  # [B, 64]

        if return_features:
            return embedding
        return self.classifier(embedding)


# Convenient alias
TextModel = ClinicalTextMLP


# ==============================================================================
# Testing and Verification Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing ClinicalTextMLP (Phase 2 Text Architecture)")
    print("=" * 70)

    # 1. Instantiate Model
    model = ClinicalTextMLP(
        input_dim=768,
        hidden_dim1=256,
        hidden_dim2=128,
        embedding_dim=64,
        dropout1=0.4,
        dropout2=0.3,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Info] Total Trainable Parameters: {total_params:,}")

    # 2. Synthetic Tensor Verification: [batch_size=4, 768]
    batch_size = 4
    input_dim = 768

    synthetic_input = torch.randn(batch_size, input_dim)
    print(f"\n[Synthetic Input Tensor] Shape: {list(synthetic_input.shape)} | Dtype: {synthetic_input.dtype}")

    with torch.no_grad():
        features_synth = model(synthetic_input, return_features=True)
        logits_synth = model(synthetic_input, return_features=False)

    print("\n--- Test 1: Synthetic Feature Extractor Mode (return_features=True) ---")
    print(f"Output Feature Shape : {list(features_synth.shape)} (Expected: [{batch_size}, 64])")
    assert features_synth.shape == (batch_size, 64), f"Expected ({batch_size}, 64), got {features_synth.shape}"
    print(" Assertion Passed: Feature embedding dimensions exactly match [4, 64].")

    print("\n--- Test 2: Synthetic Classification Mode (return_features=False) ---")
    print(f"Output Logit Shape   : {list(logits_synth.shape)} (Expected: [{batch_size}, 1])")
    assert logits_synth.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {logits_synth.shape}"
    print(" Assertion Passed: Classifier output dimensions exactly match [4, 1].")

    # 3. Real Integration Test via MultimodalAutismDataset
    print("\n--- Test 3: Real Dataset Integration Test ---")
    try:
        import sys
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.data_processing.multimodal_dataset import MultimodalAutismDataset

        dataset = MultimodalAutismDataset(
            video_dir=Path("data/processed/video_landmarks"),
            audio_dir=Path("data/processed/audio_features"),
            text_dir=Path("data/processed/text_embeddings"),
            labels_file=Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
            max_video_frames=50,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        real_batch = next(iter(dataloader))
        real_text = real_batch["text"]

        print(f"Loaded Real Text Batch Shape  : {list(real_text.shape)} | Dtype: {real_text.dtype}")

        with torch.no_grad():
            real_features = model(real_text, return_features=True)
            real_logits = model(real_text, return_features=False)

        print(f"Real Text Feature Shape       : {list(real_features.shape)}")
        print(f"Real Text Classification Logits: {list(real_logits.shape)}")

        assert real_features.shape == (batch_size, 64)
        assert real_logits.shape == (batch_size, 1)
        print(" Assertion Passed: Real dataset batch seamlessly processed with expected dimensions.")

    except Exception as e:
        print(f"Real dataset integration error: {e}")
        raise e

    print("\n" + "=" * 70)
    print("All ClinicalTextMLP tests passed successfully!")
    print("=" * 70)
