"""
src/models/audio_model.py

Dedicated 2D Convolutional Deep Learning Module for processing acoustic MFCC representations
in the Multimodal Early Autism Screening system (Phase 2).

Architecture:
1. Reshape: [B, 40, 313] -> [B, 1, 40, 313]
2. 3-Stage 2D CNN:
   - Block 1: Conv2d(1 -> 32) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
   - Block 2: Conv2d(32 -> 64) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
   - Block 3: Conv2d(64 -> 128) -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d(1, 1)
3. Flatten & Projection: Linear(128, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
4. Optional Classification Head: Linear(128, 1) controlled by return_features flag
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AcousticCNNModel(nn.Module):
    """
    Acoustic 2D CNN Sub-Network for processing standardized MFCC spectrograms.

    Can operate in:
    1. Feature Extractor Mode (return_features=True): Returns [batch_size, 128] dense embedding
       for multimodal late fusion integration.
    2. Classification Mode (return_features=False): Returns [batch_size, 1] raw logit for
       unimodal standalone training/evaluation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # ----------------------------------------------------------------------
        # Stage 1: Initial feature extraction & temporal-spectral pooling
        # Input: [B, 1, 40, 313] -> MaxPool(2, 2) -> Output: [B, 32, 20, 156]
        # ----------------------------------------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # ----------------------------------------------------------------------
        # Stage 2: Intermediate acoustic pattern & prosody modeling
        # Input: [B, 32, 20, 156] -> MaxPool(2, 2) -> Output: [B, 64, 10, 78]
        # ----------------------------------------------------------------------
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # ----------------------------------------------------------------------
        # Stage 3: High-level vocalization representation & global collapse
        # Input: [B, 64, 10, 78] -> AdaptiveAvgPool(1, 1) -> Output: [B, 128, 1, 1]
        # ----------------------------------------------------------------------
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # ----------------------------------------------------------------------
        # Stage 4: Projection to standardized 128-dimensional embedding space
        # ----------------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ----------------------------------------------------------------------
        # Stage 5: Standalone binary classification head for unimodal baseline
        # ----------------------------------------------------------------------
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for AcousticCNNModel.

        Args:
            x: Input tensor of shape [batch_size, 40, 313] or [batch_size, 1, 40, 313].
            return_features: If True, returns [batch_size, 128] embedding for multimodal fusion.
                             If False, returns [batch_size, 1] logit for standalone classification.

        Returns:
            torch.Tensor of shape [batch_size, 128] if return_features=True
            torch.Tensor of shape [batch_size, 1] if return_features=False
        """
        # Step 1: Ensure 4D tensor [B, 1, 40, 313]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(
                f"Expected 3D or 4D input tensor, received tensor of shape {list(x.shape)}"
            )

        # Step 2: Pass through 3-Stage 2D Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Step 3: Flatten [B, 128, 1, 1] -> [B, 128]
        x_flat = torch.flatten(x, start_dim=1)

        # Step 4: Latent audio projection
        embedding = self.projector(x_flat)  # [B, 128]

        # Step 5: Feature extraction or classification head
        if return_features:
            return embedding
        return self.classifier(embedding)


# Convenient alias
AudioModel = AcousticCNNModel


# ==============================================================================
# Testing and Verification Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing AcousticCNNModel (Phase 2 Audio Architecture)")
    print("=" * 70)

    # 1. Instantiate Model
    model = AcousticCNNModel(
        in_channels=1,
        embedding_dim=128,
        dropout=0.3,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Info] Total Trainable Parameters: {total_params:,}")

    # 2. Synthetic Tensor Verification: [batch_size=4, 40, 313]
    batch_size = 4
    num_mfcc = 40
    time_steps = 313

    synthetic_input = torch.randn(batch_size, num_mfcc, time_steps)
    print(f"\n[Synthetic Input Tensor] Shape: {list(synthetic_input.shape)} | Dtype: {synthetic_input.dtype}")

    with torch.no_grad():
        features_synth = model(synthetic_input, return_features=True)
        logits_synth = model(synthetic_input, return_features=False)

    print("\n--- Test 1: Synthetic Feature Extractor Mode (return_features=True) ---")
    print(f"Output Feature Shape : {list(features_synth.shape)} (Expected: [{batch_size}, 128])")
    assert features_synth.shape == (batch_size, 128), f"Expected ({batch_size}, 128), got {features_synth.shape}"
    print(" Assertion Passed: Feature embedding dimensions exactly match [4, 128].")

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
        real_audio = real_batch["audio"]

        print(f"Loaded Real Audio Batch Shape : {list(real_audio.shape)} | Dtype: {real_audio.dtype}")

        with torch.no_grad():
            real_features = model(real_audio, return_features=True)
            real_logits = model(real_audio, return_features=False)

        print(f"Real Audio Feature Shape      : {list(real_features.shape)}")
        print(f"Real Audio Classification Logits: {list(real_logits.shape)}")

        assert real_features.shape == (batch_size, 128)
        assert real_logits.shape == (batch_size, 1)
        print(" Assertion Passed: Real dataset batch seamlessly processed with expected dimensions.")

    except Exception as e:
        print(f"Real dataset integration error: {e}")
        raise e

    print("\n" + "=" * 70)
    print("All AcousticCNNModel tests passed successfully!")
    print("=" * 70)
