"""
src/models/vision_model.py

Dedicated Spatio-Temporal Deep Learning Module for processing facial landmark sequences
in the Multimodal Early Autism Screening system (Phase 2).

Architecture:
1. Spatial Flattening: [B, 50, 92, 3] -> [B, 50, 276]
2. 1D Temporal Convolution: Conv1d -> BatchNorm1d -> ReLU -> Dropout(0.3)
3. BiLSTM Temporal Modeling: 2-layer Bidirectional LSTM (hidden_size=64) -> [B, 50, 128]
4. Temporal Attention Pooling: Collapses temporal dimension -> [B, 128]
5. Linear Projection: Linear(128, 128)
6. Optional Classification Head: Linear(128, 1) controlled by return_features flag
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    """
    Computes an attention-weighted sum of time-step representations.
    Learns to prioritize micro-behaviors (e.g. eye contact breaks, gaze shifts).
    """

    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, in_features]
            mask: Optional boolean mask of shape [batch_size, seq_len] (True for valid frames)
        Returns:
            pooled: Tensor of shape [batch_size, in_features]
            weights: Tensor of shape [batch_size, seq_len, 1]
        """
        # Calculate attention energy scores: [B, T, 1]
        scores = self.attention_net(x)

        if mask is not None:
            # Mask out invalid/padded frames
            mask_expanded = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask_expanded, -1e9)

        # Softmax over temporal dimension
        weights = F.softmax(scores, dim=1)  # [B, T, 1]

        # Weighted sum: sum([B, T, D] * [B, T, 1], dim=1) -> [B, D]
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights


class TemporalConvBlock(nn.Module):
    """
    1D Temporal Convolutional Block along the sequence dimension
    to capture local frame-to-frame transitional dynamics.
    """

    def __init__(
        self,
        in_channels: int = 276,
        out_channels: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # preserves temporal sequence length
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, in_channels]
        Returns:
            Tensor of shape [batch_size, seq_len, out_channels]
        """
        # Conv1d expects [B, C, L], so transpose [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Transpose back: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        return x


class VisionLandmarkModel(nn.Module):
    """
    Vision Sub-Network for processing temporal 3D facial landmark sequences.
    
    Can operate in:
    1. Feature Extractor Mode (return_features=True): Returns [batch_size, 128] dense embedding
       for multimodal late fusion integration.
    2. Classification Mode (return_features=False): Returns [batch_size, 1] raw logit for
       unimodal standalone training/evaluation.
    """

    def __init__(
        self,
        num_landmarks: int = 92,
        spatial_coords: int = 3,
        conv_channels: int = 128,
        kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        pooling_type: str = "attention",
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.spatial_coords = spatial_coords
        self.raw_feature_dim = num_landmarks * spatial_coords  # 92 * 3 = 276
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type.lower()

        # 1. Temporal 1D Convolution Block (captures local landmark transitions)
        self.temporal_conv = TemporalConvBlock(
            in_channels=self.raw_feature_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. Bidirectional LSTM (captures long-range behavioral trajectories)
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        bilstm_out_dim = lstm_hidden_size * 2  # 64 * 2 = 128

        # 3. Temporal Pooling Layer
        if self.pooling_type == "attention":
            self.temporal_pooling = TemporalAttentionPooling(
                in_features=bilstm_out_dim, hidden_dim=64
            )
        elif self.pooling_type == "mean":
            self.temporal_pooling = None
        else:
            raise ValueError(f"Unknown pooling_type: '{pooling_type}'. Supported: 'attention', 'mean'.")

        # 4. Projection Layer (maps pooled representation to final embedding dimension)
        self.projector = nn.Sequential(
            nn.Linear(bilstm_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # 5. Standalone Binary Classification Head (predicts ASD behavioral risk)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention_weights: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Vision Landmark Model.

        Args:
            x: Input tensor of shape [batch_size, 50, 92, 3] or [batch_size, 50, 276].
            return_features: If True, returns [batch_size, 128] embedding for multimodal fusion.
                             If False, returns [batch_size, 1] logit for standalone classification.
            return_attention_weights: If True (and using attention pooling), also returns
                                      temporal attention weights [batch_size, seq_len, 1].
            mask: Optional boolean mask [batch_size, seq_len] for valid time-steps.

        Returns:
            torch.Tensor of shape [batch_size, 128] if return_features=True
            torch.Tensor of shape [batch_size, 1] if return_features=False
            (Optionally tuple with attention weights if return_attention_weights=True)
        """
        # Step 1: Reshape spatial coordinates to flattened feature vector
        if x.ndim == 4:
            # [B, T, N, C] -> [B, T, N * C]
            batch_size, seq_len, num_landmarks, coords = x.shape
            x_flat = x.view(batch_size, seq_len, num_landmarks * coords)
        elif x.ndim == 3:
            # Already flattened: [B, T, D]
            x_flat = x
        else:
            raise ValueError(
                f"Expected 3D or 4D input tensor, received tensor of shape {list(x.shape)}"
            )

        # Step 2: 1D Temporal Convolution
        conv_feats = self.temporal_conv(x_flat)  # [B, T, conv_channels]

        # Step 3: 2-layer Bidirectional LSTM
        lstm_out, _ = self.bilstm(conv_feats)  # [B, T, 128]

        # Step 4: Temporal Pooling
        attention_weights = None
        if self.pooling_type == "attention":
            pooled, attention_weights = self.temporal_pooling(lstm_out, mask=mask)  # [B, 128]
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (lstm_out * mask_expanded).sum(dim=1) / torch.clamp(
                    mask_expanded.sum(dim=1), min=1e-8
                )
            else:
                pooled = lstm_out.mean(dim=1)  # [B, 128]

        # Step 5: Linear Projection to 128-dimensional embedding
        embedding = self.projector(pooled)  # [B, 128]

        # Step 6: Return features or classification logits
        if return_features:
            out = embedding
        else:
            out = self.classifier(embedding)  # [B, 1]

        if return_attention_weights and attention_weights is not None:
            return out, attention_weights
        return out


# Alias for flexible importing
VisionModel = VisionLandmarkModel


# ==============================================================================
# Testing and Verification Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing VisionLandmarkModel (Phase 2 Vision Architecture)")
    print("=" * 70)

    # Instantiate model
    model = VisionLandmarkModel(
        num_landmarks=92,
        spatial_coords=3,
        conv_channels=128,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        embedding_dim=128,
        dropout=0.3,
        pooling_type="attention",
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Info] Total Trainable Parameters: {total_params:,}")

    # Generate dummy input tensor matching specification: [batch_size=4, 50, 92, 3]
    batch_size = 4
    seq_len = 50
    num_landmarks = 92
    spatial_coords = 3

    dummy_input = torch.randn(batch_size, seq_len, num_landmarks, spatial_coords)
    print(f"\n[Input Tensor] Shape: {list(dummy_input.shape)} | Dtype: {dummy_input.dtype}")

    # 1. Test Feature Extraction Mode (for Multimodal Fusion)
    with torch.no_grad():
        features, attn_weights = model(
            dummy_input, return_features=True, return_attention_weights=True
        )

    print("\n--- Test 1: Multimodal Feature Extractor Mode (return_features=True) ---")
    print(f"Output Feature Tensor Shape : {list(features.shape)} (Expected: [{batch_size}, 128])")
    print(f"Attention Weights Shape     : {list(attn_weights.shape)} (Expected: [{batch_size}, {seq_len}, 1])")
    assert features.shape == (
        batch_size,
        128,
    ), f"Expected shape ({batch_size}, 128), got {features.shape}"
    assert attn_weights.shape == (
        batch_size,
        seq_len,
        1,
    ), f"Expected shape ({batch_size}, {seq_len}, 1), got {attn_weights.shape}"
    print(" Assertion Passed: Feature embedding dimensions exactly match [4, 128].")

    # 2. Test Standalone Classification Mode
    with torch.no_grad():
        logits = model(dummy_input, return_features=False)

    print("\n--- Test 2: Standalone Classification Mode (return_features=False) ---")
    print(f"Output Logits Tensor Shape  : {list(logits.shape)} (Expected: [{batch_size}, 1])")
    assert logits.shape == (
        batch_size,
        1,
    ), f"Expected shape ({batch_size}, 1), got {logits.shape}"
    print(" Assertion Passed: Classifier output dimensions exactly match [4, 1].")

    # 3. Test with Pre-flattened Input [4, 50, 276]
    dummy_flat = dummy_input.view(batch_size, seq_len, num_landmarks * spatial_coords)
    with torch.no_grad():
        features_flat = model(dummy_flat, return_features=True)
    assert features_flat.shape == (batch_size, 128)
    print("\n--- Test 3: Pre-flattened Input Handling [4, 50, 276] ---")
    print(f"Pre-flattened Output Shape  : {list(features_flat.shape)}")
    print(" Assertion Passed: Successfully handled pre-flattened 3D input tensor.")

    print("\n" + "=" * 70)
    print("All VisionLandmarkModel tests passed successfully!")
    print("=" * 70)
