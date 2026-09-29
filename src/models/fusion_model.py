"""
src/models/fusion_model.py

Unified Multimodal Late Fusion Network for Early Autism Screening (Phase 2).
Combines:
1. Vision Sub-Network (VisionLandmarkModel): 128-dim 3D facial landmark dynamics with temporal padding mask
2. Audio Sub-Network (AcousticCNNModel): 128-dim acoustic MFCC representations
3. Text Sub-Network (ClinicalTextMLP): 64-dim clinical observation embeddings (optional in dual-mode)

Fused Feature Space:
- 3-Modality (Vision + Audio + Text): 128 + 128 + 64 = 320 dimensions.
- 2-Modality (Vision + Audio): 128 + 128 = 256 dimensions.

Includes:
- Temporal attention masking hook for variable-length video landmarks.
- Modality weighting hook for Phase 3 Soft Computing (Genetic Algorithm optimization).
- Sub-feature extraction hook for Explainable AI (SHAP & LIME).
- LayerNorm architecture ensuring robustness to single-sample (batch_size=1) inference and training.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root in sys.path for direct script execution and package imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.vision_model import VisionLandmarkModel
from src.models.audio_model import AcousticCNNModel
from src.models.text_model import ClinicalTextMLP


class MultimodalAutismClassifier(nn.Module):
    """
    Unified Multimodal Late Fusion Network combining Vision, Audio, and optional Text streams.

    Features:
    - End-to-end multi-stream feature extraction and late fusion.
    - Dual-mode architecture: Supports both 3-modality (V+A+T) and 2-modality (V+A) screening.
    - Temporal attention masking: Passes sequence validity masks to VisionLandmarkModel.
    - Modality weighting hook: Enables dynamic scalar weighting of individual modalities (Phase 3 GA).
    - Explainability hook: Returns sub-modality feature representations for Phase 3 SHAP and LIME.
    - Robust LayerNorm normalization: Seamless operation on batch_size=1 without BatchNorm degradation.
    """

    def __init__(
        self,
        vision_model: Optional[VisionLandmarkModel] = None,
        audio_model: Optional[AcousticCNNModel] = None,
        text_model: Optional[ClinicalTextMLP] = None,
        use_text: bool = True,
        vision_dim: int = 128,
        audio_dim: int = 128,
        text_dim: int = 64,
        fusion_hidden1: int = 128,
        fusion_hidden2: int = 32,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.use_text = use_text
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim if use_text else 0
        self.fused_dim = vision_dim + audio_dim + (text_dim if use_text else 0)

        # ----------------------------------------------------------------------
        # 1. Instantiate Unimodal Sub-Networks
        # ----------------------------------------------------------------------
        self.vision_model = (
            vision_model if vision_model is not None else VisionLandmarkModel(embedding_dim=vision_dim)
        )
        self.audio_model = (
            audio_model if audio_model is not None else AcousticCNNModel(embedding_dim=audio_dim)
        )
        if self.use_text:
            self.text_model = (
                text_model if text_model is not None else ClinicalTextMLP(embedding_dim=text_dim)
            )
        else:
            self.text_model = None

        # ----------------------------------------------------------------------
        # 2. Late Fusion Classification Head (LayerNorm for batch_size=1 robustness)
        # Input: [B, fused_dim] -> [B, 128] -> [B, 32] -> [B, 1]
        # ----------------------------------------------------------------------
        self.fusion_head = nn.Sequential(
            # Stage 1: Initial cross-modal interaction
            nn.Linear(self.fused_dim, fusion_hidden1),
            nn.LayerNorm(fusion_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            # Stage 2: Low-dimensional bottleneck
            nn.Linear(fusion_hidden1, fusion_hidden2),
            nn.LayerNorm(fusion_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            # Stage 3: Clinical risk prediction logit
            nn.Linear(fusion_hidden2, 1),
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        modality_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        return_sub_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for Multimodal Late Fusion.

        Args:
            video: Facial landmark tensor of shape [batch_size, 50, 92, 3] or [batch_size, 50, 276].
            audio: Acoustic MFCC tensor of shape [batch_size, 40, 313] or [batch_size, 1, 40, 313].
            text: Optional clinical text embedding tensor of shape [batch_size, 768].
            video_mask: Optional boolean mask [batch_size, seq_len] for valid landmark frames.
            modality_weights: Optional weights (w_v, w_a, [w_t]) to scale modality contributions.
            return_sub_features: If True, returns (logits, sub_features_dict).

        Returns:
            logits: Tensor of shape [batch_size, 1]
            (Optional) sub_features: Dict of unimodal latent tensors
        """
        # Step 1: Extract unimodal representations
        v_emb = self.vision_model(video, return_features=True, mask=video_mask)  # [B, 128]
        a_emb = self.audio_model(audio, return_features=True)                   # [B, 128]

        has_text = self.use_text and (text is not None) and (self.text_model is not None)
        if has_text:
            t_emb = self.text_model(text, return_features=True)                  # [B, 64]
        else:
            t_emb = None

        # Step 2: Apply optional modality weights (Phase 3 Genetic Algorithm hook)
        if modality_weights is not None:
            if isinstance(modality_weights, (list, tuple)):
                weights = list(modality_weights)
            elif isinstance(modality_weights, torch.Tensor):
                weights = modality_weights.tolist()
            else:
                raise TypeError(f"Unsupported modality_weights type: {type(modality_weights)}")

            if has_text and len(weights) >= 3:
                v_emb = v_emb * weights[0]
                a_emb = a_emb * weights[1]
                t_emb = t_emb * weights[2]
            elif len(weights) >= 2:
                v_emb = v_emb * weights[0]
                a_emb = a_emb * weights[1]

        # Step 3: Concatenate multimodal vectors into unified representation
        if has_text:
            fused = torch.cat([v_emb, a_emb, t_emb], dim=1)  # [B, 320]
        else:
            fused = torch.cat([v_emb, a_emb], dim=1)         # [B, 256]

        # Step 4: Late fusion classification head
        logits = self.fusion_head(fused)  # [B, 1]

        # Step 5: Return prediction and optional sub-modality features
        if return_sub_features:
            sub_features = {
                "vision": v_emb,
                "audio": a_emb,
                "fused": fused,
            }
            if has_text:
                sub_features["text"] = t_emb
            return logits, sub_features

        return logits


# Convenient alias
MultimodalFusionModel = MultimodalAutismClassifier


# ==============================================================================
# Testing and Verification Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing MultimodalAutismClassifier (Phase 2 Late Fusion Architecture)")
    print("=" * 70)

    # 1. Instantiate 3-Modality Unified Model
    model_3m = MultimodalAutismClassifier(
        use_text=True,
        vision_dim=128,
        audio_dim=128,
        text_dim=64,
        fusion_hidden1=128,
        fusion_hidden2=32,
    )
    model_3m.eval()

    def get_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("\n[Parameter Breakdown - 3-Modality (V+A+T)]")
    print(f"  • Vision Sub-Network (VisionLandmarkModel) : {get_params(model_3m.vision_model):>10,}")
    print(f"  • Audio Sub-Network  (AcousticCNNModel)    : {get_params(model_3m.audio_model):>10,}")
    print(f"  • Text Sub-Network   (ClinicalTextMLP)     : {get_params(model_3m.text_model):>10,}")
    print(f"  • Late Fusion Head   (LayerNorm Classifier): {get_params(model_3m.fusion_head):>10,}")
    print(f"  {'='*48}")
    print(f"  • Total Trainable Parameters               : {get_params(model_3m):>10,}")

    # 2. Synthetic Forward Pass Verification (3-Modality)
    batch_size = 4
    video_synth = torch.randn(batch_size, 50, 92, 3)
    video_mask_synth = torch.ones(batch_size, 50, dtype=torch.bool)
    video_mask_synth[:, 35:] = False  # Simulate padded frames
    audio_synth = torch.randn(batch_size, 40, 313)
    text_synth = torch.randn(batch_size, 768)

    with torch.no_grad():
        logits_3m, sub_3m = model_3m(
            video_synth, audio_synth, text_synth, video_mask=video_mask_synth, return_sub_features=True
        )
        logits_weighted = model_3m(
            video_synth, audio_synth, text_synth, video_mask=video_mask_synth, modality_weights=(0.5, 0.3, 0.2)
        )

    assert logits_3m.shape == (batch_size, 1)
    assert sub_3m["vision"].shape == (batch_size, 128)
    assert sub_3m["audio"].shape == (batch_size, 128)
    assert sub_3m["text"].shape == (batch_size, 64)
    assert sub_3m["fused"].shape == (batch_size, 320)
    print("\n--- Test 1: 3-Modality Forward Pass & Masking --- [PASSED]")

    # 3. Test 2-Modality Architecture (Video + Audio)
    model_2m = MultimodalAutismClassifier(use_text=False)
    model_2m.eval()
    with torch.no_grad():
        logits_2m, sub_2m = model_2m(
            video_synth, audio_synth, video_mask=video_mask_synth, return_sub_features=True
        )
    assert logits_2m.shape == (batch_size, 1)
    assert sub_2m["fused"].shape == (batch_size, 256)
    print("--- Test 2: 2-Modality Forward Pass (V+A) --- [PASSED]")

    # 4. Test Single-Sample (batch_size=1) in Training Mode (LayerNorm Check)
    model_3m.train()
    v_single = torch.randn(1, 50, 92, 3)
    v_mask_single = torch.ones(1, 50, dtype=torch.bool)
    a_single = torch.randn(1, 40, 313)
    t_single = torch.randn(1, 768)
    out_single = model_3m(v_single, a_single, t_single, video_mask=v_mask_single)
    assert out_single.shape == (1, 1)
    print("--- Test 3: Batch Size = 1 Training Mode (LayerNorm Verification) --- [PASSED]")

    # 5. Real End-to-End Dataset Integration Test
    print("\n--- Test 4: Real Dataset End-to-End Integration Test ---")
    from src.data_processing.multimodal_dataset import MultimodalAutismDataset

    dataset = MultimodalAutismDataset(
        video_dir=Path("data/processed/video_landmarks"),
        audio_dir=Path("data/processed/audio_features"),
        text_dir=Path("data/processed/text_embeddings"),
        labels_file=Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
        max_video_frames=50,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(dataloader))

    optimizer = torch.optim.AdamW(model_3m.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    logits_real = model_3m(batch["video"], batch["audio"], batch["text"], video_mask=batch["video_mask"])
    loss = criterion(logits_real.squeeze(1), batch["label"].float())
    loss.backward()
    optimizer.step()

    print(f"  • Real Batch Training Loss: {loss.item():.4f}")
    print("--- Test 4: Real Dataset Gradient Update --- [PASSED]")

    print("\n" + "=" * 70)
    print("All MultimodalAutismClassifier tests passed successfully!")
    print("=" * 70)
