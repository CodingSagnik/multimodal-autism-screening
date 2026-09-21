"""
src/models/fusion_model.py

Unified Multimodal Late Fusion Network for Early Autism Screening (Phase 2).
Combines:
1. Vision Sub-Network (VisionLandmarkModel): 128-dim 3D facial landmark dynamics
2. Audio Sub-Network (AcousticCNNModel): 128-dim acoustic MFCC representations
3. Text Sub-Network (ClinicalTextMLP): 64-dim M-CHAT-R clinical narrative embeddings

Fused Feature Space: 128 + 128 + 64 = 320 dimensions.
Includes a Modality Weighting Hook for Phase 3 Soft Computing (Genetic Algorithm optimization)
and sub-feature extraction for Explainable AI (SHAP & LIME).
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
    Unified Multimodal Late Fusion Network combining Vision, Audio, and Text streams.

    Features:
    - End-to-end multi-stream feature extraction and fusion.
    - Modality weighting hook: Enables dynamic scalar weighting of individual modalities,
      designed for Phase 3 Genetic Algorithm optimization.
    - Explainability hook: Returns sub-modality feature representations for Phase 3
      SHAP and LIME interpretability analysis.
    """

    def __init__(
        self,
        vision_model: Optional[VisionLandmarkModel] = None,
        audio_model: Optional[AcousticCNNModel] = None,
        text_model: Optional[ClinicalTextMLP] = None,
        vision_dim: int = 128,
        audio_dim: int = 128,
        text_dim: int = 64,
        fusion_hidden1: int = 128,
        fusion_hidden2: int = 32,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.fused_dim = vision_dim + audio_dim + text_dim  # 128 + 128 + 64 = 320

        # ----------------------------------------------------------------------
        # 1. Instantiate Unimodal Sub-Networks
        # ----------------------------------------------------------------------
        self.vision_model = vision_model if vision_model is not None else VisionLandmarkModel(embedding_dim=vision_dim)
        self.audio_model = audio_model if audio_model is not None else AcousticCNNModel(embedding_dim=audio_dim)
        self.text_model = text_model if text_model is not None else ClinicalTextMLP(embedding_dim=text_dim)

        # ----------------------------------------------------------------------
        # 2. Late Fusion Classification Head
        # Input: [B, 320] -> [B, 128] -> [B, 32] -> [B, 1]
        # ----------------------------------------------------------------------
        self.fusion_head = nn.Sequential(
            # Stage 1: Initial cross-modal interaction
            nn.Linear(self.fused_dim, fusion_hidden1),
            nn.BatchNorm1d(fusion_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            # Stage 2: Low-dimensional bottleneck
            nn.Linear(fusion_hidden1, fusion_hidden2),
            nn.BatchNorm1d(fusion_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            # Stage 3: Clinical risk prediction logit
            nn.Linear(fusion_hidden2, 1),
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        modality_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        return_sub_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for Multimodal Late Fusion.

        Args:
            video: Facial landmark tensor of shape [batch_size, 50, 92, 3] or [batch_size, 50, 276].
            audio: Acoustic MFCC tensor of shape [batch_size, 40, 313] or [batch_size, 1, 40, 313].
            text: Clinical text embedding tensor of shape [batch_size, 768].
            modality_weights: Optional 3-element tuple/list/tensor (w_v, w_a, w_t) to scale
                              modality contributions prior to concatenation.
            return_sub_features: If True, returns (logits, {"vision": v_emb, "audio": a_emb, "text": t_emb})
                                 for Phase 3 XAI (SHAP/LIME) and Genetic Algorithm weight optimization.
                                 If False, returns raw logits [batch_size, 1].

        Returns:
            logits: Tensor of shape [batch_size, 1]
            (Optional) sub_features: Dict of unimodal latent tensors
        """
        # Step 1: Extract unimodal representations
        v_emb = self.vision_model(video, return_features=True)  # [B, 128]
        a_emb = self.audio_model(audio, return_features=True)    # [B, 128]
        t_emb = self.text_model(text, return_features=True)      # [B, 64]

        # Step 2: Apply optional modality weights (Phase 3 Genetic Algorithm hook)
        if modality_weights is not None:
            if isinstance(modality_weights, (list, tuple)):
                w_v, w_a, w_t = modality_weights
            elif isinstance(modality_weights, torch.Tensor):
                w_v, w_a, w_t = modality_weights[0], modality_weights[1], modality_weights[2]
            else:
                raise TypeError(f"Unsupported modality_weights type: {type(modality_weights)}")

            v_emb = v_emb * w_v
            a_emb = a_emb * w_a
            t_emb = t_emb * w_t

        # Step 3: Concatenate multimodal vectors into unified representation
        fused = torch.cat([v_emb, a_emb, t_emb], dim=1)  # [B, 320]

        # Step 4: Late fusion classification head
        logits = self.fusion_head(fused)  # [B, 1]

        # Step 5: Return prediction and optional sub-modality features
        if return_sub_features:
            sub_features = {
                "vision": v_emb,
                "audio": a_emb,
                "text": t_emb,
                "fused": fused,
            }
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

    # 1. Instantiate Unified Model
    model = MultimodalAutismClassifier(
        vision_dim=128,
        audio_dim=128,
        text_dim=64,
        fusion_hidden1=128,
        fusion_hidden2=32,
        dropout1=0.4,
        dropout2=0.2,
    )
    model.eval()

    # 2. Print Parameter Breakdown
    def get_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    v_params = get_params(model.vision_model)
    a_params = get_params(model.audio_model)
    t_params = get_params(model.text_model)
    h_params = get_params(model.fusion_head)
    total_params = get_params(model)

    print("\n[Parameter Breakdown]")
    print(f"  • Vision Sub-Network (VisionLandmarkModel) : {v_params:>10,}")
    print(f"  • Audio Sub-Network  (AcousticCNNModel)    : {a_params:>10,}")
    print(f"  • Text Sub-Network   (ClinicalTextMLP)     : {t_params:>10,}")
    print(f"  • Late Fusion Head   (MLP Classifier)      : {h_params:>10,}")
    print(f"  {'='*48}")
    print(f"  • Total Trainable Parameters               : {total_params:>10,}")

    # 3. Synthetic Forward Pass Verification
    batch_size = 4
    video_synth = torch.randn(batch_size, 50, 92, 3)
    audio_synth = torch.randn(batch_size, 40, 313)
    text_synth = torch.randn(batch_size, 768)

    print("\n[Synthetic Input Shapes]")
    print(f"  • Video : {list(video_synth.shape)}")
    print(f"  • Audio : {list(audio_synth.shape)}")
    print(f"  • Text  : {list(text_synth.shape)}")

    with torch.no_grad():
        # Test standard forward pass
        logits = model(video_synth, audio_synth, text_synth)

        # Test forward pass with sub-features extraction (XAI / GA hook)
        logits_with_feats, sub_feats = model(
            video_synth, audio_synth, text_synth, return_sub_features=True
        )

        # Test forward pass with modality weighting
        weights = (0.5, 0.3, 0.2)
        logits_weighted = model(
            video_synth, audio_synth, text_synth, modality_weights=weights
        )

    print("\n--- Test 1: Standard Forward Pass ---")
    print(f"Logits Shape : {list(logits.shape)} (Expected: [{batch_size}, 1])")
    assert logits.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {logits.shape}"
    print(" Assertion Passed: Logits output dimensions exactly match [4, 1].")

    print("\n--- Test 2: Sub-Features Return Mode (for XAI & Genetic Algorithms) ---")
    assert "vision" in sub_feats and "audio" in sub_feats and "text" in sub_feats
    print(f"Vision Embedding Shape : {list(sub_feats['vision'].shape)} (Expected: [{batch_size}, 128])")
    print(f"Audio Embedding Shape  : {list(sub_feats['audio'].shape)} (Expected: [{batch_size}, 128])")
    print(f"Text Embedding Shape   : {list(sub_feats['text'].shape)} (Expected: [{batch_size}, 64])")
    assert sub_feats["vision"].shape == (batch_size, 128)
    assert sub_feats["audio"].shape == (batch_size, 128)
    assert sub_feats["text"].shape == (batch_size, 64)
    print(" Assertion Passed: Sub-modality feature embeddings match expected shapes.")

    print("\n--- Test 3: Modality Weighting Hook ---")
    print(f"Weighted Logits Shape  : {list(logits_weighted.shape)} (Expected: [{batch_size}, 1])")
    assert logits_weighted.shape == (batch_size, 1)
    print(" Assertion Passed: Modality weighting successfully applied.")

    # 4. Real End-to-End Integration with MultimodalAutismDataset
    print("\n--- Test 4: Real Dataset End-to-End Integration Test ---")
    try:
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

        real_video = batch["video"]
        real_audio = batch["audio"]
        real_text = batch["text"]
        real_labels = batch["label"]

        print(f"Loaded Real Batch from MultimodalAutismDataset:")
        print(f"  • Real Video Shape : {list(real_video.shape)}")
        print(f"  • Real Audio Shape : {list(real_audio.shape)}")
        print(f"  • Real Text Shape  : {list(real_text.shape)}")
        print(f"  • Real Label Shape : {list(real_labels.shape)}")

        # Test forward pass and gradient computation
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        real_logits = model(real_video, real_audio, real_text)
        loss = criterion(real_logits.squeeze(1), real_labels.float())
        loss.backward()
        optimizer.step()

        print(f"\nReal Batch Training Loss : {loss.item():.4f}")
        print(f"Real Batch Predictions   : {torch.sigmoid(real_logits).squeeze().tolist()}")
        print(" Assertion Passed: Complete forward pass, backpropagation, and optimizer step succeeded without errors.")

    except Exception as e:
        print(f"Real dataset integration error: {e}")
        raise e

    print("\n" + "=" * 70)
    print("All MultimodalAutismClassifier tests passed successfully!")
    print("=" * 70)
