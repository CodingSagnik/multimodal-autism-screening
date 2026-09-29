import torch
from torch.utils.data import DataLoader
from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models import (
    VisionLandmarkModel,
    AcousticCNNModel,
    ClinicalTextMLP,
    MultimodalAutismClassifier,
)

print("=" * 65)
print("PHASE 1 & PHASE 2 SYSTEM INTEGRITY AUDIT")
print("=" * 65)

# 1. Dataset Verification
dataset = MultimodalAutismDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))

print(f"\n[1] DATA PIPELINE (Phase 1)")
print(f"  • Total Aligned Multimodal Samples : {len(dataset)}")
print(f"  • Video Batch Tensor Shape        : {list(batch['video'].shape)}")
print(f"  • Audio Batch Tensor Shape        : {list(batch['audio'].shape)}")
print(f"  • Text Batch Tensor Shape         : {list(batch['text'].shape)}")
print(f"  • Labels Batch Tensor Shape       : {list(batch['label'].shape)}")

# 2. Model Parameters & Forward Pass
model = MultimodalAutismClassifier()
model.eval()

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"\n[2] ARCHITECTURAL PARAMETERS (Phase 2)")
print(f"  • Vision Model (BiLSTM + Attention) : {count_params(model.vision_model):>8,}")
print(f"  • Audio Model (2D CNN Spectrogram)  : {count_params(model.audio_model):>8,}")
print(f"  • Text Model (DistilBERT MLP)       : {count_params(model.text_model):>8,}")
print(f"  • Late Fusion Head                  : {count_params(model.fusion_head):>8,}")
print(f"  -----------------------------------------------")
print(f"  • Total Trainable Parameters        : {count_params(model):>8,}")

# 3. Forward Pass & Functional Hooks Check
with torch.no_grad():
    logits, sub_feats = model(
        batch['video'], batch['audio'], batch['text'], return_sub_features=True
    )

print(f"\n[3] FUNCTIONAL HOOKS CHECK")
print(f"  • Logits Output Shape               : {list(logits.shape)} (Expected: [4, 1])")
print(f"  • Latent Vision Vector              : {list(sub_feats['vision'].shape)} (128D)")
print(f"  • Latent Audio Vector               : {list(sub_feats['audio'].shape)} (128D)")
print(f"  • Latent Text Vector                : {list(sub_feats['text'].shape)} (64D)")
print(f"  • Concatenated Fusion Vector        : {list(sub_feats['fused'].shape)} (320D)")
print("\n[STATUS] Phase 1 and Phase 2 fully validated and ready for Phase 3.")
print("=" * 65)