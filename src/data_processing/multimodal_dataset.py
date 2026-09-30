"""
multimodal_dataset.py

Custom PyTorch Dataset class (MultimodalAutismDataset) to align and load synchronized
multimodal behavioral streams:
1. 3D Facial Landmarks (Video) with explicit temporal boolean padding masks
2. Standardized MFCCs (Audio)
3. Dense Clinical Text Embeddings (Text, optional for dual-mode operation)

Supports dual-mode execution:
- 3-Modality (Video + Audio + Text)
- 2-Modality (Video + Audio)
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MultimodalAutismDataset(Dataset):
    """
    Custom PyTorch Dataset for Multimodal Early Autism Screening.
    Aligns video landmark sequences, audio MFCC features, and clinical text embeddings.
    Strictly drops unaligned or missing records to guarantee 1-to-1 sample correspondence.
    Generates video padding masks for valid temporal attention pooling.
    """

    def __init__(
        self,
        video_dir: Union[str, Path] = Path("data/processed/video_landmarks"),
        audio_dir: Union[str, Path] = Path("data/processed/audio_features"),
        text_dir: Optional[Union[str, Path]] = Path("data/processed/text_embeddings"),
        labels_file: Optional[Union[str, Path]] = Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
        modalities: Sequence[str] = ("video", "audio", "text"),
        max_video_frames: Optional[int] = 50,
        flatten_landmarks: bool = False,
        transform_video: Optional[Callable] = None,
        transform_audio: Optional[Callable] = None,
        transform_text: Optional[Callable] = None,
    ):
        """
        Args:
            video_dir: Directory containing preprocessed video landmark files (.npy or .pt).
            audio_dir: Directory containing preprocessed audio MFCC files (.npy or .pt).
            text_dir: Directory containing preprocessed text embeddings (.npy, .pt, or dictionary).
            labels_file: Optional path to CSV containing ground-truth diagnostic labels.
            modalities: Modalities to include. Defaults to ("video", "audio", "text").
                        Can also be ("video", "audio") for 2-modality screening.
            max_video_frames: Optional fixed temporal length for video tensors (pads/truncates).
            flatten_landmarks: If True, flattens spatial landmark dims (T, N*3).
            transform_video: Optional transform/augmentation callable for video tensor.
            transform_audio: Optional transform/augmentation callable for audio tensor.
            transform_text: Optional transform/augmentation callable for text tensor.
        """
        self.modalities = tuple(m.lower().strip() for m in modalities)
        valid_mods = {"video", "audio", "text"}
        for m in self.modalities:
            if m not in valid_mods:
                raise ValueError(f"Unknown modality: '{m}'. Supported: {valid_mods}")

        self.video_dir = Path(video_dir).resolve() if "video" in self.modalities else None
        self.audio_dir = Path(audio_dir).resolve() if "audio" in self.modalities else None
        self.text_dir = Path(text_dir).resolve() if ("text" in self.modalities and text_dir) else None
        self.labels_file = Path(labels_file).resolve() if labels_file else None
        self.max_video_frames = max_video_frames
        self.flatten_landmarks = flatten_landmarks
        self.transform_video = transform_video
        self.transform_audio = transform_audio
        self.transform_text = transform_text

        # Validate directory existence for active modalities
        for d_name, d_path in [
            ("Video", self.video_dir),
            ("Audio", self.audio_dir),
            ("Text", self.text_dir),
        ]:
            if d_path is not None and not d_path.exists():
                raise FileNotFoundError(f"{d_name} directory not found: {d_path}")

        # Index available files in active modalities
        self.video_files: Dict[str, Path] = {}
        self.audio_files: Dict[str, Path] = {}
        self.text_files: Dict[str, Path] = {}
        self.text_dict_bank: Dict[str, torch.Tensor] = {}

        if "video" in self.modalities and self.video_dir:
            self.video_files = self._scan_modality_files(self.video_dir)
        if "audio" in self.modalities and self.audio_dir:
            self.audio_files = self._scan_modality_files(self.audio_dir)
        if "text" in self.modalities and self.text_dir:
            self.text_files, self.text_dict_bank = self._scan_text_modality(self.text_dir)

        # Load labels mapping
        self.labels_map = self._load_labels_map(self.labels_file)

        # Compute strict intersection across active modalities
        self.active_ids = self._align_modalities()

        logger.info(
            f"Dataset alignment complete: {len(self.active_ids)} samples aligned across modalities {self.modalities}."
        )

    def _scan_modality_files(self, directory: Path) -> Dict[str, Path]:
        """Maps sample ID (file stem) to absolute file path (.npy or .pt)."""
        file_map: Dict[str, Path] = {}
        for f in directory.iterdir():
            if f.suffix.lower() in [".npy", ".pt"]:
                file_map[f.stem] = f
        return file_map

    def _scan_text_modality(
        self, directory: Path
    ) -> Tuple[Dict[str, Path], Dict[str, torch.Tensor]]:
        """
        Discovers text embeddings from individual files or monolithic serialized tensor bank.
        Excludes legacy mchat_embedded.pt to prevent unaligned cross-dataset pollution.
        """
        text_files: Dict[str, Path] = {}
        text_dict_bank: Dict[str, torch.Tensor] = {}

        # 1. Check for individual .npy / .pt files
        for f in directory.iterdir():
            if f.name in ["video_text_embeddings.pt", "video_text_narratives.csv", "mchat_embedded.pt"]:
                continue
            if f.suffix.lower() in [".npy", ".pt"]:
                text_files[f.stem] = f

        # 2. Check for monolithic dictionary banks (.pt)
        bank_path = directory / "video_text_embeddings.pt"
        if bank_path.exists():
            try:
                loaded = torch.load(bank_path, weights_only=False)
                if isinstance(loaded, dict) and "patient_ids" in loaded and "embeddings" in loaded:
                    p_ids = loaded["patient_ids"]
                    embs = loaded["embeddings"]
                    if isinstance(embs, np.ndarray):
                        embs = torch.from_numpy(embs)
                    for i, pid in enumerate(p_ids):
                        text_dict_bank[str(pid)] = embs[i].float()
            except Exception as e:
                logger.warning(f"Could not load dictionary bank from {bank_path}: {e}")

        return text_files, text_dict_bank

    def _load_labels_map(self, labels_path: Optional[Path]) -> Dict[str, int]:
        """Loads binary clinical labels from metadata CSV."""
        labels_map: Dict[str, int] = {}
        if labels_path and labels_path.exists():
            try:
                df = pd.read_csv(labels_path)
                df.columns = [c.strip() for c in df.columns]

                # Case 1: AV-ASD dataset.csv format (Video_ID, Background, symptom columns)
                if "Video_ID" in df.columns:
                    symptom_cols = [c for c in df.columns if c not in ["Video_ID", "Background"]]
                    for _, row in df.iterrows():
                        vid = str(row["Video_ID"]).strip()
                        is_background = row.get("Background", 0) == 1
                        has_symptoms = any(row.get(col, 0) == 1 for col in symptom_cols)
                        # 1 = ASD behavioral risk, 0 = Typical / Background control
                        label = 0 if is_background or not has_symptoms else 1
                        labels_map[vid] = label

                # Case 2: M-CHAT format with Class column (for standalone M-CHAT evaluation)
                elif "Class" in df.columns:
                    id_col = "Patient_ID" if "Patient_ID" in df.columns else None
                    for idx, row in df.iterrows():
                        pid = str(row[id_col]).strip() if id_col else f"PATIENT_{idx:05d}"
                        cls_val = str(row["Class"]).strip().upper()
                        label = 1 if cls_val in ["YES", "1", "TRUE", "POSITIVE", "Y"] else 0
                        labels_map[pid] = label

            except Exception as e:
                logger.warning(f"Error parsing labels file {labels_path}: {e}")

        return labels_map

    def _align_modalities(self) -> List[str]:
        """
        Finds exact intersection of IDs present in active modalities.
        Drops any sample missing one or more of the specified modalities.
        """
        id_sets: List[Set[str]] = []
        counts: Dict[str, int] = {}

        if "video" in self.modalities:
            v_set = set(self.video_files.keys())
            id_sets.append(v_set)
            counts["Video"] = len(v_set)

        if "audio" in self.modalities:
            a_set = set(self.audio_files.keys())
            id_sets.append(a_set)
            counts["Audio"] = len(a_set)

        if "text" in self.modalities:
            t_set = set(self.text_files.keys()).union(set(self.text_dict_bank.keys()))
            id_sets.append(t_set)
            counts["Text"] = len(t_set)

        if not id_sets:
            raise ValueError("No active modalities configured.")

        # Strict intersection across active modalities
        common_ids = sorted(list(set.intersection(*id_sets)))
        total_union = set.union(*id_sets)
        dropped_count = len(total_union) - len(common_ids)

        if dropped_count > 0:
            count_str = ", ".join(f"{k}: {v}" for k, v in counts.items())
            logger.info(
                f"Unaligned Sample Filter: {dropped_count} incomplete samples dropped. "
                f"({count_str} -> Aligned: {len(common_ids)})"
            )

        if not common_ids:
            raise ValueError(
                f"No overlapping samples found across configured modalities: {self.modalities}."
            )

        return common_ids

    def _load_tensor(self, file_path: Path) -> torch.Tensor:
        """Loads .npy or .pt file into a float32 PyTorch tensor."""
        if file_path.suffix.lower() == ".npy":
            arr = np.load(file_path)
            return torch.from_numpy(arr).float()
        elif file_path.suffix.lower() == ".pt":
            tensor = torch.load(file_path, weights_only=False)
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            return tensor.float()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def __len__(self) -> int:
        return len(self.active_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves aligned multimodal sample at index idx.

        Returns:
            dict containing:
                "video": torch.FloatTensor [T, num_landmarks, 3] or [T, num_landmarks*3]
                "video_mask": torch.BoolTensor [T] (True for valid frames, False for padding)
                "audio": torch.FloatTensor [n_mfcc, time_frames]
                "text": torch.FloatTensor [embedding_dim] (if 'text' in modalities)
                "label": torch.LongTensor scalar
                "sample_id": str
        """
        sample_id = self.active_ids[idx]
        sample_dict: Dict[str, Union[torch.Tensor, str]] = {"sample_id": sample_id}

        # 1. Load Video Landmarks & Generate Temporal Attention Padding Mask
        if "video" in self.modalities:
            video_path = self.video_files[sample_id]
            video_tensor = self._load_tensor(video_path)  # Shape: [T, N, 3]
            t_len = video_tensor.shape[0]

            if self.max_video_frames is not None:
                if t_len >= self.max_video_frames:
                    video_tensor = video_tensor[: self.max_video_frames]
                    valid_len = self.max_video_frames
                else:
                    valid_len = t_len
                    pad_shape = (self.max_video_frames - t_len,) + video_tensor.shape[1:]
                    padding = torch.zeros(pad_shape, dtype=video_tensor.dtype)
                    video_tensor = torch.cat([video_tensor, padding], dim=0)

                video_mask = torch.zeros(self.max_video_frames, dtype=torch.bool)
                video_mask[:valid_len] = True
            else:
                video_mask = torch.ones(t_len, dtype=torch.bool)

            # Optional landmark spatial flattening: [T, N, 3] -> [T, N*3]
            if self.flatten_landmarks and video_tensor.ndim == 3:
                video_tensor = video_tensor.view(video_tensor.shape[0], -1)

            if self.transform_video is not None:
                video_tensor = self.transform_video(video_tensor)

            sample_dict["video"] = video_tensor.float()
            sample_dict["video_mask"] = video_mask.bool()

        # 2. Load Audio MFCCs
        if "audio" in self.modalities:
            audio_path = self.audio_files[sample_id]
            audio_tensor = self._load_tensor(audio_path)  # Shape: [n_mfcc, time_frames]
            if self.transform_audio is not None:
                audio_tensor = self.transform_audio(audio_tensor)
            sample_dict["audio"] = audio_tensor.float()

        # 3. Load Text Embeddings (if included)
        if "text" in self.modalities:
            if sample_id in self.text_files:
                text_tensor = self._load_tensor(self.text_files[sample_id])
            elif sample_id in self.text_dict_bank:
                text_tensor = self.text_dict_bank[sample_id]
            else:
                raise KeyError(f"Text embedding missing for aligned sample: {sample_id}")

            if text_tensor.ndim > 1:
                text_tensor = text_tensor.squeeze()
            if self.transform_text is not None:
                text_tensor = self.transform_text(text_tensor)
            sample_dict["text"] = text_tensor.float()

        # 4. Load Clinical Label
        raw_label = self.labels_map.get(sample_id, 0)
        sample_dict["label"] = torch.tensor(raw_label, dtype=torch.long)

        return sample_dict

    def compute_pos_weight(self) -> torch.Tensor:
        """
        Computes BCEWithLogitsLoss pos_weight to correct for class imbalance.

        For binary classification where y=1 (ASD) is the majority class:
            pos_weight = n_negative / n_positive

        This down-weights the majority positive class so that the effective
        loss contribution from each class is balanced:
            Positive effective = n_pos * pos_weight = n_neg
            Negative effective = n_neg * 1.0        = n_neg

        Returns:
            torch.Tensor: Scalar pos_weight for BCEWithLogitsLoss.
        """
        labels = [self.labels_map.get(sid, 0) for sid in self.active_ids]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.warning("Single-class dataset detected. Returning pos_weight=1.0.")
            return torch.tensor([1.0])
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        logger.info(
            f"Class balance: {n_pos} positive (ASD), {n_neg} negative (Control). "
            f"pos_weight={pos_weight.item():.4f}"
        )
        return pos_weight

    def get_sampler_weights(self) -> torch.Tensor:
        """
        Computes per-sample weights for torch.utils.data.WeightedRandomSampler
        to achieve class-balanced mini-batches during training.

        Each sample receives weight inversely proportional to its class frequency:
            w_i = N / (2 * N_class_i)

        Usage:
            sampler = WeightedRandomSampler(
                weights=dataset.get_sampler_weights(),
                num_samples=len(dataset),
                replacement=True,
            )
            loader = DataLoader(dataset, batch_size=16, sampler=sampler)

        Returns:
            torch.Tensor: Per-sample weights of shape [len(dataset)].
        """
        labels = [self.labels_map.get(sid, 0) for sid in self.active_ids]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        n_total = len(labels)
        weight_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
        weights = torch.tensor(
            [weight_pos if l == 1 else weight_neg for l in labels],
            dtype=torch.float32,
        )
        return weights


# ==============================================================================
# Verification and Test Script
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing MultimodalAutismDataset Pipeline and DataLoader Iteration")
    print("=" * 70)

    # 1. Test 3-Modality Configuration (Video + Audio + Text)
    print("\n--- Test 1: 3-Modality Configuration (Video + Audio + Text) ---")
    dataset_3m = MultimodalAutismDataset(
        video_dir=Path("data/processed/video_landmarks"),
        audio_dir=Path("data/processed/audio_features"),
        text_dir=Path("data/processed/text_embeddings"),
        labels_file=Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
        modalities=("video", "audio", "text"),
        max_video_frames=50,
        flatten_landmarks=False,
    )
    print(f"[Dataset 3M] Total Aligned Samples: {len(dataset_3m)}")

    loader_3m = DataLoader(dataset_3m, batch_size=4, shuffle=True)
    batch_3m = next(iter(loader_3m))

    print(f"  • Video Tensor Shape      : {batch_3m['video'].shape} | Dtype: {batch_3m['video'].dtype}")
    print(f"  • Video Mask Shape        : {batch_3m['video_mask'].shape} | Dtype: {batch_3m['video_mask'].dtype}")
    print(f"  • Video Mask True Count   : {batch_3m['video_mask'].sum(dim=1).tolist()} / 50")
    print(f"  • Audio Tensor Shape      : {batch_3m['audio'].shape} | Dtype: {batch_3m['audio'].dtype}")
    print(f"  • Text Tensor Shape       : {batch_3m['text'].shape}  | Dtype: {batch_3m['text'].dtype}")
    print(f"  • Label Tensor Shape      : {batch_3m['label'].shape} | Dtype: {batch_3m['label'].dtype}")

    assert batch_3m["video"].dtype == torch.float32
    assert batch_3m["video_mask"].dtype == torch.bool
    assert batch_3m["audio"].dtype == torch.float32
    assert batch_3m["text"].dtype == torch.float32
    assert batch_3m["label"].dtype == torch.int64
    print("  => 3-Modality Test: PASSED")

    # 2. Test 2-Modality Configuration (Video + Audio)
    print("\n--- Test 2: 2-Modality Configuration (Video + Audio) ---")
    dataset_2m = MultimodalAutismDataset(
        video_dir=Path("data/processed/video_landmarks"),
        audio_dir=Path("data/processed/audio_features"),
        labels_file=Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
        modalities=("video", "audio"),
        max_video_frames=50,
    )
    print(f"[Dataset 2M] Total Aligned Samples: {len(dataset_2m)}")

    loader_2m = DataLoader(dataset_2m, batch_size=4, shuffle=True)
    batch_2m = next(iter(loader_2m))

    assert "text" not in batch_2m
    assert "video_mask" in batch_2m
    assert batch_2m["video"].shape == (4, 50, 92, 3)
    assert batch_2m["video_mask"].shape == (4, 50)
    assert batch_2m["audio"].shape == (4, 40, 313)
    print("  => 2-Modality Test: PASSED")

    print("\nAll MultimodalAutismDataset tests passed successfully!")
