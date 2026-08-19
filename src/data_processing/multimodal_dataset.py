"""
multimodal_dataset.py

Custom PyTorch Dataset class (MultimodalAutismDataset) to align and load three
synchronized multimodal streams:
1. 3D Facial Landmarks (Video)
2. Standardized MFCCs (Audio)
3. Dense Clinical Text Embeddings (Text)
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

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
    """

    def __init__(
        self,
        video_dir: Union[str, Path] = Path("data/processed/video_landmarks"),
        audio_dir: Union[str, Path] = Path("data/processed/audio_features"),
        text_dir: Union[str, Path] = Path("data/processed/text_embeddings"),
        labels_file: Optional[Union[str, Path]] = Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
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
            max_video_frames: Optional fixed temporal length for video tensors (pads/truncates).
            flatten_landmarks: If True, flattens spatial landmark dims (T, N*3).
            transform_video: Optional transform/augmentation callable for video tensor.
            transform_audio: Optional transform/augmentation callable for audio tensor.
            transform_text: Optional transform/augmentation callable for text tensor.
        """
        self.video_dir = Path(video_dir).resolve()
        self.audio_dir = Path(audio_dir).resolve()
        self.text_dir = Path(text_dir).resolve()
        self.labels_file = Path(labels_file).resolve() if labels_file else None
        self.max_video_frames = max_video_frames
        self.flatten_landmarks = flatten_landmarks
        self.transform_video = transform_video
        self.transform_audio = transform_audio
        self.transform_text = transform_text

        # Validate directory existence
        for d_name, d_path in [
            ("Video", self.video_dir),
            ("Audio", self.audio_dir),
            ("Text", self.text_dir),
        ]:
            if not d_path.exists():
                raise FileNotFoundError(f"{d_name} directory not found: {d_path}")

        # Index available files in each modality
        self.video_files = self._scan_modality_files(self.video_dir)
        self.audio_files = self._scan_modality_files(self.audio_dir)
        self.text_files, self.text_dict_bank = self._scan_text_modality(self.text_dir)

        # Load labels mapping
        self.labels_map = self._load_labels_map(self.labels_file)

        # Compute strict intersection across modalities
        self.active_ids = self._align_modalities()

        logger.info(
            f"Dataset alignment complete: {len(self.active_ids)} perfectly matched multimodal samples."
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
        Discovers text embeddings from individual files or serialized PyTorch dictionary banks.
        """
        text_files: Dict[str, Path] = {}
        text_dict_bank: Dict[str, torch.Tensor] = {}

        # 1. Check for individual .npy / .pt files
        for f in directory.iterdir():
            if f.name in ["mchat_embedded.pt", "video_text_embeddings.pt"]:
                continue
            if f.suffix.lower() in [".npy", ".pt"]:
                text_files[f.stem] = f

        # 2. Check for monolithic dictionary banks (.pt)
        for dict_file_name in ["video_text_embeddings.pt", "mchat_embedded.pt"]:
            dict_path = directory / dict_file_name
            if dict_path.exists():
                try:
                    loaded = torch.load(dict_path, weights_only=False)
                    if isinstance(loaded, dict) and "patient_ids" in loaded and "embeddings" in loaded:
                        p_ids = loaded["patient_ids"]
                        embs = loaded["embeddings"]
                        if isinstance(embs, np.ndarray):
                            embs = torch.from_numpy(embs)
                        for i, pid in enumerate(p_ids):
                            text_dict_bank[str(pid)] = embs[i].float()
                except Exception as e:
                    logger.warning(f"Could not load dictionary bank from {dict_path}: {e}")

        return text_files, text_dict_bank

    def _load_labels_map(self, labels_path: Optional[Path]) -> Dict[str, int]:
        """Loads binary or categorical clinical labels from metadata CSV."""
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

                # Case 2: M-CHAT format with Class column
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
        Finds exact intersection of IDs present in video, audio, and text modalities.
        Drops any sample missing one or more modalities.
        """
        v_ids: Set[str] = set(self.video_files.keys())
        a_ids: Set[str] = set(self.audio_files.keys())
        t_ids: Set[str] = set(self.text_files.keys()).union(set(self.text_dict_bank.keys()))

        # Strict intersection across all 3 streams
        common_ids = sorted(list(v_ids.intersection(a_ids).intersection(t_ids)))

        total_union = v_ids.union(a_ids).union(t_ids)
        dropped_count = len(total_union) - len(common_ids)

        if dropped_count > 0:
            logger.info(
                f"Unaligned Sample Filter: {dropped_count} incomplete samples dropped. "
                f"(Video: {len(v_ids)}, Audio: {len(a_ids)}, Text: {len(t_ids)} -> Aligned: {len(common_ids)})"
            )

        if not common_ids:
            raise ValueError(
                "No overlapping samples found across video, audio, and text directories."
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
                "audio": torch.FloatTensor [n_mfcc, time_frames]
                "text": torch.FloatTensor [embedding_dim]
                "label": torch.LongTensor scalar
        """
        sample_id = self.active_ids[idx]

        # 1. Load Video Landmarks
        video_path = self.video_files[sample_id]
        video_tensor = self._load_tensor(video_path)  # Shape: [T, N, 3]

        # Optional temporal padding / truncation for batching
        if self.max_video_frames is not None:
            t_len = video_tensor.shape[0]
            if t_len > self.max_video_frames:
                video_tensor = video_tensor[: self.max_video_frames]
            elif t_len < self.max_video_frames:
                pad_shape = (self.max_video_frames - t_len,) + video_tensor.shape[1:]
                padding = torch.zeros(pad_shape, dtype=video_tensor.dtype)
                video_tensor = torch.cat([video_tensor, padding], dim=0)

        # Optional landmark spatial flattening: [T, N, 3] -> [T, N*3]
        if self.flatten_landmarks and video_tensor.ndim == 3:
            video_tensor = video_tensor.view(video_tensor.shape[0], -1)

        # 2. Load Audio MFCCs
        audio_path = self.audio_files[sample_id]
        audio_tensor = self._load_tensor(audio_path)  # Shape: [n_mfcc, time_frames]

        # 3. Load Text Embeddings
        if sample_id in self.text_files:
            text_tensor = self._load_tensor(self.text_files[sample_id])
        elif sample_id in self.text_dict_bank:
            text_tensor = self.text_dict_bank[sample_id]
        else:
            raise KeyError(f"Text embedding missing for aligned sample: {sample_id}")

        if text_tensor.ndim > 1:
            text_tensor = text_tensor.squeeze()

        # 4. Load Clinical Label
        raw_label = self.labels_map.get(sample_id, 0)
        label_tensor = torch.tensor(raw_label, dtype=torch.long)

        # Apply optional transforms
        if self.transform_video is not None:
            video_tensor = self.transform_video(video_tensor)
        if self.transform_audio is not None:
            audio_tensor = self.transform_audio(audio_tensor)
        if self.transform_text is not None:
            text_tensor = self.transform_text(text_tensor)

        return {
            "video": video_tensor.float(),
            "audio": audio_tensor.float(),
            "text": text_tensor.float(),
            "label": label_tensor,
            "sample_id": sample_id,
        }


# ==============================================================================
# Verification and Test Script
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing MultimodalAutismDataset Pipeline and DataLoader Iteration")
    print("=" * 70)

    # Initialize Dataset with paths to preprocessed modalities
    dataset = MultimodalAutismDataset(
        video_dir=Path("data/processed/video_landmarks"),
        audio_dir=Path("data/processed/audio_features"),
        text_dir=Path("data/processed/text_embeddings"),
        labels_file=Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
        max_video_frames=50,  # Standardize temporal length for uniform batch tensor collation
        flatten_landmarks=False,
    )

    print(f"\n[Dataset] Total Aligned Samples: {len(dataset)}")

    # Instantiate PyTorch DataLoader
    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for seamless execution on Windows
        drop_last=False,
    )

    print(f"[DataLoader] Initialized with batch size: {batch_size}")
    print("[DataLoader] Iterating through first batch...\n")

    # Fetch and verify first batch
    for batch_idx, batch in enumerate(dataloader):
        video_batch = batch["video"]
        audio_batch = batch["audio"]
        text_batch = batch["text"]
        label_batch = batch["label"]
        sample_ids = batch["sample_id"]

        print("-" * 50)
        print(f"Batch Index: {batch_idx + 1}")
        print(f"Sample IDs: {sample_ids}")
        print(f"  • Video Tensor Shape : {video_batch.shape} | Dtype: {video_batch.dtype}")
        print(f"  • Audio Tensor Shape : {audio_batch.shape} | Dtype: {audio_batch.dtype}")
        print(f"  • Text Tensor Shape  : {text_batch.shape}  | Dtype: {text_batch.dtype}")
        print(f"  • Label Tensor Shape : {label_batch.shape}        | Dtype: {label_batch.dtype}")
        print(f"  • Labels Values      : {label_batch.tolist()}")
        print("-" * 50)

        # Assert correct PyTorch tensor dtypes
        assert video_batch.dtype == torch.float32, "Video tensor must be float32"
        assert audio_batch.dtype == torch.float32, "Audio tensor must be float32"
        assert text_batch.dtype == torch.float32, "Text tensor must be float32"
        assert label_batch.dtype == torch.int64, "Label tensor must be int64 (long)"
        break

    print("\nDataset test completed successfully! All modalities aligned and verified.")
