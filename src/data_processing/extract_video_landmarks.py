"""
extract_video_landmarks.py

Extracts 3D spatial facial landmarks (eyes, mouth, social engagement features)
from pediatric behavioral video clips at a fixed sampling rate using MediaPipe Face Mesh.
Saves preprocessed sequences as NumPy (.npy) or PyTorch (.pt) tensors.
"""

import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe is required. Please install via 'pip install mediapipe'.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Landmark Subset Definitions (MediaPipe Face Mesh 468-point index mapping)
# ==============================================================================
# Left and Right Eye contours and key interaction points
LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Left and Right Eyebrows (crucial for affect, joint attention, and emotion cues)
LEFT_EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_LANDMARKS = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Lips / Mouth (outer & inner contours for vocalization, smiling, social reciprocity)
LIPS_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Combined social engagement subset (eyes + eyebrows + mouth: 92 unique landmarks)
SOCIAL_ENGAGEMENT_LANDMARKS = sorted(
    list(set(LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + LEFT_EYEBROW_LANDMARKS + RIGHT_EYEBROW_LANDMARKS + LIPS_LANDMARKS))
)


class MediaPipeFaceMeshExtractor:
    """
    Robust Face Mesh extractor supporting both MediaPipe Tasks API (>=0.10.0 / 1.0+)
    and legacy MediaPipe Solutions API (<0.10.0).
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(
        self,
        model_path: Union[str, Path] = Path("models/face_landmarker.task"),
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.use_tasks_api = False

        # Try Tasks API first (MediaPipe 0.10.x / 1.0+)
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            # Auto-download model asset if missing
            if not self.model_path.exists():
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading MediaPipe Face Landmarker model to {self.model_path}...")
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                logger.info("Face Landmarker model downloaded successfully.")

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_presence_confidence,
            )
            self.detector = FaceLandmarker.create_from_options(options)
            self.use_tasks_api = True
            logger.info("Initialized MediaPipe FaceLandmarker via Tasks API.")
        except Exception as tasks_err:
            logger.debug(f"Tasks API unavailable or failed ({tasks_err}). Falling back to legacy Solutions API...")
            # Fallback to legacy solutions API (MediaPipe <0.10.x)
            try:
                self.detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_presence_confidence,
                )
                self.use_tasks_api = False
                logger.info("Initialized MediaPipe FaceMesh via legacy Solutions API.")
            except Exception as legacy_err:
                raise RuntimeError(
                    f"Failed to initialize MediaPipe Face Mesh with both Tasks API ({tasks_err}) and Solutions API ({legacy_err})."
                )

    def extract_frame_landmarks(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 468 3D (x, y, z) landmarks from a single RGB frame.
        Returns:
            np.ndarray of shape (468, 3) or None if no face is detected.
        """
        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                landmarks = result.face_landmarks[0]
                return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            return None
        else:
            result = self.detector.process(rgb_frame)
            if result.multi_face_landmarks and len(result.multi_face_landmarks) > 0:
                landmarks = result.multi_face_landmarks[0].landmark
                return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            return None


def get_landmark_indices(subset: str) -> List[int]:
    """
    Returns the landmark indices corresponding to the chosen anatomical subset.
    """
    subset = subset.lower()
    if subset == "social_engagement":
        return SOCIAL_ENGAGEMENT_LANDMARKS
    elif subset == "eyes":
        return sorted(list(set(LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS)))
    elif subset == "mouth" or subset == "lips":
        return sorted(list(set(LIPS_LANDMARKS)))
    elif subset == "all":
        return list(range(468))
    else:
        raise ValueError(
            f"Unknown subset '{subset}'. Choose from: 'social_engagement', 'eyes', 'mouth', 'all'."
        )


def process_video(
    video_path: Path,
    extractor: MediaPipeFaceMeshExtractor,
    target_fps: float = 5.0,
    landmark_indices: Optional[List[int]] = None,
    imputation_strategy: str = "last_known",
) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
    """
    Processes a single video file, sampling frames at target_fps, extracting landmarks,
    and handling missing face detections.

    Args:
        video_path: Path to the input video.
        extractor: MediaPipeFaceMeshExtractor instance.
        target_fps: Target sampling rate in frames per second.
        landmark_indices: List of landmark indices to retain. If None, uses all 468.
        imputation_strategy: 'last_known' (forward-fill with zero fallback) or 'zero'.

    Returns:
        tuple: (landmark_sequence array of shape [T, num_landmarks, 3], metadata dict)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0  # Fallback default

    # Determine frame sampling interval
    sample_interval = max(1, int(round(video_fps / target_fps)))

    if landmark_indices is None:
        num_landmarks = 468
    else:
        num_landmarks = len(landmark_indices)

    frame_idx = 0
    sampled_frames_count = 0
    detected_faces_count = 0

    extracted_sequence: List[Optional[np.ndarray]] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            sampled_frames_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_468 = extractor.extract_frame_landmarks(rgb_frame)

            if landmarks_468 is not None:
                detected_faces_count += 1
                if landmark_indices is not None:
                    selected_landmarks = landmarks_468[landmark_indices, :]
                else:
                    selected_landmarks = landmarks_468
                extracted_sequence.append(selected_landmarks)
            else:
                extracted_sequence.append(None)

        frame_idx += 1

    cap.release()

    if sampled_frames_count == 0:
        # Empty video fallback
        final_array = np.zeros((1, num_landmarks, 3), dtype=np.float32)
        meta = {
            "total_sampled_frames": 0,
            "detected_frames": 0,
            "detection_rate": 0.0,
        }
        return final_array, meta

    # Handle missing detections (imputation)
    imputed_sequence: List[np.ndarray] = []
    last_valid_frame: Optional[np.ndarray] = None

    for frame_landmarks in extracted_sequence:
        if frame_landmarks is not None:
            imputed_sequence.append(frame_landmarks)
            last_valid_frame = frame_landmarks
        else:
            if imputation_strategy == "last_known" and last_valid_frame is not None:
                imputed_sequence.append(last_valid_frame.copy())
            else:
                # Zero padding
                imputed_sequence.append(np.zeros((num_landmarks, 3), dtype=np.float32))

    # If the video began with missing frames before any face was detected,
    # backfill with the first detected face if using 'last_known'
    if imputation_strategy == "last_known" and last_valid_frame is not None:
        first_valid = None
        for item in extracted_sequence:
            if item is not None:
                first_valid = item
                break
        if first_valid is not None:
            for i in range(len(imputed_sequence)):
                if extracted_sequence[i] is None:
                    # If this was in the leading block before first detection
                    if np.all(imputed_sequence[i] == 0):
                        imputed_sequence[i] = first_valid.copy()
                else:
                    break

    final_array = np.stack(imputed_sequence, axis=0).astype(np.float32)

    detection_rate = (detected_faces_count / sampled_frames_count) * 100.0 if sampled_frames_count > 0 else 0.0
    meta = {
        "video_fps": video_fps,
        "total_video_frames": total_frames,
        "total_sampled_frames": sampled_frames_count,
        "detected_frames": detected_faces_count,
        "detection_rate_pct": round(detection_rate, 2),
        "sequence_shape": list(final_array.shape),
    }

    return final_array, meta


def preprocess_all_videos(
    input_dir: Union[str, Path] = Path("data/raw/video"),
    output_dir: Union[str, Path] = Path("data/processed/video_landmarks"),
    model_path: Union[str, Path] = Path("models/face_landmarker.task"),
    target_fps: float = 5.0,
    subset: str = "social_engagement",
    imputation_strategy: str = "last_known",
    output_format: str = "npy",
) -> None:
    """
    Iterates through all .mp4 video files in input_dir, extracts landmark sequences,
    and saves them to output_dir with matching base filenames.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    model_file = Path(model_path).resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(input_path.glob("*.mp4")))
    if not video_files:
        # Case insensitive fallback on Windows
        video_files = sorted([p for p in input_path.iterdir() if p.suffix.lower() == ".mp4"])

    if not video_files:
        logger.warning(f"No .mp4 files found in {input_path}")
        return

    landmark_indices = get_landmark_indices(subset)
    num_selected = len(landmark_indices)
    logger.info(f"Found {len(video_files)} video files in {input_path}")
    logger.info(
        f"Configuration: Target FPS={target_fps}, Landmark Subset='{subset}' ({num_selected} points), "
        f"Imputation='{imputation_strategy}', Format='{output_format}'"
    )

    extractor = MediaPipeFaceMeshExtractor(model_path=model_file)

    successful = 0
    failed = 0

    for video_file in tqdm(video_files, desc="Processing Videos"):
        try:
            landmarks_seq, meta = process_video(
                video_path=video_file,
                extractor=extractor,
                target_fps=target_fps,
                landmark_indices=landmark_indices,
                imputation_strategy=imputation_strategy,
            )

            # Output filename matches video stem
            if output_format.lower() == "pt":
                out_file = output_path / f"{video_file.stem}.pt"
                tensor_seq = torch.from_numpy(landmarks_seq)
                torch.save(tensor_seq, out_file)
            else:
                out_file = output_path / f"{video_file.stem}.npy"
                np.save(out_file, landmarks_seq)

            successful += 1

        except Exception as e:
            logger.error(f"Error processing {video_file.name}: {e}")
            failed += 1

    logger.info(f"Preprocessing completed: {successful} succeeded, {failed} failed.")
    logger.info(f"Output landmark files saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess pediatric behavioral videos using MediaPipe Face Mesh."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/video",
        help="Path to directory containing input .mp4 files (default: data/raw/video)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/video_landmarks",
        help="Path to directory for processed landmark sequences (default: data/processed/video_landmarks)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/face_landmarker.task",
        help="Path to MediaPipe face landmarker model asset (default: models/face_landmarker.task)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target sampling rate in frames per second (default: 5.0)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="social_engagement",
        choices=["social_engagement", "eyes", "mouth", "all"],
        help="Landmark subset to extract (default: social_engagement [92 landmarks: eyes + mouth + eyebrows])",
    )
    parser.add_argument(
        "--imputation",
        type=str,
        default="last_known",
        choices=["last_known", "zero"],
        help="Missing face imputation strategy (default: last_known)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npy",
        choices=["npy", "pt"],
        help="Output storage format: 'npy' (NumPy) or 'pt' (PyTorch tensor) (default: npy)",
    )

    args = parser.parse_args()

    preprocess_all_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        target_fps=args.fps,
        subset=args.subset,
        imputation_strategy=args.imputation,
        output_format=args.format,
    )
