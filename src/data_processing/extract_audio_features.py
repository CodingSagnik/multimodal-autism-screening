"""
extract_audio_features.py

Extracts standardized Mel-Frequency Cepstral Coefficients (MFCCs) from pediatric
behavioral audio clips. Resamples audio to 16kHz, standardizes clip length via
padding/truncation, and saves features as NumPy (.npy) or PyTorch (.pt) files.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AudioMFCCExtractor:
    """
    Standardized MFCC feature extractor with fixed-duration padding/truncation
    and optional delta feature computation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 10.0,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 512,
        include_deltas: bool = False,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_deltas = include_deltas
        self.normalize = normalize

    def load_and_standardize_audio(self, audio_path: Path) -> Tuple[np.ndarray, float]:
        """
        Loads audio resampled to target sample_rate and standardizes length
        using truncation (if longer than duration) or zero-padding (if shorter).

        Returns:
            (standardized_waveform, original_duration_in_seconds)
        """
        # Load audio and resample to standard sample rate
        y, orig_sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        original_duration = len(y) / self.sample_rate

        # Truncate if longer, pad with zeros if shorter
        if len(y) > self.target_samples:
            y_standardized = y[: self.target_samples]
        elif len(y) < self.target_samples:
            y_standardized = librosa.util.fix_length(y, size=self.target_samples)
        else:
            y_standardized = y

        return y_standardized.astype(np.float32), original_duration

    def extract_mfcc(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Computes MFCC matrix for standardized audio waveform.

        Returns:
            np.ndarray of shape (n_mfcc, n_frames) or (3 * n_mfcc, n_frames) if deltas are included.
        """
        mfcc = librosa.feature.mfcc(
            y=audio_waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        if self.include_deltas:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

        if self.normalize:
            # Cepstral Mean and Variance Normalization (CMVN)
            mean = np.mean(mfcc, axis=1, keepdims=True)
            std = np.std(mfcc, axis=1, keepdims=True) + 1e-8
            mfcc = (mfcc - mean) / std

        return mfcc.astype(np.float32)


def process_audio_file(
    audio_path: Path,
    extractor: AudioMFCCExtractor,
) -> Tuple[np.ndarray, Dict[str, Union[int, float, list]]]:
    """
    Processes a single audio file and extracts fixed-shape MFCCs.

    Returns:
        (mfcc_matrix, metadata_dict)
    """
    waveform, orig_duration = extractor.load_and_standardize_audio(audio_path)
    mfcc_features = extractor.extract_mfcc(waveform)

    meta = {
        "original_duration_sec": round(orig_duration, 2),
        "target_duration_sec": extractor.duration,
        "sample_rate": extractor.sample_rate,
        "mfcc_shape": list(mfcc_features.shape),
    }

    return mfcc_features, meta


def preprocess_all_audio(
    input_dir: Union[str, Path] = Path("data/raw/audio"),
    output_dir: Union[str, Path] = Path("data/processed/audio_features"),
    sample_rate: int = 16000,
    duration: float = 10.0,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
    include_deltas: bool = False,
    normalize: bool = False,
    output_format: str = "npy",
) -> None:
    """
    Iterates through all audio files in input_dir, computes fixed-length MFCCs,
    and saves them to output_dir with matching base filenames.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported audio file extensions
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted(
        [p for p in input_path.iterdir() if p.suffix.lower() in audio_extensions]
    )

    if not audio_files:
        logger.warning(f"No audio files found in {input_path}")
        return

    logger.info(f"Found {len(audio_files)} audio files in {input_path}")
    logger.info(
        f"Configuration: Sample Rate={sample_rate}Hz, Duration={duration}s, "
        f"n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}, "
        f"Deltas={include_deltas}, Format='{output_format}'"
    )

    extractor = AudioMFCCExtractor(
        sample_rate=sample_rate,
        duration=duration,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        include_deltas=include_deltas,
        normalize=normalize,
    )

    successful = 0
    failed = 0

    for audio_file in tqdm(audio_files, desc="Processing Audio"):
        try:
            mfcc_matrix, meta = process_audio_file(audio_file, extractor)

            if output_format.lower() == "pt":
                out_file = output_path / f"{audio_file.stem}.pt"
                tensor_mfcc = torch.from_numpy(mfcc_matrix)
                torch.save(tensor_mfcc, out_file)
            else:
                out_file = output_path / f"{audio_file.stem}.npy"
                np.save(out_file, mfcc_matrix)

            successful += 1
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")
            failed += 1

    logger.info(f"Preprocessing completed: {successful} succeeded, {failed} failed.")
    logger.info(f"Standardized audio feature files saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw pediatric audio recordings into standardized MFCC feature matrices."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/audio",
        help="Path to directory containing input audio files (default: data/raw/audio)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/audio_features",
        help="Path to directory for processed audio features (default: data/processed/audio_features)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio resampling rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Fixed duration in seconds for padding/truncation (default: 10.0)",
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients to extract (default: 40)",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=1024,
        help="FFT window size (default: 1024)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Hop length for STFT (default: 512)",
    )
    parser.add_argument(
        "--include_deltas",
        action="store_true",
        help="Include delta and delta-delta features (120 channels total)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply Cepstral Mean and Variance Normalization (CMVN)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npy",
        choices=["npy", "pt"],
        help="Output storage format: 'npy' (NumPy) or 'pt' (PyTorch tensor) (default: npy)",
    )

    args = parser.parse_args()

    preprocess_all_audio(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        include_deltas=args.include_deltas,
        normalize=args.normalize,
        output_format=args.format,
    )
