"""
generate_video_text_embeddings.py

Generates leak-free clinical observation text narratives and dense embeddings
for the AV-ASD multimodal dataset clips using:
1. Whisper ASR transcription of raw audio clips (speech/vocalization content)
2. Dense DistilBERT pooled embeddings of the resulting clinical narratives

CRITICAL AUDIT RESOLUTION (Issues 1, 3, R1):
- Prior embeddings (v1) encoded symptom descriptions -> 100% data leakage.
- Prior embeddings (v2) used timestamp-only metadata -> 0% discriminative signal.
- This version (v3) uses Whisper ASR to extract genuine speech content from
  the raw audio, providing real multimodal text signal with zero label leakage.
- Falls back to metadata-only narratives when raw audio is unavailable.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ClinicalNarrativeDataset(Dataset):
    """PyTorch Dataset for batch tokenization of narrative strings."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def transcribe_audio_clips(
    raw_audio_dir: Path,
    video_ids: List[str],
    whisper_model: str = "openai/whisper-base",
    language: str = "en",
    device: Optional[str] = None,
) -> Dict[str, str]:
    """
    Transcribes raw audio clips using OpenAI Whisper ASR via HuggingFace pipeline.

    This extracts genuine speech/vocalization content from the audio recordings,
    providing authentic multimodal text signal without any label leakage.

    Args:
        raw_audio_dir: Directory containing raw audio files (.wav, .mp3, .flac).
        video_ids: List of video clip identifiers to transcribe.
        whisper_model: HuggingFace model identifier for Whisper ASR.
        language: Target language for transcription (default: "en").
        device: Compute device ("cuda" or "cpu"). Auto-detects if None.

    Returns:
        Dict mapping video_id -> transcription text.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError("transformers is required for Whisper ASR.")

    logger.info(f"Loading Whisper ASR model '{whisper_model}' on device '{device}'...")
    device_arg = 0 if device == "cuda" else -1
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        device=device_arg,
    )

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    transcripts: Dict[str, str] = {}

    logger.info(f"Transcribing {len(video_ids)} audio clips...")
    for vid in tqdm(video_ids, desc="Whisper ASR Transcription"):
        audio_file = None
        for ext in audio_extensions:
            candidate = raw_audio_dir / f"{vid}{ext}"
            if candidate.exists():
                audio_file = candidate
                break

        if audio_file is None:
            logger.debug(f"No audio file found for '{vid}', skipping ASR.")
            transcripts[vid] = ""
            continue

        try:
            result = asr(
                str(audio_file),
                return_timestamps=True,
                generate_kwargs={"language": language, "task": "transcribe"},
            )
            text = result.get("text", "").strip()
            transcripts[vid] = text
        except Exception as e:
            logger.warning(f"ASR failed for '{vid}': {e}")
            transcripts[vid] = ""

    n_transcribed = sum(1 for t in transcripts.values() if t)
    logger.info(
        f"ASR complete: {n_transcribed}/{len(video_ids)} clips produced transcriptions."
    )
    return transcripts


def synthesize_clean_narrative(
    video_id: str,
    transcript: str = "",
) -> str:
    """
    Synthesizes a clinical observation narrative combining:
    1. Neutral recording metadata (timestamps, duration)
    2. Whisper ASR transcription of speech/vocalization content (if available)

    No diagnostic labels, symptom descriptions, or class-discriminatory
    phrasing are injected into the narrative.
    """
    parts = video_id.rsplit("_", 2)
    if len(parts) == 3:
        source_id, start_str, end_str = parts
        try:
            start_sec = int(start_str)
            end_sec = int(end_str)
            duration = max(0, end_sec - start_sec)
        except ValueError:
            start_sec, end_sec, duration = 0, 0, 0
    else:
        source_id = video_id
        start_sec, end_sec, duration = 0, 0, 0

    clean_transcript = transcript.strip().lower() if transcript else ""

    if clean_transcript:
        narrative = (
            f"pediatric behavioral screening clip {video_id}. "
            f"recording interval from second {start_sec} to second {end_sec}, "
            f"duration {duration} seconds. "
            f"speech and vocalization transcript: {clean_transcript}"
        )
    else:
        narrative = (
            f"pediatric behavioral screening clip {video_id}. "
            f"recording interval from second {start_sec} to second {end_sec}, "
            f"duration {duration} seconds. "
            f"no intelligible speech or vocalization detected in audio segment."
        )
    return narrative


def extract_dense_embeddings(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "cls",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Extracts pooled dense embeddings from text using a Transformer backbone.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading transformer model '{model_name}' on device '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = ClinicalNarrativeDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings: List[torch.Tensor] = []

    logger.info(f"Extracting dense text embeddings for {len(texts)} clips using {pooling} pooling...")
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Extracting Text Embeddings"):
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            if pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                mask_exp = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                emb = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
            else:
                emb = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def generate_and_save_video_text_embeddings(
    video_dir: Union[str, Path] = Path("data/processed/video_landmarks"),
    raw_audio_dir: Optional[Union[str, Path]] = Path("data/raw/audio"),
    labels_file: Optional[Union[str, Path]] = Path("data/raw/AV-ASD_repo/dataset/csvs/dataset.csv"),
    output_dir: Union[str, Path] = Path("data/processed/text_embeddings"),
    model_name: str = "distilbert-base-uncased",
    whisper_model: str = "openai/whisper-base",
    language: str = "en",
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "cls",
    skip_asr: bool = False,
) -> None:
    """
    Full pipeline to generate leak-free clinical narratives with Whisper ASR
    speech transcriptions, extract dense Transformer embeddings, and save
    both individual .npy files and the monolithic video_text_embeddings.pt bank.
    """
    video_path = Path(video_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all video clip IDs
    video_files = sorted([f for f in video_path.iterdir() if f.suffix.lower() in [".npy", ".pt"]])
    if not video_files:
        raise FileNotFoundError(f"No processed video files found in {video_path}")

    video_ids = [f.stem for f in video_files]
    logger.info(f"Discovered {len(video_ids)} video clip IDs to process.")

    # 2. Load ground-truth labels (for metadata dictionary ONLY, NOT used in narratives)
    labels_map: Dict[str, int] = {}
    if labels_file and Path(labels_file).exists():
        df_labels = pd.read_csv(labels_file)
        df_labels.columns = [c.strip() for c in df_labels.columns]
        symptom_cols = [c for c in df_labels.columns if c not in ["Video_ID", "Background"]]
        for _, row in df_labels.iterrows():
            vid = str(row["Video_ID"]).strip()
            is_background = row.get("Background", 0) == 1
            has_symptoms = any(row.get(col, 0) == 1 for col in symptom_cols)
            labels_map[vid] = 0 if is_background or not has_symptoms else 1

    labels_list = [labels_map.get(vid, 0) for vid in video_ids]

    # 3. Transcribe audio clips with Whisper ASR (if raw audio is available)
    transcripts: Dict[str, str] = {}
    if not skip_asr and raw_audio_dir is not None:
        audio_path = Path(raw_audio_dir).resolve()
        if audio_path.exists() and audio_path.is_dir():
            transcripts = transcribe_audio_clips(
                raw_audio_dir=audio_path,
                video_ids=video_ids,
                whisper_model=whisper_model,
                language=language,
            )
        else:
            logger.warning(
                f"Raw audio directory not found: {audio_path}. "
                f"Falling back to metadata-only narratives (no ASR transcription)."
            )
    else:
        logger.info("ASR transcription skipped. Using metadata-only narratives.")

    # 4. Synthesize clinical narratives (with or without ASR transcription)
    narratives: List[str] = [
        synthesize_clean_narrative(vid, transcript=transcripts.get(vid, ""))
        for vid in video_ids
    ]

    # Save companion narratives CSV for inspection and auditing
    narratives_csv = out_dir / "video_text_narratives.csv"
    df_narratives = pd.DataFrame({
        "video_id": video_ids,
        "label": labels_list,
        "has_transcript": [bool(transcripts.get(vid, "").strip()) for vid in video_ids],
        "narrative": narratives,
    })
    df_narratives.to_csv(narratives_csv, index=False)
    logger.info(f"Saved clinical narratives to {narratives_csv}")

    # 5. Extract Transformer embeddings
    embeddings = extract_dense_embeddings(
        texts=narratives,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )  # [N, 768]

    # 6. Save individual .npy files
    logger.info(f"Saving individual .npy files to {out_dir}...")
    for idx, vid in enumerate(video_ids):
        emb_vec = embeddings[idx].numpy().astype(np.float32)
        np.save(out_dir / f"{vid}.npy", emb_vec)

    # 7. Save monolithic serialized PyTorch dictionary
    bank_path = out_dir / "video_text_embeddings.pt"
    n_transcribed = sum(1 for t in transcripts.values() if t.strip()) if transcripts else 0
    bank_dict = {
        "patient_ids": video_ids,
        "embeddings": embeddings,
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "clinical_texts": narratives,
        "metadata": {
            "model_name": model_name,
            "embedding_dim": str(embeddings.shape[1]),
            "num_samples": str(len(video_ids)),
            "pooling_method": pooling,
            "asr_model": whisper_model if not skip_asr else "none",
            "clips_with_transcript": str(n_transcribed),
            "leak_free_guarantee": "True - narratives derived from ASR speech content and neutral metadata only",
        },
    }
    torch.save(bank_dict, bank_path)
    logger.info(
        f"Successfully saved {len(video_ids)} text embeddings to {bank_path} "
        f"(Tensor shape: {embeddings.shape}, ASR transcriptions: {n_transcribed}/{len(video_ids)})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate leak-free clinical narratives with Whisper ASR and DistilBERT embeddings for AV-ASD video clips."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data/processed/video_landmarks",
        help="Path to processed video landmarks directory",
    )
    parser.add_argument(
        "--raw_audio_dir",
        type=str,
        default="data/raw/audio",
        help="Path to raw audio clips directory for Whisper ASR transcription",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default="data/raw/AV-ASD_repo/dataset/csvs/dataset.csv",
        help="Path to dataset.csv ground truth annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/text_embeddings",
        help="Output directory for text embeddings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Transformer model identifier for text embedding",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-base",
        help="Whisper ASR model identifier (default: openai/whisper-base)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Target language for ASR transcription (default: en)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token sequence length (default: 256, increased for ASR transcripts)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling method",
    )
    parser.add_argument(
        "--skip_asr",
        action="store_true",
        help="Skip Whisper ASR transcription and use metadata-only narratives",
    )

    args = parser.parse_args()

    generate_and_save_video_text_embeddings(
        video_dir=args.video_dir,
        raw_audio_dir=args.raw_audio_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        whisper_model=args.whisper_model,
        language=args.language,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        skip_asr=args.skip_asr,
    )
