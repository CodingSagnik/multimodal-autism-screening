"""
extract_text_embeddings.py

Processes clinical M-CHAT-R questionnaire tabular and text responses, cleans
and standardizes behavioral descriptions, and extracts dense pooled embeddings
using a pre-trained Transformer (e.g., DistilBERT or ClinicalBERT).
Saves the serialized PyTorch dictionary to data/processed/text_embeddings/mchat_embedded.pt.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Clinical Semantic Mapping for M-CHAT-R Questionnaire Items
# ==============================================================================
MCHAT_ITEM_DESCRIPTIONS = {
    "A1": "joint attention and looking when pointed to",
    "A2": "auditory responsiveness and hearing concern",
    "A3": "pretend and imaginative play",
    "A4": "motor exploration and climbing on objects",
    "A5": "unusual repetitive finger movements near eyes",
    "A6": "protoimperative pointing to request help or objects",
    "A7": "protodeclarative pointing to share interest",
    "A8": "social interest in other children",
    "A9": "showing objects to caregivers to share interest",
    "A10": "orienting and responding to name when called",
}


class ClinicalTextDataset(Dataset):
    """PyTorch Dataset for batching clinical narrative texts."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def clean_and_format_clinical_text(
    df: pd.DataFrame,
    id_column: Optional[str] = None,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, List[str]]:
    """
    Cleans raw DataFrame, handles NaNs, standardizes casing, creates patient IDs,
    and synthesizes descriptive clinical narratives for transformer tokenization.

    Returns:
        patient_ids: list of patient identifiers
        clinical_texts: list of standardized clinical narrative strings
        labels: np.ndarray of binary class labels (1 for ASD risk, 0 for control)
        tabular_features: np.ndarray of numerical encoded features
        feature_names: list of feature names
    """
    df_clean = df.copy()

    # 1. Clean column names (strip whitespace and handle casing)
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Normalize known column name variations
    col_mapping = {
        "jaundice": "jaundice",
        "jauundice": "jaundice",
        "family_asd": "family_asd",
        "class/asd": "class",
        "class": "class",
        "sex": "sex",
        "age": "age",
    }
    df_clean.rename(
        columns={c: col_mapping.get(c.lower(), c) for c in df_clean.columns},
        inplace=True,
    )

    # 2. Assign or extract patient IDs
    if id_column and id_column in df_clean.columns:
        patient_ids = df_clean[id_column].astype(str).str.strip().tolist()
    else:
        # Generate standardized zero-padded patient IDs
        patient_ids = [f"PATIENT_{i:05d}" for i in range(len(df_clean))]

    # 3. Clean categorical text fields & handle NaNs
    text_cols = [c for c in df_clean.columns if df_clean[c].dtype == object]
    for col in text_cols:
        df_clean[col] = df_clean[col].fillna("unknown").astype(str).str.strip().str.lower()

    # Numeric columns
    num_cols = [c for c in df_clean.columns if df_clean[c].dtype in [np.int64, np.float64, int, float]]
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(0)

    # 4. Extract labels if present
    labels = np.zeros(len(df_clean), dtype=np.int64)
    if "class" in df_clean.columns:
        # Standardize YES/NO / 1/0
        class_series = df_clean["class"].astype(str).str.strip().str.upper()
        labels = np.where(class_series.isin(["YES", "1", "TRUE", "POSITIVE", "Y"]), 1, 0)

    # 5. Build standardized clinical narrative for each row
    clinical_texts: List[str] = []
    tabular_records: List[List[float]] = []

    mchat_cols = [f"A{i}" for i in range(1, 11)]

    for idx, row in df_clean.iterrows():
        pid = patient_ids[idx]
        age = row.get("age", "unknown")
        sex = row.get("sex", "unknown")
        jaundice = row.get("jaundice", "unknown")
        family_asd = row.get("family_asd", "unknown")

        # Describe questionnaire responses
        q_responses: List[str] = []
        tab_row: List[float] = []

        for col in mchat_cols:
            val = row.get(col, 0)
            try:
                val_int = int(float(val))
            except (ValueError, TypeError):
                val_int = 1 if str(val).lower() in ["yes", "y", "1", "true"] else 0

            tab_row.append(float(val_int))
            item_desc = MCHAT_ITEM_DESCRIPTIONS.get(col, f"item {col}")
            resp_str = "positive/present" if val_int == 1 else "negative/absent"
            q_responses.append(f"{item_desc}: {resp_str}")

        # Add demographic tabular values
        try:
            age_val = float(age) if age != "unknown" else 0.0
        except (ValueError, TypeError):
            age_val = 0.0
        tab_row.append(age_val)
        tab_row.append(1.0 if sex == "m" else 0.0)
        tab_row.append(1.0 if jaundice in ["yes", "y", "1", "true"] else 0.0)
        tab_row.append(1.0 if family_asd in ["yes", "y", "1", "true"] else 0.0)

        tabular_records.append(tab_row)

        # Synthesize structured clinical narrative
        narrative = (
            f"clinical screening record for {pid.lower()}. "
            f"toddler age: {age} months. biological sex: {sex}. "
            f"history of neonatal jaundice: {jaundice}. "
            f"family history of autism spectrum disorder: {family_asd}. "
            f"m-chat-r behavioral assessment responses: {'; '.join(q_responses)}."
        )
        clinical_texts.append(narrative)

    feature_names = mchat_cols + ["age_months", "is_male", "history_jaundice", "family_asd"]
    tabular_features = np.array(tabular_records, dtype=np.float32)

    return patient_ids, clinical_texts, labels, tabular_features, feature_names


def extract_embeddings(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
    pooling: str = "cls",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Extracts pooled dense embeddings for a list of clinical texts using a HuggingFace model.

    Args:
        texts: List of clinical narrative strings.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        max_length: Max sequence length for tokenizer.
        pooling: 'cls' ([CLS] token representation) or 'mean' (attention-weighted mean pooling).
        device: 'cuda', 'cpu', or None for auto-selection.

    Returns:
        torch.Tensor of shape (N, hidden_dim).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading tokenizer and model '{model_name}' on device '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = ClinicalTextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings: List[torch.Tensor] = []

    logger.info(f"Extracting embeddings for {len(texts)} records using {pooling} pooling...")
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
                # For BERT/DistilBERT, index 0 of last_hidden_state is the [CLS] representation
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                # Attention-weighted mean pooling
                input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def process_and_export_mchat_dataset(
    csv_path: Union[str, Path] = Path("data/raw/text/mchat_results.csv"),
    output_path: Union[str, Path] = Path("data/processed/text_embeddings/mchat_embedded.pt"),
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
    pooling: str = "cls",
    save_csv_companion: bool = True,
) -> None:
    """
    Full pipeline to clean M-CHAT questionnaire data, generate clinical narratives,
    extract Transformer embeddings, and save the serialized dataset dictionary.
    """
    csv_file = Path(csv_path).resolve()
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        raise FileNotFoundError(f"Input M-CHAT CSV file not found: {csv_file}")

    logger.info(f"Loading raw clinical data from {csv_file}...")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded raw dataset with shape {df.shape}")

    # Clean and synthesize clinical text
    patient_ids, clinical_texts, labels, tabular_features, feature_names = clean_and_format_clinical_text(df)

    # Extract dense embeddings
    embeddings = extract_embeddings(
        texts=clinical_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )

    # Prepare serialized dictionary
    processed_dict: Dict[str, Union[torch.Tensor, List[str], Dict[str, str]]] = {
        "patient_ids": patient_ids,
        "embeddings": embeddings,
        "labels": torch.from_numpy(labels),
        "tabular_features": torch.from_numpy(tabular_features),
        "feature_names": feature_names,
        "clinical_texts": clinical_texts,
        "metadata": {
            "model_name": model_name,
            "embedding_dim": str(embeddings.shape[1]),
            "num_samples": str(len(patient_ids)),
            "pooling_method": pooling,
            "raw_source_file": str(csv_file.name),
        },
    }

    # Save PyTorch serialized dictionary
    logger.info(f"Saving serialized dataset dictionary to {out_file}...")
    torch.save(processed_dict, out_file)
    logger.info(f"Successfully saved {len(patient_ids)} embedded patient records to {out_file} (Tensor shape: {embeddings.shape})")

    # Optional CSV companion export
    if save_csv_companion:
        companion_csv = out_file.with_suffix(".csv")
        df_export = pd.DataFrame({
            "patient_id": patient_ids,
            "label": labels,
            "clinical_narrative": clinical_texts,
        })
        df_export.to_csv(companion_csv, index=False)
        logger.info(f"Saved companion clinical text CSV to {companion_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process M-CHAT-R clinical questionnaire data and generate dense text embeddings using Hugging Face Transformers."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/raw/text/mchat_results.csv",
        help="Path to input M-CHAT CSV file (default: data/raw/text/mchat_results.csv)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/text_embeddings/mchat_embedded.pt",
        help="Path to output serialized PyTorch file (default: data/processed/text_embeddings/mchat_embedded.pt)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pre-trained HuggingFace model identifier (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for transformer inference (default: 64)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token sequence length (default: 128)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling strategy for sentence representation (default: cls)",
    )
    parser.add_argument(
        "--no_csv_companion",
        action="store_true",
        help="Disable exporting companion CSV with clinical narratives",
    )

    args = parser.parse_args()

    process_and_export_mchat_dataset(
        csv_path=args.csv_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        save_csv_companion=not args.no_csv_companion,
    )
