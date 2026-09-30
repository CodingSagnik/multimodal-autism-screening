"""
tests/test_leakage_free.py

Automated test suite verifying the elimination of data leakage in the text modality:
1. Verifies that narrative text uses a uniform template structure across both classes.
2. Verifies that narratives do not contain template-injected label-discriminative phrasing.
3. Tests that a linear probe trained on text embeddings cannot achieve near-perfect
   classification, confirming that labels are not deterministically encoded.

Note: With Whisper ASR transcriptions, some genuine discriminative signal from
speech patterns is expected and acceptable. The test guards against systematic
template-based leakage, not natural linguistic variation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def test_narrative_leak_freedom():
    csv_path = Path("data/processed/text_embeddings/video_text_narratives.csv")
    assert csv_path.exists(), f"Narratives CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    # 1. Template-injection leakage tokens check
    # These tokens would ONLY appear if the narrative generation pipeline
    # injected label-derived discriminative phrasing (as in the original leaked v1).
    # Natural speech transcripts from Whisper ASR may contain clinical terms
    # (e.g., a parent saying "autism") — those are genuine signal, NOT leakage.
    injection_tokens = [
        "control baseline recording",
        "behavioral atypicalities",
        "no acute stereotypies",
        "exhibited pediatric behavioral",
        "typical pediatric behavior",
        "structured observational",
        "natural unstructured",
    ]

    for token in injection_tokens:
        matches = df["narrative"].str.contains(token, case=False)
        assert not matches.any(), (
            f"Template-injection leakage detected! Token '{token}' found in narratives. "
            f"This indicates the generation pipeline is injecting label-derived text."
        )

    print("  * [PASS] No template-injection leakage tokens found in narratives.")

    # 2. Structural uniformity check: same template prefix for both classes
    control_narratives = df[df["label"] == 0]["narrative"].tolist()
    asd_narratives = df[df["label"] == 1]["narrative"].tolist()

    assert len(control_narratives) > 0 and len(asd_narratives) > 0, (
        "Expected both ASD and Control narratives in the dataset."
    )

    expected_prefix = "pediatric behavioral screening clip"
    assert all(n.startswith(expected_prefix) for n in control_narratives), (
        "Control narratives do not use the expected template prefix."
    )
    assert all(n.startswith(expected_prefix) for n in asd_narratives), (
        "ASD narratives do not use the expected template prefix."
    )

    print(
        f"  * [PASS] Narrative template prefix is identical across classes "
        f"({len(control_narratives)} Control, {len(asd_narratives)} ASD-risk)."
    )

    # 3. Content diversity check: narratives should not all be identical
    # (identical narratives would indicate metadata-only mode with no real speech)
    unique_narratives = df["narrative"].nunique()
    print(f"  * [INFO] Unique narratives: {unique_narratives}/{len(df)}")
    if unique_narratives == len(df):
        print("  * [PASS] All narratives are unique (indicates genuine ASR content).")
    else:
        print(f"  * [WARN] {len(df) - unique_narratives} duplicate narratives detected.")


def test_text_embedding_linear_separability():
    pt_path = Path("data/processed/text_embeddings/video_text_embeddings.pt")
    assert pt_path.exists(), f"Embeddings bank not found: {pt_path}"
    data = torch.load(pt_path, weights_only=False)

    embeddings = data["embeddings"].numpy()  # [171, 768]
    labels = data["labels"].numpy()          # [171]

    # 5-fold cross-validated logistic regression should not achieve near-perfect accuracy.
    # With genuine ASR transcripts, some signal is expected (speech patterns differ),
    # but 100% accuracy would indicate systematic label encoding (leakage).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
    mean_acc = scores.mean()

    # Majority class baseline
    majority_baseline = max(np.mean(labels == 1), np.mean(labels == 0))
    print(f"  - Majority Class Baseline  : {majority_baseline:.3f}")
    print(f"  - 5-Fold Cross-Val Accuracy: {mean_acc:.3f} (+/- {scores.std():.3f})")

    # Accuracy must not be trivially near 100% (indicates template leakage).
    # Some genuine signal from ASR speech content is expected and acceptable.
    assert mean_acc < 0.95, (
        f"Suspiciously high accuracy ({mean_acc:.3f}) suggests potential residual leakage! "
        f"Expected < 0.95 for genuine ASR-derived text."
    )
    print(f"  * [PASS] Text embeddings do not trivially leak the diagnostic label (CV accuracy: {mean_acc:.3f}).")


if __name__ == "__main__":
    print("=" * 65)
    print("AUDIT VERIFICATION: TEXT MODALITY DATA LEAKAGE TEST SUITE")
    print("=" * 65)
    test_narrative_leak_freedom()
    test_text_embedding_linear_separability()
    print("=" * 65)
    print("All leakage-free audit assertions passed successfully!")
    print("=" * 65)
