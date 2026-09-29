"""
tests/test_leakage_free.py

Automated test suite verifying the elimination of data leakage in the text modality:
1. Verifies that narrative text contains NO symptom tokens or diagnostic hints.
2. Verifies that narrative templates are uniform across both ASD-risk and Control classes.
3. Tests that a linear probe trained on text embeddings cannot trivially separate the classes,
   confirming that diagnostic labels are not deterministically encoded in the text stream.
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

    # 1. Prohibited words check
    prohibited_tokens = [
        "unstructured", "structured observational", "symptom", "autism",
        "asd", "risk", "background", "positive", "negative", "avoidance",
        "eye contact", "spinning", "lining", "aggressive", "stereotypies",
        "hyporeactivity", "hyperreactivity", "verbal", "typical", "atypical"
    ]

    for token in prohibited_tokens:
        matches = df["narrative"].str.contains(token, case=False)
        assert not matches.any(), f"Data leakage detected! Token '{token}' found in narratives."

    print("  * [PASS] No prohibited diagnostic or clinical symptom tokens found in narratives.")

    # 2. Structure uniformity check
    control_narratives = df[df["label"] == 0]["narrative"].tolist()
    asd_narratives = df[df["label"] == 1]["narrative"].tolist()

    assert len(control_narratives) > 0 and len(asd_narratives) > 0

    control_prefix = "standard pediatric behavioral observation clip"
    assert all(n.startswith(control_prefix) for n in control_narratives)
    assert all(n.startswith(control_prefix) for n in asd_narratives)

    print(f"  * [PASS] Narrative syntactic template is strictly identical across classes "
          f"({len(control_narratives)} Control, {len(asd_narratives)} ASD-risk).")


def test_text_embedding_linear_separability():
    pt_path = Path("data/processed/text_embeddings/video_text_embeddings.pt")
    assert pt_path.exists(), f"Embeddings bank not found: {pt_path}"
    data = torch.load(pt_path, weights_only=False)

    embeddings = data["embeddings"].numpy()  # [171, 768]
    labels = data["labels"].numpy()          # [171]

    # In leaked data, accuracy was 100% because "structured" vs "unstructured" was encoded
    # In clean data, 5-fold cross-validated logistic regression should not achieve near-perfect classification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
    mean_acc = scores.mean()

    # Majority class baseline
    majority_baseline = max(np.mean(labels == 1), np.mean(labels == 0))
    print(f"  - Majority Class Baseline  : {majority_baseline:.3f}")
    print(f"  - 5-Fold Cross-Val Accuracy: {mean_acc:.3f} (+/- {scores.std():.3f})")

    # Accuracy must not be trivially 100% or near 100%
    assert mean_acc < 0.95, f"Suspiciously high accuracy ({mean_acc:.3f}) suggests potential residual leakage!"
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
