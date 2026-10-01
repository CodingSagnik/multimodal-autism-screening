"""
tests/test_phase3_xai.py

Unit and integration tests for Phase 3 Explainable AI (XAI) and Pandemic Context Module.
Verifies:
1. MultimodalSHAPExplainer initialization, 320D feature space, and background sampling.
2. SHAP computation, modality aggregation, and visual plot generation.
3. MultimodalLIMEExplainer local surrogate decision rules.
4. PandemicContextAnalyzer sensitivity scores, video frame landmark heatmaps, and report compilation.
"""

from pathlib import Path
import shutil
import numpy as np
import torch

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier
from src.explainability.shap_explainer import MultimodalSHAPExplainer
from src.explainability.lime_explainer import MultimodalLIMEExplainer
from src.explainability.pandemic_context import PandemicContextAnalyzer

TEMP_TEST_DIR = Path("tests/temp_xai_test_artifacts")


def setup_module():
    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)


def teardown_module():
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)


def test_shap_explainer_initialization():
    """Verify SHAP explainer sets up 320D feature space."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    explainer = MultimodalSHAPExplainer(model, dataset, n_background_samples=10)
    assert len(explainer.feature_names) == 320
    assert explainer.background_features.shape[1] == 320
    assert explainer.background_features.shape[0] == 10


def test_shap_computation_and_plots():
    """Verify SHAP attribution calculation and visualization output."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    explainer = MultimodalSHAPExplainer(model, dataset, n_background_samples=10)
    results = explainer.compute_shap_values(indices=[0, 1], n_samples=20)

    assert results["shap_values"].shape == (2, 320)
    assert "vision" in results["modality_shap"]
    assert "audio" in results["modality_shap"]
    assert "text" in results["modality_shap"]
    assert len(results["predictions"]) == 2

    # Test plots
    imp_path = TEMP_TEST_DIR / "shap_modality_importance.png"
    sum_path = TEMP_TEST_DIR / "shap_summary.png"
    wf_path = TEMP_TEST_DIR / "shap_waterfall.png"

    explainer.plot_modality_importance(results, imp_path)
    explainer.plot_shap_summary(results, sum_path)
    explainer.plot_sample_waterfall(results, sample_idx=0, save_path=wf_path)

    assert imp_path.exists()
    assert sum_path.exists()
    assert wf_path.exists()


def test_lime_explainer():
    """Verify LIME local surrogate explanations and comparison."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    explainer = MultimodalLIMEExplainer(model, dataset, n_background_samples=10)
    exp = explainer.explain_sample(0, num_features=5)

    assert exp["sample_id"] is not None
    assert abs(exp["prob_control"] + exp["prob_asd"] - 1.0) < 1e-3
    assert len(exp["feature_weights"]) <= 5

    plot_path = TEMP_TEST_DIR / "lime_sample_0.png"
    explainer.plot_sample_explanation(exp, plot_path)
    assert plot_path.exists()


def test_pandemic_context_analyzer():
    """Verify pandemic sensitivity ratios, video heatmaps, and report compilation."""
    dataset = MultimodalAutismDataset()
    model = MultimodalAutismClassifier()

    shap_exp = MultimodalSHAPExplainer(model, dataset, n_background_samples=10)
    shap_res = shap_exp.compute_shap_values(indices=[0, 1], n_samples=20)

    analyzer = PandemicContextAnalyzer(model, dataset)
    scores = analyzer.compute_pandemic_sensitivity_scores(shap_res)

    assert len(scores["per_sample_breakdown"]) == 2
    for b in scores["per_sample_breakdown"]:
        assert 0.0 <= b["pandemic_sensitivity"] <= 1.0
        assert 0.0 <= b["asd_specificity"] <= 1.0

    # Video heatmap
    hm_path = analyzer.plot_video_frame_heatmap(0, TEMP_TEST_DIR / "heatmap_test.png")
    assert hm_path.exists()

    # Audio spectrogram
    sp_path = analyzer.plot_audio_attribution_spectrogram(0, TEMP_TEST_DIR / "spectrogram_test.png")
    assert sp_path.exists()

    # Pandemic report
    rep_path = analyzer.generate_pandemic_report(scores, TEMP_TEST_DIR)
    assert rep_path.exists()
    assert (TEMP_TEST_DIR / "pandemic_disentanglement_scatter.png").exists()


if __name__ == "__main__":
    setup_module()
    try:
        print("Running Phase 3 XAI & Pandemic Context Tests...")
        test_shap_explainer_initialization()
        print("  • test_shap_explainer_initialization: PASSED")
        test_shap_computation_and_plots()
        print("  • test_shap_computation_and_plots: PASSED")
        test_lime_explainer()
        print("  • test_lime_explainer: PASSED")
        test_pandemic_context_analyzer()
        print("  • test_pandemic_context_analyzer: PASSED")
        print("All Phase 3 XAI & Pandemic Context Tests PASSED!")
    finally:
        teardown_module()
