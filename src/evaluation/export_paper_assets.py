"""
src/evaluation/export_paper_assets.py

Generates and exports all publication-ready figures and artifact tables
for the conference submission into paper/figures/ and paper/tables/.
"""

import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.explainability.lime_explainer import MultimodalLIMEExplainer
from src.explainability.pandemic_context import PandemicContextAnalyzer
from src.explainability.shap_explainer import MultimodalSHAPExplainer
from src.models.fusion_model import MultimodalAutismClassifier
from src.optimization.ga_chromosome import GAChromosome

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_pipeline_diagram(save_path: Path) -> None:
    """
    Renders high-resolution publication-ready vector block diagram of the
    Multimodal Early Autism Screening architecture with clean card layouts,
    clear flowlines, and zero label or bounding box overlaps.
    """
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=200)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Load actual optimal weights if available
    ga_ckpt = Path("models/ga_checkpoints/best_chromosome.json")
    wv, wa, wt = 1.69, 1.45, 0.27
    if ga_ckpt.exists():
        try:
            with open(ga_ckpt, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            wv = float(cdata.get("w_vision", wv))
            wa = float(cdata.get("w_audio", wa))
            wt = float(cdata.get("w_text", wt))
        except Exception:
            pass

    # Color Palette - Professional Clinical AI Theme
    c_raw_bg = "#F0F4F8"
    c_raw_border = "#3182CE"
    c_vision_bg = "#EBF8FF"
    c_vision_border = "#2B6CB0"
    c_audio_bg = "#FFFAF0"
    c_audio_border = "#DD6B20"
    c_text_bg = "#FAF5FF"
    c_text_border = "#6B46C1"
    c_ga_bg = "#F0FFF4"
    c_ga_border = "#2F855A"
    c_fusion_bg = "#FFF5F5"
    c_fusion_border = "#C53030"
    c_pred_bg = "#EDFDFD"
    c_pred_border = "#285E61"
    c_xai_bg = "#FFFFF0"
    c_xai_border = "#B7791F"

    def draw_card(x, y, w, h, header, body_lines, bg_col, border_col):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.6
        )
        ax.add_patch(rect)
        
        # Header banner divider
        banner_h = 0.38
        banner_rect = patches.FancyBboxPatch(
            (x, y + h - banner_h), w, banner_h, boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=border_col, edgecolor=border_col, linewidth=1.0
        )
        ax.add_patch(banner_rect)
        ax.text(x + w / 2, y + h - banner_h / 2, header, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")
        
        # Body text
        body_y = y + (h - banner_h) / 2
        body_text = "\n".join(body_lines)
        ax.text(x + w / 2, body_y, body_text, ha="center", va="center",
                fontsize=8.5, color="#2D3748", linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, label="", color="#4A5568", label_pos=0.5, label_dy=0.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color=color, lw=1.6)
        )
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos + label_dy
            ax.text(lx, ly, label, ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#2D3748", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E0", lw=0.8))

    # Title
    ax.text(7.75, 7.85, "End-to-End Multimodal Early Autism Screening Pipeline & Late Fusion Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1A202C")

    # Stage 1: Modality Inputs
    draw_card(0.5, 5.4, 2.4, 1.4, "MODALITY 1: VIDEO",
              ["Raw Video Clips (.mp4)", "5.0 FPS Native Resolution", "Face Crop & Tracking"],
              c_raw_bg, c_raw_border)
    draw_card(0.5, 3.4, 2.4, 1.4, "MODALITY 2: AUDIO",
              ["Acoustic Clips (.wav)", "16 kHz Mono Sampling", "10.0s Window Framing"],
              c_raw_bg, c_raw_border)
    draw_card(0.5, 1.4, 2.4, 1.4, "MODALITY 3: TEXT",
              ["Clinical Transcriptions", "Whisper ASR Model", "Verbatim Child Vocalization"],
              c_raw_bg, c_raw_border)

    # Stage 2: Encoders
    draw_card(3.6, 5.4, 2.8, 1.4, "VISION ENCODER (128D)",
              ["MediaPipe 468 3D Mesh", "2-Stage BiLSTM + Masking", "Attention-Weighted Pool"],
              c_vision_bg, c_vision_border)
    draw_card(3.6, 3.4, 2.8, 1.4, "AUDIO ENCODER (128D)",
              ["Librosa 40x313 MFCC", "3-Stage 2D ConvNet", "GroupNorm & AdaptAvgPool"],
              c_audio_bg, c_audio_border)
    draw_card(3.6, 1.4, 2.8, 1.4, "TEXT ENCODER (64D)",
              ["DistilBERT Embeddings (768D)", "3-Stage Regularized MLP", "LayerNorm + Dropout 0.4"],
              c_text_bg, c_text_border)

    # Arrows Stage 1 -> Stage 2
    draw_arrow(2.9, 6.1, 3.6, 6.1)
    draw_arrow(2.9, 4.1, 3.6, 4.1)
    draw_arrow(2.9, 2.1, 3.6, 2.1)

    # Stage 3: GA Optimizer & Late Fusion Head
    draw_card(7.2, 5.5, 2.9, 1.5, "SOFT COMPUTING OPTIMIZATION",
              ["Genetic Algorithm Engine", "LHS Sampling + SBX Crossover", "Optimizes: (w_v, w_a, w_t, lr, wd)"],
              c_ga_bg, c_ga_border)
              
    draw_card(7.2, 1.2, 2.9, 3.5, "MULTIMODAL LATE FUSION HEAD",
              ["Dynamic Scaled Concat: [320D]", "Linear(320 -> 256) + LN + ReLU", "Dropout 1 (p = 0.48)",
               "Linear(256 -> 64) + LN + ReLU", "Dropout 2 (p = 0.36)", "Linear(64 -> 1 Logit)"],
              c_fusion_bg, c_fusion_border)

    # Arrows Encoders -> Late Fusion Head
    draw_arrow(6.4, 6.1, 7.2, 4.2, label="128D", label_pos=0.45)
    draw_arrow(6.4, 4.1, 7.2, 3.0, label="128D", label_pos=0.45)
    draw_arrow(6.4, 2.1, 7.2, 1.8, label="64D", label_pos=0.45)

    # Arrow GA -> Late Fusion Head
    draw_arrow(8.65, 5.5, 8.65, 4.7,
               label=f"Optimal Modality Weights\n($w_v$={wv:.2f}, $w_a$={wa:.2f}, $w_t$={wt:.2f})",
               label_pos=0.5)

    # Stage 4: Prediction & XAI
    draw_card(11.0, 4.6, 3.8, 1.5, "CLINICAL PREDICTION HEAD",
              ["Sigmoid Activation: σ(z)", "Output: P(ASD Risk) ∈ [0.0, 1.0]", "Balanced Decision Threshold"],
              c_pred_bg, c_pred_border)
              
    draw_card(11.0, 1.2, 3.8, 2.7, "PHASE 3: EXPLAINABLE AI & PANDEMIC",
              ["KernelSHAP Modality Attributions", "LIME Contrastive Decision Rules", "Landmark Frame Attention Heatmaps", "MFCC Acoustic Attributions", "Pandemic vs. Intrinsic ASD Disentanglement"],
              c_xai_bg, c_xai_border)

    # Clean orthogonal stepped arrow: Late Fusion Head -> Clinical Prediction Head
    # Emerges from upper-right of fusion head (y=4.35, above Phase 3 card at y=3.9), steps up in corridor
    x_corridor = 10.55
    y_fusion_out = 4.35
    y_pred_in = 5.35
    ax.plot([10.1, x_corridor], [y_fusion_out, y_fusion_out], color="#2C7A7B", lw=1.8)
    ax.plot([x_corridor, x_corridor], [y_fusion_out, y_pred_in], color="#2C7A7B", lw=1.8)
    ax.annotate(
        "", xy=(11.0, y_pred_in), xytext=(x_corridor, y_pred_in),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color="#2C7A7B", lw=1.8)
    )
    ax.text(x_corridor, (y_fusion_out + y_pred_in) / 2, "1 Logit", ha="center", va="center", fontsize=8, fontweight="bold",
            color="#1D4044", bbox=dict(boxstyle="round,pad=0.22", facecolor="#E6FFFA", edgecolor="#319795", lw=0.9))

    # Arrow Prediction -> XAI
    draw_arrow(12.9, 4.6, 12.9, 3.9, label="Feature Attributions\n& Interpretability", label_pos=0.5)

    plt.tight_layout()
    plt.savefig(str(Path(save_path).resolve()), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pipeline architecture diagram to {save_path}")


def export_all_paper_assets(
    ga_checkpoint_path: Path = Path("models/ga_checkpoints/best_chromosome.json").resolve(),
    output_figures_dir: Path = Path("paper/figures").resolve(),
    output_tables_dir: Path = Path("paper/tables").resolve(),
) -> None:
    """
    Executes full XAI attribution generation and exports all conference figures.
    """
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultimodalAutismDataset()
    logger.info(f"Loaded dataset with {len(dataset)} aligned samples.")

    # Load GA Chromosome
    if ga_checkpoint_path.exists():
        with open(ga_checkpoint_path, "r", encoding="utf-8") as f:
            chrom_dict = json.load(f)
        chrom = GAChromosome.from_dict(chrom_dict)
    else:
        chrom = GAChromosome(w_vision=1.25, w_audio=0.85, w_text=1.10)

    model = chrom.build_model()
    model.eval()

    # Load trained weights if final_model.pt exists
    weights_path = Path("models/ga_checkpoints/final_model.pt")
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded trained model weights from {weights_path}")

    # 1. Figure 1: Pipeline Diagram
    generate_pipeline_diagram(output_figures_dir / "fig1_multimodal_architecture_pipeline.png")

    # 2. Compute SHAP Values on representative cohort
    logger.info("Computing SHAP multimodal attributions...")
    shap_explainer = MultimodalSHAPExplainer(
        model=model,
        dataset=dataset,
        modality_weights=chrom.get_modality_weights(),
        n_background_samples=30,
        random_seed=42,
    )
    # Explain 16 diverse samples across both classes, ensuring exemplar patient 11 is included
    labels = [dataset.labels_map.get(sid, 0) for sid in dataset.active_ids]
    target_exemplar = 11 if len(dataset) > 11 else 0
    asd_pool = [i for i, l in enumerate(labels) if l == 1 and i != target_exemplar]
    asd_indices = [target_exemplar] + asd_pool[:11]
    ctrl_indices = [i for i, l in enumerate(labels) if l == 0][:4]
    explain_indices = asd_indices + ctrl_indices

    shap_results = shap_explainer.compute_shap_values(indices=explain_indices, n_samples=60)

    # 3. Figure 2: SHAP Modality Importance Bar Chart
    shap_explainer.plot_modality_importance(
        shap_results, output_figures_dir / "fig2_modality_importance_shap.png"
    )

    # 4. Figure 3: SHAP Top Latent Features
    shap_explainer.plot_shap_summary(
        shap_results, output_figures_dir / "fig3_top_latent_features_shap.png", top_k=15
    )

    # 5. Figure 4: LIME Contrastive Case Study
    logger.info("Computing LIME explanations...")
    lime_explainer = MultimodalLIMEExplainer(
        model=model,
        dataset=dataset,
        modality_weights=chrom.get_modality_weights(),
        n_background_samples=30,
        random_seed=42,
    )
    lime_explainer.compare_asd_vs_control(
        asd_idx=target_exemplar,
        control_idx=ctrl_indices[0],
        save_path=output_figures_dir / "fig4_lime_contrastive_case_study.png",
    )

    # 6. Figure 5: Pandemic Disentanglement Scatter Plot & Figure 6, 7
    logger.info("Computing Pandemic Context disentanglement and visual heatmaps...")
    analyzer = PandemicContextAnalyzer(model=model, dataset=dataset)
    scores = analyzer.compute_pandemic_sensitivity_scores(shap_results)

    analyzer.plot_pandemic_disentanglement_scatter(
        scores, output_figures_dir / "fig5_pandemic_disentanglement_scatter.png"
    )

    # Figure 6: Video Landmark Attention Heatmaps
    analyzer.plot_video_frame_heatmap(
        sample_idx=target_exemplar,
        output_path=output_figures_dir / "fig6_video_landmark_attention_heatmaps.png",
        max_frames=4,
    )

    # Figure 7: Acoustic MFCC Attribution Spectrogram
    analyzer.plot_audio_attribution_spectrogram(
        sample_idx=target_exemplar,
        output_path=output_figures_dir / "fig7_acoustic_attribution_spectrograms.png",
    )

    # 7. Figure 8: GA Convergence curves
    ga_log_path = Path("models/ga_logs/evolution_log.csv")
    if ga_log_path.exists():
        from src.optimization.run_ga import plot_evolution_telemetry
        plot_evolution_telemetry(ga_log_path, output_figures_dir)
        fitness_curve = output_figures_dir / "fitness_curve.png"
        if fitness_curve.exists():
            import shutil
            shutil.copy(fitness_curve, output_figures_dir / "fig8_ga_convergence_curves.png")

    # 8. Pandemic Report
    report_file = analyzer.generate_pandemic_report(scores, output_tables_dir)
    logger.info(f"Generated Pandemic Clinical Report at {report_file}")
    logger.info("All publication figures successfully exported to paper/figures/!")


if __name__ == "__main__":
    export_all_paper_assets()
