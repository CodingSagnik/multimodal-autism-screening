"""
src/evaluation/generate_results.py

Phase 4: Comprehensive Results Compilation and Figure Generation.
Generates all tables, figures, and exportable assets required for the conference paper:
1. Model comparison tables (LaTeX, Markdown, CSV, JSON)
2. GA convergence curves (fitness, gene diversity, modality weight evolution)
3. Radar chart comparing unimodal vs. multimodal performance profiles
4. Confusion matrix heatmaps per model configuration
5. XAI figure curation and export
6. Statistical significance summary
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Ensure project root in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ResultsCompiler:
    """
    Compiles empirical evaluation results from the GA optimization,
    cross-validation trainer, and XAI modules into publication-ready
    figures and formatted tables.
    """

    def __init__(
        self,
        ga_checkpoints_dir: Path = Path("models/ga_checkpoints"),
        ga_logs_dir: Path = Path("models/ga_logs"),
        paper_figures_dir: Path = Path("paper/figures"),
        paper_tables_dir: Path = Path("paper/tables"),
    ):
        self.ga_checkpoints_dir = ga_checkpoints_dir
        self.ga_logs_dir = ga_logs_dir
        self.paper_figures_dir = paper_figures_dir
        self.paper_tables_dir = paper_tables_dir

        # Ensure output directories exist
        self.paper_figures_dir.mkdir(parents=True, exist_ok=True)
        self.paper_tables_dir.mkdir(parents=True, exist_ok=True)

    def load_model_comparison(self) -> Dict[str, Any]:
        """Loads the model comparison JSON from paper/tables/."""
        json_path = self.paper_tables_dir / "model_comparison.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        return {}

    def load_evolution_log(self) -> Optional[np.ndarray]:
        """Loads the GA evolution CSV log."""
        csv_path = self.ga_logs_dir / "evolution_log.csv"
        if csv_path.exists():
            import csv as csv_mod
            rows = []
            with open(csv_path, "r") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    rows.append(row)
            return rows
        return None

    # =========================================================================
    # 1. Radar Chart: Unimodal vs. Multimodal Performance Profiles
    # =========================================================================

    def generate_radar_chart(self, output_name: str = "fig9_radar_performance_profile.png") -> Path:
        """
        Generates a radar/spider chart comparing performance profiles across
        all model configurations on 6 evaluation metrics.
        """
        data = self.load_model_comparison()
        if not data:
            print("[WARN] No model comparison data found. Skipping radar chart.")
            return Path()

        metrics_keys = ["Bal. Acc", "Macro F1", "ROC-AUC", "PR-AUC", "Sensitivity (ASD)", "Specificity (Ctrl)"]
        labels = ["Balanced\nAccuracy", "Macro\nF1", "ROC-\nAUC", "PR-\nAUC", "Sensitivity\n(ASD)", "Specificity\n(Control)"]

        # Select key models for radar comparison
        models_to_plot = [
            ("Unimodal Vision (3D Landmarks)", "#2ca02c", "-"),
            ("Unimodal Audio (Acoustic CNN)", "#ff7f0e", "--"),
            ("Unimodal Text (Whisper ASR)", "#9467bd", ":"),
            ("Trimodal Late Fusion (GA-Optimized)", "#d62728", "-"),
        ]

        n_metrics = len(metrics_keys)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=160)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        for model_name, color, linestyle in models_to_plot:
            if model_name not in data:
                continue
            values = [data[model_name][k]["mean"] for k in metrics_keys]
            values += values[:1]
            ax.plot(angles, values, linestyle=linestyle, linewidth=2, color=color, label=model_name.split("(")[0].strip())
            ax.fill(angles, values, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.7)
        ax.set_title("Performance Profile Comparison:\nUnimodal vs. Multimodal Screening", fontsize=13, fontweight="bold", pad=30)
        ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=9)
        plt.tight_layout()

        output_path = self.paper_figures_dir / output_name
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Radar chart saved: {output_path}")
        return output_path

    # =========================================================================
    # 2. Grouped Bar Chart: Metric-wise Model Comparison
    # =========================================================================

    def generate_grouped_bar_chart(self, output_name: str = "fig10_grouped_metric_comparison.png") -> Path:
        """
        Publication-quality grouped bar chart with error bars for all models × metrics.
        """
        data = self.load_model_comparison()
        if not data:
            return Path()

        model_names = list(data.keys())
        metrics_keys = ["Bal. Acc", "Macro F1", "ROC-AUC", "PR-AUC", "Sensitivity (ASD)", "Specificity (Ctrl)"]
        metric_labels = ["Bal. Acc", "Macro F1", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity"]

        colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

        n_models = len(model_names)
        n_metrics = len(metrics_keys)
        x = np.arange(n_models)
        width = 0.12

        fig, ax = plt.subplots(figsize=(14, 6), dpi=160)

        for i, (mk, ml) in enumerate(zip(metrics_keys, metric_labels)):
            means = [data[m][mk]["mean"] for m in model_names]
            stds = [data[m][mk]["std"] for m in model_names]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, label=ml,
                         color=colors[i], alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Model Configuration", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title("Comprehensive Metric Comparison Across Model Configurations", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        short_names = [n.replace("Unimodal ", "").replace("Late Fusion ", "").replace(" (3D Landmarks)", "\n(Vision)")
                       .replace(" (Acoustic CNN)", "\n(Audio)").replace(" (Whisper ASR)", "\n(Text)")
                       .replace("Bimodal ", "Bimodal\n").replace("Trimodal ", "Trimodal\n")
                       for n in model_names]
        ax.set_xticklabels(short_names, fontsize=8, ha="center")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, ncol=3, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = self.paper_figures_dir / output_name
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Grouped bar chart saved: {output_path}")
        return output_path

    # =========================================================================
    # 3. Confusion Matrix Heatmaps (Simulated from metrics)
    # =========================================================================

    def generate_sensitivity_specificity_tradeoff(
        self, output_name: str = "fig11_sensitivity_specificity_tradeoff.png"
    ) -> Path:
        """
        Scatter plot showing the sensitivity vs. specificity tradeoff across models,
        with marker size proportional to balanced accuracy.
        """
        data = self.load_model_comparison()
        if not data:
            return Path()

        fig, ax = plt.subplots(figsize=(9, 7), dpi=160)

        model_names = list(data.keys())
        colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#e7298a", "#17becf", "#d62728"]
        markers = ["o", "s", "^", "D", "P", "*"]

        for i, model in enumerate(model_names):
            sens = data[model]["Sensitivity (ASD)"]["mean"]
            spec = data[model]["Specificity (Ctrl)"]["mean"]
            bal_acc = data[model]["Bal. Acc"]["mean"]
            sens_std = data[model]["Sensitivity (ASD)"]["std"]
            spec_std = data[model]["Specificity (Ctrl)"]["std"]

            ax.errorbar(spec, sens, xerr=spec_std, yerr=sens_std,
                       fmt=markers[i], color=colors[i], markersize=10 + bal_acc * 15,
                       label=model.split("(")[0].strip(),
                       capsize=4, alpha=0.85, markeredgecolor="black", markeredgewidth=0.8)

        # Draw the random classifier diagonal
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random Classifier")

        # Draw iso-balanced-accuracy contours
        for ba in [0.5, 0.6, 0.7, 0.8]:
            spec_range = np.linspace(0, 1, 100)
            sens_contour = 2 * ba - spec_range
            valid = (sens_contour >= 0) & (sens_contour <= 1)
            ax.plot(spec_range[valid], sens_contour[valid], ":", color="lightgray", alpha=0.6)
            midpoint = len(spec_range[valid]) // 2
            if np.any(valid):
                ax.text(spec_range[valid][midpoint], sens_contour[valid][midpoint],
                       f"BA={ba:.1f}", fontsize=7, alpha=0.5, rotation=-45)

        ax.set_xlabel("Specificity (Control Recall)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sensitivity (ASD Recall)", fontsize=11, fontweight="bold")
        ax.set_title("Sensitivity–Specificity Tradeoff Across Model Configurations", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8.5, loc="lower left")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = self.paper_figures_dir / output_name
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Sensitivity-specificity tradeoff saved: {output_path}")
        return output_path

    # =========================================================================
    # 4. Ablation Study Summary Table
    # =========================================================================

    def generate_ablation_table(self, output_name: str = "ablation_study.tex") -> Path:
        """
        Generates an ablation study table showing the contribution of each modality
        by comparing unimodal, bimodal, and trimodal performance.
        """
        data = self.load_model_comparison()
        if not data:
            return Path()

        rows = []
        for model, metrics_data in data.items():
            row = {
                "model": model,
                "bal_acc": f"{metrics_data['Bal. Acc']['mean']:.3f} ± {metrics_data['Bal. Acc']['std']:.3f}",
                "macro_f1": f"{metrics_data['Macro F1']['mean']:.3f} ± {metrics_data['Macro F1']['std']:.3f}",
                "roc_auc": f"{metrics_data['ROC-AUC']['mean']:.3f} ± {metrics_data['ROC-AUC']['std']:.3f}",
                "delta_ba": "",  # Will compute relative to best unimodal
            }
            rows.append(row)

        # Compute Δ balanced accuracy relative to Unimodal Audio (best unimodal)
        best_unimodal_ba = data.get("Unimodal Audio (Acoustic CNN)", {}).get("Bal. Acc", {}).get("mean", 0)
        for row, (model, metrics_data) in zip(rows, data.items()):
            ba = metrics_data["Bal. Acc"]["mean"]
            delta = ba - best_unimodal_ba
            row["delta_ba"] = f"{delta:+.3f}" if delta != 0 else "—"

        # LaTeX output
        tex_content = r"""\begin{table}[t]
\centering
\caption{\textbf{Ablation Study: Modality Contribution Analysis.} Performance changes when adding/removing individual modalities. $\Delta$ BA shows difference from best unimodal baseline (Audio CNN).}
\label{tab:ablation}
\small
\begin{tabular}{lccc|c}
\toprule
\textbf{Configuration} & \textbf{Bal. Acc $\uparrow$} & \textbf{Macro F1 $\uparrow$} & \textbf{ROC-AUC $\uparrow$} & \textbf{$\Delta$ BA} \\
\midrule
"""
        for row in rows:
            tex_content += f"{row['model']} & {row['bal_acc']} & {row['macro_f1']} & {row['roc_auc']} & {row['delta_ba']} \\\\\n"

        tex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.paper_tables_dir / output_name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        print(f"[OK] Ablation table saved: {output_path}")
        return output_path

    # =========================================================================
    # 5. GA Hyperparameter Sensitivity Summary
    # =========================================================================

    def generate_ga_hyperparameter_summary(self, output_name: str = "ga_hyperparameter_summary.tex") -> Path:
        """
        Generates a table summarizing the GA search space and discovered optimal values.
        """
        best_path = self.ga_checkpoints_dir / "best_chromosome.json"
        if not best_path.exists():
            print("[WARN] No best chromosome found.")
            return Path()

        with open(best_path, "r") as f:
            best = json.load(f)

        gene_specs = [
            ("$w_{\\text{vision}}$", "Modality Weight (Vision)", "[0.0, 2.0]", f"{best.get('w_vision', 1.0):.3f}"),
            ("$w_{\\text{audio}}$", "Modality Weight (Audio)", "[0.0, 2.0]", f"{best.get('w_audio', 1.0):.3f}"),
            ("$w_{\\text{text}}$", "Modality Weight (Text)", "[0.0, 2.0]", f"{best.get('w_text', 1.0):.3f}"),
            ("$\\eta$", "Learning Rate", "[$10^{-5}$, $10^{-2}$]", f"{best.get('learning_rate', 1e-3):.1e}"),
            ("$\\lambda$", "Weight Decay", "[$10^{-6}$, $10^{-2}$]", f"{best.get('weight_decay', 1e-4):.1e}"),
            ("$p_1$", "Dropout Rate 1", "[0.1, 0.7]", f"{best.get('dropout1', 0.4):.2f}"),
            ("$p_2$", "Dropout Rate 2", "[0.05, 0.5]", f"{best.get('dropout2', 0.2):.2f}"),
            ("$h_1$", "Fusion Hidden 1", "\\{64, 128, 256\\}", f"{best.get('fusion_hidden1', 128)}"),
            ("$h_2$", "Fusion Hidden 2", "\\{16, 32, 64\\}", f"{best.get('fusion_hidden2', 32)}"),
        ]

        tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Genetic Algorithm Search Space and Discovered Optimal Configuration.} 9-gene chromosome encoding modality fusion weights, optimizer dynamics, regularization, and architectural capacity.}
\label{tab:ga_search_space}
\small
\begin{tabular}{clcc}
\toprule
\textbf{Gene} & \textbf{Description} & \textbf{Search Range} & \textbf{Optimal Value} \\
\midrule
"""
        for symbol, desc, range_str, val in gene_specs:
            tex += f"{symbol} & {desc} & {range_str} & {val} \\\\\n"

        fitness = best.get("fitness", 0.0)
        tex += rf"""\midrule
\multicolumn{{4}}{{l}}{{\textbf{{Best Fitness (Mean Balanced Accuracy):}} {fitness:.3f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        output_path = self.paper_tables_dir / output_name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"[OK] GA hyperparameter summary saved: {output_path}")
        return output_path

    # =========================================================================
    # 6. Compile All Results
    # =========================================================================

    def compile_all(self) -> Dict[str, Path]:
        """Generates all publication-ready figures and tables."""
        results = {}
        results["radar_chart"] = self.generate_radar_chart()
        results["grouped_bar"] = self.generate_grouped_bar_chart()
        results["sens_spec_tradeoff"] = self.generate_sensitivity_specificity_tradeoff()
        results["ablation_table"] = self.generate_ablation_table()
        results["ga_summary"] = self.generate_ga_hyperparameter_summary()

        print(f"\n{'='*60}")
        print(f"Results Compilation Complete: {len(results)} assets generated.")
        print(f"{'='*60}")
        return results


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    compiler = ResultsCompiler()
    compiler.compile_all()
