"""
src/optimization/run_ga.py

Command-line entry point for Genetic Algorithm Optimization of Multimodal Autism Screening (Phase 3).
Executes:
1. Evolutionary search over modality fusion weights and hyperparameters.
2. Full 5-fold cross-validation of the best-found chromosome.
3. Training of the final deployment model on the entire dataset.
4. Generation of evolution telemetry plots (fitness curves, modality weights, convergence).
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.optimization.ga_chromosome import GAChromosome
from src.optimization.genetic_algorithm import GeneticAlgorithmEngine
from src.training.trainer import AutismScreeningTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def plot_evolution_telemetry(log_csv_path: Path, output_dir: Path) -> None:
    """
    Generates evolution telemetry plots from the GA log CSV:
    1. Fitness curve (Best, Mean, Worst).
    2. Modality weights evolution (w_vision, w_audio, w_text).
    3. Regularization & learning rate convergence.
    """
    if not log_csv_path.exists():
        logger.warning(f"Log CSV not found at {log_csv_path}. Skipping plot generation.")
        return

    df = pd.read_csv(log_csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fitness Curves Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(df["generation"], df["best_fitness"], "b-o", linewidth=2.0, label="Best (Elitist) Fitness")
    plt.plot(df["generation"], df["mean_fitness"], "g--s", linewidth=1.5, label="Population Mean")
    if "worst_fitness" in df.columns:
        plt.plot(df["generation"], df["worst_fitness"], "r:", linewidth=1.2, label="Population Worst")
    plt.fill_between(
        df["generation"],
        df["mean_fitness"] - df.get("std_fitness", 0.0),
        df["mean_fitness"] + df.get("std_fitness", 0.0),
        alpha=0.15,
        color="green",
        label="±1 Std Dev",
    )
    plt.title("Genetic Algorithm Optimization - Fitness Progression", fontsize=14, fontweight="bold")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Validation Balanced Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    fitness_plot = output_dir / "fitness_curve.png"
    plt.savefig(fitness_plot)
    plt.close()
    logger.info(f"Saved fitness curve to {fitness_plot}")

    # 2. Modality Weights Evolution Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(df["generation"], df["w_vision"], "c-^", linewidth=2.0, label=r"Vision Weight ($w_{v}$)")
    plt.plot(df["generation"], df["w_audio"], "m-v", linewidth=2.0, label=r"Audio Weight ($w_{a}$)")
    plt.plot(df["generation"], df["w_text"], "y-d", linewidth=2.0, label=r"Text Weight ($w_{t}$)")
    plt.axhline(1.0, color="gray", linestyle=":", label="Baseline (w=1.0)")
    plt.title("Dynamic Modality Fusion Weights Evolution", fontsize=14, fontweight="bold")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Optimal Weight Multiplier", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    weights_plot = output_dir / "modality_weights_evolution.png"
    plt.savefig(weights_plot)
    plt.close()
    logger.info(f"Saved modality weights plot to {weights_plot}")

    # 3. Learning Rate and Regularization Convergence
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    color = "tab:blue"
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Learning Rate (log-scale)", color=color, fontsize=12)
    ax1.semilogy(df["generation"], df["learning_rate"], color=color, marker="o", linewidth=2.0, label="Learning Rate")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Dropout 1 Rate", color=color, fontsize=12)
    ax2.plot(df["generation"], df["dropout1"], color=color, marker="s", linestyle="--", linewidth=1.5, label="Dropout 1")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Hyperparameter Convergence Trajectory", fontsize=14, fontweight="bold")
    plt.tight_layout()
    hyper_plot = output_dir / "gene_convergence.png"
    plt.savefig(hyper_plot)
    plt.close()
    logger.info(f"Saved hyperparameter convergence plot to {hyper_plot}")


def run_pipeline(
    population_size: int = 20,
    n_generations: int = 30,
    fitness_folds: int = 3,
    fitness_epochs: int = 25,
    final_folds: int = 5,
    final_epochs: int = 50,
    random_seed: int = 42,
    resume: Optional[str] = None,
    checkpoint_dir: Path = Path("models/ga_checkpoints"),
    log_dir: Path = Path("models/ga_logs"),
) -> None:
    """
    Full GA optimization and post-hoc validation pipeline.
    """
    logger.info("=" * 70)
    logger.info("Initializing Phase 3: Genetic Algorithm Optimization Pipeline")
    logger.info("=" * 70)

    # 1. Load Aligned Dataset
    dataset = MultimodalAutismDataset()
    logger.info(f"Loaded MultimodalAutismDataset with {len(dataset)} aligned samples.")

    # 2. Instantiate and Run GA Engine
    engine = GeneticAlgorithmEngine(
        dataset=dataset,
        population_size=population_size,
        n_generations=n_generations,
        fitness_folds=fitness_folds,
        fitness_epochs=fitness_epochs,
        random_seed=random_seed,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    best_chrom = engine.evolve(resume_checkpoint=resume)
    logger.info("\n" + "=" * 70)
    logger.info("GA Evolutionary Optimization Complete!")
    logger.info("=" * 70)
    logger.info(f"Best Candidate Fitness (3-fold Bal Acc): {best_chrom.fitness:.4f}")
    logger.info(
        f"Optimal Modality Weights: Vision={best_chrom.w_vision:.3f}, "
        f"Audio={best_chrom.w_audio:.3f}, Text={best_chrom.w_text:.3f}"
    )
    logger.info(
        f"Optimal Hyperparameters: LR={best_chrom.learning_rate:.2e}, "
        f"WD={best_chrom.weight_decay:.2e}, Dropout1={best_chrom.dropout1:.2f}, "
        f"Hidden=({best_chrom.fusion_hidden1}, {best_chrom.fusion_hidden2})"
    )

    # 3. Retrain Best Model with Full 5-Fold Cross Validation
    logger.info("\nRetraining Best Configuration with Full 5-Fold Cross-Validation...")
    full_eval_model = best_chrom.build_model()
    eval_trainer = AutismScreeningTrainer(
        model=full_eval_model,
        dataset=dataset,
        n_folds=final_folds,
        num_epochs=final_epochs,
        learning_rate=best_chrom.learning_rate,
        weight_decay=best_chrom.weight_decay,
        modality_weights=best_chrom.get_modality_weights(),
        random_seed=random_seed,
        verbose=False,
    )
    final_metrics = eval_trainer.run_cv()

    logger.info("\n--- Final 5-Fold Cross-Validation Performance ---")
    logger.info(f"  • Balanced Accuracy : {final_metrics['mean_balanced_accuracy']:.4f} ± {final_metrics['std_balanced_accuracy']:.4f}")
    logger.info(f"  • Macro F1-Score    : {final_metrics['mean_macro_f1']:.4f} ± {final_metrics['std_macro_f1']:.4f}")
    logger.info(f"  • ROC-AUC           : {final_metrics['mean_roc_auc']:.4f} ± {final_metrics['std_roc_auc']:.4f}")
    logger.info(f"  • PR-AUC            : {final_metrics['mean_pr_auc']:.4f} ± {final_metrics['std_pr_auc']:.4f}")
    logger.info(f"  • Sensitivity (ASD) : {final_metrics['mean_sensitivity']:.4f} ± {final_metrics['std_sensitivity']:.4f}")
    logger.info(f"  • Specificity (Ctrl): {final_metrics['mean_specificity']:.4f} ± {final_metrics['std_specificity']:.4f}")

    # Save evaluation summary
    eval_summary_file = checkpoint_dir / "final_evaluation_metrics.json"
    with open(eval_summary_file, "w", encoding="utf-8") as f:
        clean_summary = {k: v for k, v in final_metrics.items() if k != "best_fold_model_state_dict" and k != "per_fold_results"}
        clean_summary["best_chromosome"] = best_chrom.to_dict()
        json.dump(clean_summary, f, indent=2)
    logger.info(f"Saved evaluation metrics to {eval_summary_file}")

    # 4. Train Final Model on 100% of Dataset for Deployment
    logger.info("\nTraining Final Production Model on 100% Dataset...")
    deploy_model = best_chrom.build_model()
    deploy_trainer = AutismScreeningTrainer(
        model=deploy_model,
        dataset=dataset,
        learning_rate=best_chrom.learning_rate,
        weight_decay=best_chrom.weight_decay,
        modality_weights=best_chrom.get_modality_weights(),
        random_seed=random_seed,
    )
    final_trained_model, _ = deploy_trainer.train_full(num_epochs=final_epochs)

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": final_trained_model.state_dict(),
            "chromosome": best_chrom.to_dict(),
            "modality_weights": best_chrom.get_modality_weights(),
            "metrics": clean_summary,
        },
        final_model_path,
    )
    logger.info(f"Saved final trained model weights to {final_model_path}")

    # 5. Generate Evolution Plots
    logger.info("\nGenerating Evolution Visualizations...")
    plot_evolution_telemetry(log_dir / "evolution_log.csv", log_dir)
    paper_fig_dir = Path("paper/figures")
    if paper_fig_dir.exists():
        plot_evolution_telemetry(log_dir / "evolution_log.csv", paper_fig_dir)
        import shutil
        if (paper_fig_dir / "fitness_curve.png").exists():
            shutil.copy(paper_fig_dir / "fitness_curve.png", paper_fig_dir / "fig8_ga_convergence_curves.png")
    logger.info("Phase 3 GA Optimization Pipeline Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm Optimization for Multimodal Autism Late Fusion")
    parser.add_argument("--population", type=int, default=20, help="GA population size")
    parser.add_argument("--generations", type=int, default=30, help="Number of evolutionary generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--fitness_folds", type=int, default=3, help="CV folds during GA evaluation")
    parser.add_argument("--fitness_epochs", type=int, default=25, help="Epochs per fold during GA evaluation")
    parser.add_argument("--final_folds", type=int, default=5, help="Folds for final validation")
    parser.add_argument("--final_epochs", type=int, default=50, help="Epochs for final validation")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test with minimal settings")

    args = parser.parse_args()

    if args.smoke_test:
        run_pipeline(
            population_size=2,
            n_generations=1,
            fitness_folds=2,
            fitness_epochs=2,
            final_folds=2,
            final_epochs=2,
            random_seed=args.seed,
            resume=args.resume,
        )
    else:
        run_pipeline(
            population_size=args.population,
            n_generations=args.generations,
            fitness_folds=args.fitness_folds,
            fitness_epochs=args.fitness_epochs,
            final_folds=args.final_folds,
            final_epochs=args.final_epochs,
            random_seed=args.seed,
            resume=args.resume,
        )
