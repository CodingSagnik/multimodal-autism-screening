"""
src/optimization/genetic_algorithm.py

Genetic Algorithm Engine for Multimodal Autism Late Fusion Optimization (Phase 3).
Features:
- Latin Hypercube Sampling (LHS) for population initialization.
- Simulated Binary Crossover (SBX) and Polynomial Mutation for real-coded chromosomes.
- Tournament selection with strict elitism preserving top candidate solutions.
- Checkpointing, resumption, and CSV evolution telemetry tracking.
"""

import copy
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.optimization.ga_chromosome import (
    GAChromosome,
    evaluate_chromosome_fitness,
)

logger = logging.getLogger(__name__)


def latin_hypercube_sampling(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates samples in [0.0, 1.0]^n_dim using Latin Hypercube Sampling.
    """
    result = np.zeros((n_samples, n_dim), dtype=np.float64)
    for j in range(n_dim):
        intervals = (np.arange(n_samples) + rng.uniform(0.0, 1.0, size=n_samples)) / n_samples
        rng.shuffle(intervals)
        result[:, j] = intervals
    return result


class GeneticAlgorithmEngine:
    """
    Evolutionary optimizer finding optimal modality weights and hyperparameters
    for MultimodalAutismClassifier.
    """

    def __init__(
        self,
        dataset: MultimodalAutismDataset,
        population_size: int = 30,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        tournament_size: int = 3,
        elitism_count: int = 2,
        fitness_folds: int = 3,
        fitness_epochs: int = 25,
        early_stopping_patience: int = 6,
        batch_size: int = 16,
        random_seed: int = 42,
        checkpoint_dir: Union[str, Path] = Path("models/ga_checkpoints"),
        log_dir: Union[str, Path] = Path("models/ga_logs"),
        device: str = "auto",
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.tournament_size = tournament_size
        self.elitism_count = max(1, elitism_count)
        self.fitness_folds = fitness_folds
        self.fitness_epochs = fitness_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device
        self.verbose = verbose

        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(random_seed)
        self.population: List[GAChromosome] = []
        self.best_chromosome: Optional[GAChromosome] = None
        self.history: List[Dict[str, Any]] = []

    def initialize_population(self) -> List[GAChromosome]:
        """
        Initializes the population via Latin Hypercube Sampling.
        Injects a default baseline chromosome (weights=1.0, default lr/wd) as individual 0.
        """
        samples = latin_hypercube_sampling(self.population_size, 9, self.rng)
        pop: List[GAChromosome] = []

        # Individual 0: Standard canonical default
        canonical = GAChromosome()
        pop.append(canonical)

        # Remaining individuals from LHS
        for i in range(1, self.population_size):
            chrom = GAChromosome.from_gene_vector(samples[i])
            pop.append(chrom)

        self.population = pop
        return pop

    def evaluate_population(self, population: List[GAChromosome]) -> None:
        """
        Evaluates fitness for any unevaluated individuals in the population.
        """
        for i, chrom in enumerate(population):
            if chrom.fitness is None:
                chrom_seed = self.random_seed + i * 100
                evaluate_chromosome_fitness(
                    chromosome=chrom,
                    dataset=self.dataset,
                    n_folds=self.fitness_folds,
                    num_epochs=self.fitness_epochs,
                    batch_size=self.batch_size,
                    random_seed=chrom_seed,
                    device=self.device,
                    early_stopping_patience=self.early_stopping_patience,
                    verbose=False,
                )

        population.sort(key=lambda c: (c.fitness if c.fitness is not None else -1.0), reverse=True)

    def tournament_select(self, population: List[GAChromosome]) -> GAChromosome:
        """
        Selects a single parent chromosome via tournament selection.
        """
        candidates = self.rng.choice(population, size=self.tournament_size, replace=True)
        best = max(candidates, key=lambda c: (c.fitness if c.fitness is not None else -1.0))
        return copy.deepcopy(best)

    def sbx_crossover(
        self, parent1: GAChromosome, parent2: GAChromosome
    ) -> Tuple[GAChromosome, GAChromosome]:
        """
        Simulated Binary Crossover (SBX) with distribution index eta_c.
        """
        if self.rng.uniform() > self.crossover_prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        g1 = parent1.to_gene_vector()
        g2 = parent2.to_gene_vector()
        c1 = np.zeros_like(g1)
        c2 = np.zeros_like(g2)

        for i in range(len(g1)):
            if self.rng.uniform() <= 0.5:
                if abs(g1[i] - g2[i]) > 1e-14:
                    y1 = min(g1[i], g2[i])
                    y2 = max(g1[i], g2[i])

                    rand = self.rng.uniform()
                    beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                    alpha = 2.0 - (beta ** -(self.eta_c + 1.0))
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
                    c1_val = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                    beta = 1.0 + (2.0 * (1.0 - y2) / (y2 - y1))
                    alpha = 2.0 - (beta ** -(self.eta_c + 1.0))
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
                    c2_val = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                    c1[i] = np.clip(c1_val, 0.0, 1.0)
                    c2[i] = np.clip(c2_val, 0.0, 1.0)
                else:
                    c1[i] = g1[i]
                    c2[i] = g2[i]
            else:
                c1[i] = g1[i]
                c2[i] = g2[i]

        child1 = GAChromosome.from_gene_vector(c1)
        child2 = GAChromosome.from_gene_vector(c2)
        return child1, child2

    def polynomial_mutation(self, chromosome: GAChromosome) -> GAChromosome:
        """
        Polynomial mutation with distribution index eta_m for continuous genes
        and uniform reset for discrete genes.
        """
        g = chromosome.to_gene_vector()
        for i in range(len(g)):
            if self.rng.uniform() <= self.mutation_prob:
                if i >= 7:  # Discrete architecture genes
                    g[i] = self.rng.uniform(0.0, 1.0)
                else:  # Continuous genes
                    y = g[i]
                    delta1 = y
                    delta2 = 1.0 - y
                    rand = self.rng.uniform()
                    mut_pow = 1.0 / (self.eta_m + 1.0)

                    if rand <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1.0))
                        delta_q = (val ** mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1.0))
                        delta_q = 1.0 - (val ** mut_pow)

                    g[i] = np.clip(y + delta_q, 0.0, 1.0)

        return GAChromosome.from_gene_vector(g)

    def log_generation(self, generation: int) -> Dict[str, Any]:
        """
        Records generation statistics and updates CSV log file.
        """
        fitnesses = [c.fitness for c in self.population if c.fitness is not None]
        best_chrom = self.population[0]
        mean_fit = float(np.mean(fitnesses)) if fitnesses else 0.0
        best_fit = float(np.max(fitnesses)) if fitnesses else 0.0
        worst_fit = float(np.min(fitnesses)) if fitnesses else 0.0
        std_fit = float(np.std(fitnesses)) if fitnesses else 0.0

        # Measure gene diversity
        gene_matrix = np.array([c.to_gene_vector() for c in self.population])
        diversity = float(np.mean(np.std(gene_matrix, axis=0)))

        record = {
            "generation": generation,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "worst_fitness": worst_fit,
            "std_fitness": std_fit,
            "gene_diversity": diversity,
            "w_vision": best_chrom.w_vision,
            "w_audio": best_chrom.w_audio,
            "w_text": best_chrom.w_text,
            "learning_rate": best_chrom.learning_rate,
            "weight_decay": best_chrom.weight_decay,
            "dropout1": best_chrom.dropout1,
            "dropout2": best_chrom.dropout2,
            "fusion_hidden1": best_chrom.fusion_hidden1,
            "fusion_hidden2": best_chrom.fusion_hidden2,
        }
        self.history.append(record)

        # Append to CSV
        csv_path = self.log_dir / "evolution_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        if self.verbose:
            logger.info(
                f"[GA Gen {generation:02d}/{self.n_generations:02d}] "
                f"Best Fit: {best_fit:.4f} | Mean: {mean_fit:.4f} | "
                f"Weights: (V={best_chrom.w_vision:.2f}, A={best_chrom.w_audio:.2f}, T={best_chrom.w_text:.2f}) | "
                f"Div: {diversity:.3f}"
            )

        return record

    def save_checkpoint(self, generation: int) -> None:
        """
        Saves full population and best chromosome to checkpoint files.
        """
        ckpt_data = {
            "generation": generation,
            "n_generations": self.n_generations,
            "population": [c.to_dict() for c in self.population],
            "best_chromosome": self.population[0].to_dict() if self.population else None,
            "random_seed": self.random_seed,
        }

        # Per-generation checkpoint
        gen_file = self.checkpoint_dir / f"generation_{generation:03d}.json"
        with open(gen_file, "w", encoding="utf-8") as f:
            json.dump(ckpt_data, f, indent=2)

        # Global best chromosome file
        best_file = self.checkpoint_dir / "best_chromosome.json"
        if self.population:
            with open(best_file, "w", encoding="utf-8") as f:
                json.dump(self.population[0].to_dict(), f, indent=2)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Loads population and resumes from a saved checkpoint file.
        Returns the generation number loaded.
        """
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.population = [GAChromosome.from_dict(d) for d in data["population"]]
        if data.get("best_chromosome"):
            self.best_chromosome = GAChromosome.from_dict(data["best_chromosome"])
        generation = int(data.get("generation", 0))
        logger.info(f"Loaded GA checkpoint from {checkpoint_path} at generation {generation}.")
        return generation

    def evolve(self, resume_checkpoint: Optional[Union[str, Path]] = None) -> GAChromosome:
        """
        Executes the main evolutionary optimization loop.
        """
        start_gen = 1
        if resume_checkpoint:
            start_gen = self.load_checkpoint(resume_checkpoint) + 1
        else:
            csv_path = self.log_dir / "evolution_log.csv"
            if csv_path.exists():
                csv_path.unlink()
            self.initialize_population()

        # Evaluate Generation 0/initial
        self.evaluate_population(self.population)
        self.best_chromosome = copy.deepcopy(self.population[0])
        self.log_generation(generation=start_gen - 1)
        self.save_checkpoint(generation=start_gen - 1)

        # Generational loop
        for gen in range(start_gen, self.n_generations + 1):
            next_generation: List[GAChromosome] = []

            # 1. Elitism: preserve top candidates
            for elite_idx in range(self.elitism_count):
                elite_copy = copy.deepcopy(self.population[elite_idx])
                next_generation.append(elite_copy)

            # 2. Reproduction loop
            while len(next_generation) < self.population_size:
                p1 = self.tournament_select(self.population)
                p2 = self.tournament_select(self.population)

                c1, c2 = self.sbx_crossover(p1, p2)
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)

                c1.fitness = None
                c2.fitness = None

                next_generation.append(c1)
                if len(next_generation) < self.population_size:
                    next_generation.append(c2)

            # 3. Evaluate new generation
            self.evaluate_population(next_generation)
            self.population = next_generation

            # Update best individual
            if self.population[0].fitness is not None:
                if (
                    self.best_chromosome is None
                    or self.best_chromosome.fitness is None
                    or self.population[0].fitness > self.best_chromosome.fitness
                ):
                    self.best_chromosome = copy.deepcopy(self.population[0])

            # 4. Telemetry and Checkpoint
            self.log_generation(generation=gen)
            if gen % 5 == 0 or gen == self.n_generations:
                self.save_checkpoint(generation=gen)

        return self.best_chromosome if self.best_chromosome is not None else self.population[0]
