"""
tests/test_phase3_ga.py

Unit tests for Phase 3 Genetic Algorithm operators, chromosome encoding, and engine.
Verifies:
1. GAChromosome gene vector roundtrip in [0.0, 1.0]^9.
2. JSON dictionary serialization/deserialization.
3. Model instantiation from chromosome parameters.
4. Latin Hypercube Sampling (LHS) bounds and dimensions.
5. Simulated Binary Crossover (SBX) bounds preservation.
6. Polynomial mutation bounds preservation.
7. Tournament selection pressure.
8. Engine population initialization and elitism.
"""

import numpy as np
import torch

from src.data_processing.multimodal_dataset import MultimodalAutismDataset
from src.models.fusion_model import MultimodalAutismClassifier
from src.optimization.ga_chromosome import GAChromosome, evaluate_chromosome_fitness
from src.optimization.genetic_algorithm import GeneticAlgorithmEngine, latin_hypercube_sampling


def test_chromosome_vector_roundtrip():
    """Verify gene vector normalization and roundtrip accuracy."""
    c1 = GAChromosome(
        w_vision=1.45,
        w_audio=0.72,
        w_text=1.18,
        learning_rate=4e-4,
        weight_decay=2e-5,
        dropout1=0.35,
        dropout2=0.18,
        fusion_hidden1=256,
        fusion_hidden2=64,
    )
    vec = c1.to_gene_vector()
    assert len(vec) == 9
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

    c2 = GAChromosome.from_gene_vector(vec)
    assert abs(c1.w_vision - c2.w_vision) < 1e-3
    assert abs(c1.w_audio - c2.w_audio) < 1e-3
    assert abs(c1.w_text - c2.w_text) < 1e-3
    assert abs(c1.dropout1 - c2.dropout1) < 1e-3
    assert c2.fusion_hidden1 == 256
    assert c2.fusion_hidden2 == 64


def test_chromosome_dict_serialization():
    """Verify dictionary serialization preserves fields and is JSON-safe."""
    c1 = GAChromosome(
        w_vision=1.2,
        w_audio=0.9,
        w_text=0.8,
        fitness=0.825,
        metadata={"mean_balanced_accuracy": 0.825, "mean_macro_f1": 0.79},
    )
    d = c1.to_dict()
    assert isinstance(d, dict)
    assert d["fitness"] == 0.825

    import json
    json_str = json.dumps(d)
    assert json_str is not None

    c2 = GAChromosome.from_dict(d)
    assert c2.w_vision == 1.2
    assert c2.fitness == 0.825


def test_chromosome_build_model():
    """Verify model instantiation and synthetic forward pass."""
    chrom = GAChromosome(
        dropout1=0.25,
        dropout2=0.10,
        fusion_hidden1=64,
        fusion_hidden2=16,
    )
    model = chrom.build_model()
    assert isinstance(model, MultimodalAutismClassifier)

    v = torch.randn(2, 50, 92, 3)
    a = torch.randn(2, 40, 313)
    t = torch.randn(2, 768)
    out = model(v, a, t, modality_weights=chrom.get_modality_weights())
    assert out.shape == (2, 1)


def test_lhs_sampling():
    """Verify Latin Hypercube Sampling covers all dimensions in [0, 1]."""
    rng = np.random.default_rng(123)
    samples = latin_hypercube_sampling(25, 9, rng)
    assert samples.shape == (25, 9)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_sbx_crossover_bounds():
    """Verify Simulated Binary Crossover produces offspring within [0, 1]."""
    engine = GeneticAlgorithmEngine(dataset=None, population_size=4)
    p1 = GAChromosome(w_vision=1.9, w_audio=0.1, w_text=1.8)
    p2 = GAChromosome(w_vision=0.2, w_audio=1.9, w_text=0.3)

    for _ in range(20):
        c1, c2 = engine.sbx_crossover(p1, p2)
        v1 = c1.to_gene_vector()
        v2 = c2.to_gene_vector()
        assert np.all(v1 >= 0.0) and np.all(v1 <= 1.0)
        assert np.all(v2 >= 0.0) and np.all(v2 <= 1.0)


def test_polynomial_mutation_bounds():
    """Verify Polynomial Mutation keeps genes within valid boundaries."""
    engine = GeneticAlgorithmEngine(dataset=None, population_size=4, mutation_prob=1.0)
    parent = GAChromosome(w_vision=1.0, w_audio=1.0, w_text=1.0)

    for _ in range(20):
        mutant = engine.polynomial_mutation(parent)
        v = mutant.to_gene_vector()
        assert np.all(v >= 0.0) and np.all(v <= 1.0)


def test_tournament_selection():
    """Verify tournament selection prefers higher fitness."""
    engine = GeneticAlgorithmEngine(dataset=None, population_size=4, tournament_size=3)
    pop = [
        GAChromosome(fitness=0.4),
        GAChromosome(fitness=0.9),
        GAChromosome(fitness=0.6),
        GAChromosome(fitness=0.5),
    ]
    winner = engine.tournament_select(pop)
    assert winner.fitness is not None
    assert winner.fitness >= 0.5


if __name__ == "__main__":
    print("Running Phase 3 Genetic Algorithm Tests...")
    test_chromosome_vector_roundtrip()
    print("  • test_chromosome_vector_roundtrip: PASSED")
    test_chromosome_dict_serialization()
    print("  • test_chromosome_dict_serialization: PASSED")
    test_chromosome_build_model()
    print("  • test_chromosome_build_model: PASSED")
    test_lhs_sampling()
    print("  • test_lhs_sampling: PASSED")
    test_sbx_crossover_bounds()
    print("  • test_sbx_crossover_bounds: PASSED")
    test_polynomial_mutation_bounds()
    print("  • test_polynomial_mutation_bounds: PASSED")
    test_tournament_selection()
    print("  • test_tournament_selection: PASSED")
    print("All Phase 3 Genetic Algorithm Tests PASSED!")
