import numpy as np
from src.estimators import MCEstimator
from src.proposals import Proposal
from src.target import *
from src.experiments import run_experiment
from pathlib import Path

def generate_exact_samples(num_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate samples from the exact distribution for comparison.

    Args:
        num_samples: Number of samples to generate.
        rng: Optional NumPy generator. If None, create a new generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = rng.normal(size=num_samples)
    y = rng.normal(loc=(1 - x**2)/2, scale=0.5)
    return np.column_stack([x, y])

if __name__ == "__main__":
    exact_proposal = Proposal(
        pdf= lambda x: target_function(x) / np.pi,
        log_pdf= lambda x: log_target_function(x) - np.log(np.pi),
        sample_generator=generate_exact_samples
    )
    estimator = MCEstimator(exact_proposal)
    # A fixed generator makes the sequence of estimates reproducible.
    rng = np.random.default_rng(20260930)
    # Resolve outputs relative to this file, independently of the working directory.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    csv_path=PROJECT_ROOT / "results"
    figure_path=PROJECT_ROOT / "figures"
    # Compare against the full-space truth, including the box truncation error.
    run_experiment(
        estimator, target_function, sample_sizes=[100, 300, 1000, 3000, 10000, 30000, 100000], num_repetitions=100,
        true_value=Z_TRUE, use_log=False, rng=rng,
        csv_path=csv_path, csv_name="exact",
        figure_path=figure_path, figure_names=("exact_estimates", "exact_rmse"), figure_titles=("Exact Estimates vs N", "Exact RMSE vs N"),
        quiet=False, fit_slope=True, show_true_value=True
    )