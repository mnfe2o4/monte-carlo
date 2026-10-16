"""
Compare refined gaussian integration with the reference integral.
"""

from src.estimators import *
from src.proposals import GaussianProposal
from src.target import *
from src.plotting import *
from src.experiments import run_experiment
from pathlib import Path

if __name__ == "__main__":
    estimator = MCEstimator(GaussianProposal(np.array([0.0, 0.0]), np.array([[0.99914662, 0.0], [0.0, 4.61945853]])))
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
        csv_path=csv_path, csv_name="refined_gaussian",
        figure_path=figure_path, figure_names=("refined_gaussian_estimates", "refined_gaussian_rmse"), figure_titles=("Refined Gaussian Estimates vs N", "Refined Gaussian RMSE vs N"),
        quiet=False, fit_slope=True, show_true_value=True
    )