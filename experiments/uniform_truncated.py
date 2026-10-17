"""
Compare uniform box integration with the full-space reference integral.
"""

from src.estimators import *
from src.proposals import UniformProposal
from src.target import *
from src.plotting import *
from src.experiments import run_experiment
from pathlib import Path

if __name__ == "__main__":
    # Each tuple is one coordinate interval, not a pair of corner points.
    estimator4 = MCEstimator(UniformProposal([(-4, 4), (-4, 4)]))
    estimator6 = MCEstimator(UniformProposal([(-6, 6), (-6, 6)]))
    # A fixed generator makes the sequence of estimates reproducible.
    rng = np.random.default_rng(20260930)
    # Resolve outputs relative to this file, independently of the working directory.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    csv_path=PROJECT_ROOT / "results" / "raw"
    figure_path=PROJECT_ROOT / "figures"
    # Compare against the full-space truth, including the box truncation error.
    run_experiment(
        estimator4, target_function, sample_sizes=[100, 300, 1000, 3000, 10000, 30000, 100000], num_repetitions=100,
        true_value=Z_TRUE, use_log=False, rng=rng,
        csv_path=csv_path, csv_name="uniform_truncated_4",
        figure_path=figure_path, figure_names=("uniform_truncated_4_estimates", "uniform_truncated_4_rmse"), figure_titles=("Uniform Truncated on $[-4, 4]^2$ Estimates vs N", "Uniform Truncated on $[-4, 4]^2$ RMSE vs N"),
        quiet=False, fit_slope=True, show_true_value=True
    )