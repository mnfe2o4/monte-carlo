"""
Compare four estimators using mean estimates and RMSE from the same summary.

Run experiments.export_summary first to summarize the stored raw repetitions.
The exact oracle is retained in the summary and omitted from the log-RMSE plot.
"""

from pathlib import Path

import matplotlib

# This experiment saves figures without opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from src.plotting import plot_estimates, plot_rmse
from src.target import Z_TRUE, Z_TRUNCATED_4


def plot_comparison(
        csv_path: Path | str, output_path: Path | str
) -> tuple[Figure, np.ndarray]:
    """
    Compare stored estimators against the full-space truth.

    Args:
        csv_path: Directory containing the combined summary.csv file.
        output_path: Path for saving the mean estimate and empirical RMSE figure.

    Returns:
        fig, axes: Figure and its mean estimate and empirical RMSE axes.
            Every curve is derived from the same saved replicate estimates.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    summary = pd.read_csv(csv_path / "summary.csv", float_precision="round_trip")
    experiments = {
        "uniform_truncated_4": r"Uniform MC ($[-4, 4]^2$)",
        "gaussian": "Initial Gaussian IS",
        "refined_gaussian": "Refined Gaussian IS",
        # The current experiment matches the covariance, with mean (0, 0.5).
        "moment_matched_gaussian": "Moment-matched Gaussian IS",
    }
    missing = set(experiments) - set(summary["method"])
    if missing:
        raise ValueError(f"Summary is missing methods: {sorted(missing)}")
    if summary.duplicated(["method", "n_samples"]).any():
        raise ValueError("Summary contains duplicate method/sample-size pairs.")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    for name, label in experiments.items():
        result = summary.loc[summary["method"] == name].sort_values("n_samples")
        sample_sizes = result["n_samples"].tolist()
        estimates = result["mean_estimate"].to_numpy()
        rmses = result["rmse_vs_pi"].to_numpy()

        # Plot the replicate mean and RMSE without generating new samples.
        plot_estimates(
            sample_sizes, estimates, label=label,
            title="Mean Estimates vs Sample Size", ax=axes[0]
        )
        plot_rmse(
            sample_sizes, rmses, label=label,
            title="Empirical RMSE vs Sample Size", ax=axes[1], fit_slope=True
        )

    # Uniform approaches its box integral; every RMSE is measured against pi.
    axes[0].axhline(Z_TRUE, color="black", linestyle="--", label=r"Full integral $Z = \pi$")
    axes[0].axhline(Z_TRUNCATED_4, color="gray", linestyle="--", label=r"Box integral $Z_{[-4,4]^2}$")
    axes[1].axhline(
        abs(Z_TRUE - Z_TRUNCATED_4), color="gray", linestyle="--",
        label="Uniform truncation bias"
    )
    # Keep the larger RMSE legend below the axes so it does not hide the data.
    axes[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_path = PROJECT_ROOT / "figures" / "comparison.png"
    fig, axes = plot_comparison(PROJECT_ROOT / "results" / "summary", output_path)
    plt.close(fig)
    print(f"Figure saved to {output_path}")
