"""
Compare stored Uniform, initial Gaussian and refined Gaussian results.

Read existing summary CSV files without rerunning the sampling experiments.
The estimate curves show single estimates; the RMSE curves summarize repeats.
"""

import csv
from pathlib import Path

import matplotlib

# This experiment saves figures without opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from src.plotting import plot_estimates, plot_rmse
from src.target import Z_TRUE, Z_TRUNCATED_4


def plot_comparison(
        csv_path: Path | str, output_path: Path | str
) -> tuple[Figure, np.ndarray]:
    """
    Plot the three stored estimators together using the full-space truth.

    Args:
        csv_path: Directory containing the three experiment summary CSV files.
        output_path: Path for saving the combined estimate and RMSE figure.

    Returns:
        fig, axes: Figure and its estimate and empirical RMSE axes.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    experiments = {
        "uniform_truncated_4": r"Uniform MC ($[-4, 4]^2$)",
        "gaussian": "Initial Gaussian IS",
        "refined_gaussian": "Refined Gaussian IS",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, label in experiments.items():
        with csv_path.joinpath(f"{name}.csv").open(newline="", encoding="utf-8-sig") as csvfile:
            rows = sorted(csv.DictReader(csvfile), key=lambda row: int(row["Sample Size"]))

        sample_sizes = [int(row["Sample Size"]) for row in rows]
        estimates = np.array([float(row["Estimate"]) for row in rows])
        rmses = np.array([float(row["RMSE"]) for row in rows])

        # Reuse the same axes for every method, with no additional sampling.
        plot_estimates(
            sample_sizes, estimates, label=label,
            title="Single Estimates vs Sample Size", ax=axes[0]
        )
        plot_rmse(
            sample_sizes, rmses, label=label,
            title="Empirical RMSE vs Sample Size", ax=axes[1], fit_slope=True
        )

    # Uniform MC approaches the box integral; all RMSEs are evaluated against pi.
    axes[0].axhline(Z_TRUE, color="black", linestyle="--", label=r"Full integral $Z = \pi$")
    axes[0].axhline(Z_TRUNCATED_4, color="gray", linestyle="--", label=r"Box integral $Z_{[-4,4]^2}$")
    axes[1].axhline(
        abs(Z_TRUE - Z_TRUNCATED_4), color="gray", linestyle="--",
        label="Uniform truncation bias"
    )
    for ax in axes:
        ax.legend(fontsize=9)
    axes[1].legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.08))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_path = PROJECT_ROOT / "figures" / "comparison.png"
    fig, axes = plot_comparison(PROJECT_ROOT / "results", output_path)
    plt.close(fig)
    print(f"Figure saved to {output_path}")
