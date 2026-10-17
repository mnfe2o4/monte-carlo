"""
Compare selected estimators using mean estimates and RMSE from one summary.

Run experiments.export_summary first to summarize the stored raw repetitions.
Edit methods in the module entry point to select curves and their order.
The exact oracle can appear in the estimate panel but is omitted from log-RMSE.
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
from src.target import Z_TRUE, Z_TRUNCATED_4, Z_TRUNCATED_6


def plot_comparison(
        csv_path: Path | str, output_path: Path | str,
        methods: list[str] | None = None
) -> tuple[Figure, np.ndarray]:
    """
    Compare selected stored estimators against the full-space truth.

    Args:
        csv_path: Directory containing the combined summary.csv file.
        output_path: Path for saving the mean estimate and empirical RMSE figure.
        methods: Method names from the summary, in plotting order. If None,
            use the original four non-oracle methods. Other methods use their
            summary names as labels when no display label is defined below.
            The exact oracle appears only in the mean estimate panel because
            its RMSE includes zeros that cannot be plotted on a log scale.

    Returns:
        fig, axes: Figure and its mean estimate and empirical RMSE axes.
            Every curve is derived from the same saved replicate estimates.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    summary = pd.read_csv(csv_path / "summary.csv", float_precision="round_trip")
    labels = {
        "uniform_truncated_4": r"Uniform MC ($[-4, 4]^2$)",
        "uniform_truncated_6": r"Uniform MC ($[-6, 6]^2$)",
        "uniform_truncated_4_largesamples": r"Uniform MC ($[-4, 4]^2$, extended run)",
        "uniform_truncated_6_largesamples": r"Uniform MC ($[-6, 6]^2$, extended run)",
        "gaussian": "Initial Gaussian IS",
        "refined_gaussian": "Refined Gaussian IS",
        "moment_matched_gaussian": "Moment-matched Gaussian IS",
        "exact": "Exact oracle",
    }
    if methods is None:
        methods = [
            "uniform_truncated_4", "gaussian", "refined_gaussian",
            "moment_matched_gaussian",
        ]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Select at least one method without repeating names.")
    missing = set(methods) - set(summary["method"])
    if missing:
        raise ValueError(f"Summary is missing methods: {sorted(missing)}")
    selected = summary.loc[summary["method"].isin(methods)]
    if selected.duplicated(["method", "n_samples"]).any():
        raise ValueError("Summary contains duplicate method/sample-size pairs.")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    for name in methods:
        label = labels.get(name, name)
        result = selected.loc[selected["method"] == name].sort_values("n_samples")
        sample_sizes = result["n_samples"].tolist()
        estimates = result["mean_estimate"].to_numpy()
        rmses = result["rmse_vs_pi"].to_numpy()

        # Plot only selected methods without generating new samples.
        plot_estimates(
            sample_sizes, estimates, label=label,
            title="Mean Estimates vs Sample Size", ax=axes[0]
        )
        if name != "exact":
            plot_rmse(
                sample_sizes, rmses, label=label,
                title="Empirical RMSE vs Sample Size", ax=axes[1],
                fit_slope=len(sample_sizes) > 1
            )

    axes[0].axhline(Z_TRUE, color="black", linestyle="--", label=r"Full integral $Z = \pi$")
    # Show box references only when a selected curve uses that truncation box.
    for bound, box_integral, color in [
        (4, Z_TRUNCATED_4, "gray"), (6, Z_TRUNCATED_6, "tab:purple"),
    ]:
        names = {f"uniform_truncated_{bound}", f"uniform_truncated_{bound}_largesamples"}
        if names.intersection(methods):
            axes[0].axhline(
                box_integral, color=color, linestyle="--",
                label=rf"Box integral $Z_{{[-{bound},{bound}]^2}}$"
            )
            axes[1].axhline(
                abs(Z_TRUE - box_integral), color=color, linestyle="--",
                label=rf"Uniform $[-{bound},{bound}]^2$ truncation bias"
            )
    # Keep the larger RMSE legend below the axes so it does not hide the data.
    axes[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    if any(name != "exact" for name in methods):
        axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    # Use method names from summary.csv; this list also controls the curve order.
    methods = [
        "uniform_truncated_4",
        "gaussian",
        "refined_gaussian",
        "moment_matched_gaussian",
    ]
    output_path = PROJECT_ROOT / "figures" / "comparison.png"
    fig, axes = plot_comparison(
        PROJECT_ROOT / "results" / "summary", output_path, methods=methods
    )
    plt.close(fig)
    print(f"Figure saved to {output_path}")
