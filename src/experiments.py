"""
Run sample-size sweeps, save summary results and optionally plot them.
"""

from .estimators import *
from .plotting import *
from pathlib import Path
from collections.abc import Callable
import numpy as np
import csv

def run_experiment(
        estimator: MCEstimator, func: Callable[[np.ndarray], float], sample_sizes: list[int], num_repetitions: int, true_value: float,
        use_log: bool = False, rng: np.random.Generator | None = None,
        csv_path: Path | str | None = None, csv_name: str = "results",
        figure_path: Path | str | None = None, figure_names: tuple[str, str] = ("MC Estimates vs N", "RMSE vs N"), figure_titles: tuple[str, str] | None = None,
        quiet: bool = False, fit_slope: bool = False, show_true_value: bool = False
):
    """
    Sweep sample sizes and optionally save estimates, empirical RMSE and plots.

    For each sample size, generate num_repetitions estimates once and save
    every estimate with its replicate index. Empirical RMSE uses the same
    repetitions; the optional estimate plot shows their first estimate.
    Total sampling uses num_repetitions * N for each sample size.

    Args:
        estimator: Monte Carlo estimator used for every configuration.
        func: Single-point integrand, or its natural logarithm when use_log=True.
        sample_sizes: Positive sample counts to evaluate.
        num_repetitions: Positive number of repeated estimates for each RMSE.
        true_value: Integral truth used for error evaluation and, optionally,
            the estimate plot. A full-space truth includes truncation error
            when the proposal has bounded support.
        use_log: If True, interpret func as the log-integrand.
        rng: Optional NumPy generator advanced through the experiment.
            Supply a seeded generator to reproduce the results.
        csv_path: Optional output directory for the raw replicate CSV.
        csv_name: CSV filename stem, without the .csv extension.
        figure_path: Optional output directory for the two figures.
        figure_names: Estimate and RMSE figure titles, also used as filename stems.
        figure_titles: Optional custom titles for the figures.
        quiet: If True, suppress progress and output-path messages.
        fit_slope: If True, add a dashed empirical power-law fit to the RMSE plot.
        show_true_value: If True, add the supplied truth as a dashed line on
            the estimate plot. Both plotting options apply when figure_path
            is provided.

    Returns:
        None. Raw estimates and figures are saved according to the options.
    """
    rmses = dict()
    results = dict()

    # Reuse one batch of repetitions for the raw output and empirical RMSE.
    for num_samples in sample_sizes:
        result = estimator.repeated_estimates(func, num_samples, num_repetitions, use_log, quiet, rng)
        rmses[num_samples] = rmse(result, true_value)
        results[num_samples] = result

        if not quiet:
            print(f"Sample Size={num_samples}, Average Estimate={result.mean()}, RMSE={rmses[num_samples]}")

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.mkdir(parents=True, exist_ok=True)

        with csv_path.joinpath(f"{csv_name}.csv").open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Sample Size", "Replicate", "Estimate"])
            for num_samples in sample_sizes:
                result = results[num_samples]
                for replicate, estimate in enumerate(result):
                    writer.writerow([num_samples, replicate + 1, estimate])

        if not quiet:
            print(f"Results saved to {csv_path.joinpath(f'{csv_name}.csv')}")

    if figure_path:
        figure_path = Path(figure_path)
        figure_path.mkdir(parents=True, exist_ok=True)

        plot_estimates(
            sample_sizes, np.array([results[num_samples][0] for num_samples in sample_sizes]),
            title=figure_titles[0] if figure_titles else figure_names[0], output_path=figure_path.joinpath(f"{figure_names[0]}.png"),
            true_value=true_value if show_true_value else None
        )
        plot_rmse(
            sample_sizes, np.array([rmses[num_samples] for num_samples in sample_sizes]),
            title=figure_titles[1] if figure_titles else figure_names[1], output_path=figure_path.joinpath(f"{figure_names[1]}.png"),
            fit_slope=fit_slope
        )

        if not quiet:
            print(f"Figures saved to {figure_path.joinpath(f'{figure_names[0]}.png')} and {figure_path.joinpath(f'{figure_names[1]}.png')}")
