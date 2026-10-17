"""
Compare uniform Monte Carlo on [-4, 4]^2 and [-6, 6]^2 using stored summaries.

Run experiments.export_summary after updating raw data, then run this module.
Generate the original comparison, the extended-run comparison, and an RMSE
comparison of both runs for each box. All errors use the full-space truth pi.
"""

from pathlib import Path

import matplotlib

# Generate the comparisons without opening a GUI or drawing new samples.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from src.plotting import plot_estimates, plot_rmse
from src.target import Z_TRUE, Z_TRUNCATED_4, Z_TRUNCATED_6


def _load_uniform_summary(csv_path: Path | str, methods: list[str]) -> pd.DataFrame:
    """
    Load and validate the requested uniform methods from the stored summary.

    Args:
        csv_path: Directory containing summary.csv exported from raw repetitions.
        methods: Method names needed by the comparison.

    Returns:
        selected: Summary rows for the requested methods, without pooling runs.
    """
    summary = pd.read_csv(Path(csv_path) / 'summary.csv', float_precision='round_trip')
    missing = set(methods) - set(summary['method'])
    if missing:
        raise ValueError(f'Summary is missing methods: {sorted(missing)}')
    selected = summary.loc[summary['method'].isin(methods)]
    if selected.duplicated(['method', 'n_samples']).any():
        raise ValueError('Summary contains duplicate method/sample-size pairs.')
    return selected


def plot_uniform_comparison(
        csv_path: Path | str, output_path: Path | str,
        large_samples: bool = False
) -> tuple[Figure, np.ndarray]:
    """
    Compare the mean estimates and empirical RMSE for two truncation boxes.

    Args:
        csv_path: Directory containing summary.csv exported from raw repetitions.
        output_path: Path for saving the two-panel comparison figure.
        large_samples: If True, use the methods ending in _largesamples,
            including their smaller N values. Otherwise use the original runs.

    Returns:
        fig, axes: Figure and its mean estimate and empirical RMSE axes.
            Dashed references identify the full integral, both box integrals,
            and each box's absolute truncation bias. These bias levels are
            asymptotic RMSE floors, not predictions of finite-sample RMSE.
            Power-law fits describe the selected run across its stored N values.
    """
    output_path = Path(output_path)
    suffix = '_largesamples' if large_samples else ''
    experiments = {
        f'uniform_truncated_4{suffix}': (r'Uniform $[-4,4]^2$', Z_TRUNCATED_4),
        f'uniform_truncated_6{suffix}': (r'Uniform $[-6,6]^2$', Z_TRUNCATED_6),
    }
    selected = _load_uniform_summary(csv_path, list(experiments))

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for i, (name, (label, box_integral)) in enumerate(experiments.items()):
        result = selected.loc[selected['method'] == name].sort_values('n_samples')
        sample_sizes = result['n_samples'].tolist()
        repetitions = ', '.join(str(n) for n in sorted(result['n_replicates'].unique()))
        curve_label = f'{label} (R={repetitions})'

        # Both panels summarize the same stored repetitions for each box and N.
        plot_estimates(
            sample_sizes, result['mean_estimate'].to_numpy(), label=curve_label,
            title='Mean Estimates vs Sample Size', ax=axes[0]
        )
        plot_rmse(
            sample_sizes, result['rmse_vs_pi'].to_numpy(), label=curve_label,
            title=r'Empirical RMSE Relative to $\pi$', ax=axes[1],
            fit_slope=len(sample_sizes) > 1
        )

        # Match each box reference and bias floor to its empirical curve's color.
        axes[0].axhline(
            box_integral, color=f'C{i}', linestyle='--',
            label=f'{label}: box integral = {box_integral:.6f}'
        )
        bias = abs(Z_TRUE - box_integral)
        axes[1].axhline(
            bias, color=f'C{i}', linestyle='--',
            label=f'{label}: truncation bias = {bias:.6f}'
        )

    axes[0].axhline(Z_TRUE, color='black', linestyle='--', label=r'Full integral $Z=\pi$')
    # Put the legends below the panels so the references do not hide the results.
    axes[0].legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
    run_label = 'Extended Sample Sizes' if large_samples else 'Original Sample Sizes'
    fig.suptitle(f'Uniform Monte Carlo: Truncation Box Comparison ({run_label})')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_uniform_rmse_comparison(
        csv_path: Path | str, output_path: Path | str
) -> tuple[Figure, np.ndarray]:
    """
    Compare original and extended-run RMSE separately for each truncation box.

    Args:
        csv_path: Directory containing summary.csv with both runs for each box.
        output_path: Path for saving the two-panel RMSE comparison figure.

    Returns:
        fig, axes: Figure with bound 4 on the left and bound 6 on the right.
            Only the original small-sample run has a fitted power-law line,
            fitted and drawn over its own N values. Extended runs include all
            stored N values and remain separate at overlapping sample sizes.
            Legends show actual repetition counts; dashed horizontal lines
            identify the absolute truncation bias relative to pi.
    """
    output_path = Path(output_path)
    methods = [
        f'uniform_truncated_{bound}{suffix}'
        for bound in [4, 6] for suffix in ['', '_largesamples']
    ]
    selected = _load_uniform_summary(csv_path, methods)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, bound, box_integral in zip(axes, [4, 6], [Z_TRUNCATED_4, Z_TRUNCATED_6]):
        for suffix, label in [('', 'Small-sample run'), ('_largesamples', 'Extended run')]:
            name = f'uniform_truncated_{bound}{suffix}'
            result = selected.loc[selected['method'] == name].sort_values('n_samples')
            sample_sizes = result['n_samples'].tolist()
            repetitions = ', '.join(str(n) for n in sorted(result['n_replicates'].unique()))

            # Fit only the original run; do not pool independent replicate sets.
            plot_rmse(
                sample_sizes, result['rmse_vs_pi'].to_numpy(),
                label=f'{label} (R={repetitions})',
                title=rf'Uniform $[-{bound},{bound}]^2$: RMSE Relative to $\pi$',
                ax=ax, fit_slope=not suffix and len(sample_sizes) > 1
            )

        bias = abs(Z_TRUE - box_integral)
        ax.axhline(
            bias, color='gray', linestyle='--',
            label=f'Truncation bias = {bias:.6f}'
        )
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)

    fig.suptitle('Uniform Monte Carlo: Small-Sample and Extended Runs')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, axes


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    csv_path = PROJECT_ROOT / 'results' / 'summary'
    figure_path = PROJECT_ROOT / 'figures'
    for large_samples, filename in [
        (False, 'uniform_comparison.png'),
        (True, 'uniform_comparison_largesamples.png'),
    ]:
        output_path = figure_path / filename
        fig, axes = plot_uniform_comparison(csv_path, output_path, large_samples=large_samples)
        plt.close(fig)
        print(f'Figure saved to {output_path}')

    output_path = figure_path / 'uniform_rmse_comparison.png'
    fig, axes = plot_uniform_rmse_comparison(csv_path, output_path)
    plt.close(fig)
    print(f'Figure saved to {output_path}')
