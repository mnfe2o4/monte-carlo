"""
Compare uniform Monte Carlo on [-4, 4]^2 and [-6, 6]^2 using stored summaries.

Run experiments.export_summary after updating raw data, then run this module.
Both empirical RMSE curves use the full-space truth pi.
"""

from pathlib import Path

import matplotlib

# Generate the comparison without opening a GUI or drawing new samples.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from src.plotting import plot_estimates, plot_rmse
from src.target import Z_TRUE, Z_TRUNCATED_4, Z_TRUNCATED_6


def plot_uniform_comparison(
        csv_path: Path | str, output_path: Path | str
) -> tuple[Figure, np.ndarray]:
    """
    Compare the mean estimates and empirical RMSE for two truncation boxes.

    Args:
        csv_path: Directory containing summary.csv exported from raw repetitions.
        output_path: Path for saving the two-panel comparison figure.

    Returns:
        fig, axes: Figure and its mean estimate and empirical RMSE axes.
            Dashed references identify the full integral, both box integrals,
            and each box's absolute truncation bias. These bias levels are
            asymptotic RMSE floors, not predictions of finite-sample RMSE.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    summary = pd.read_csv(csv_path / 'summary.csv', float_precision='round_trip')
    experiments = {
        'uniform_truncated_4': (r'Uniform $[-4,4]^2$', Z_TRUNCATED_4),
        'uniform_truncated_6': (r'Uniform $[-6,6]^2$', Z_TRUNCATED_6),
    }
    missing = set(experiments) - set(summary['method'])
    if missing:
        raise ValueError(f'Summary is missing methods: {sorted(missing)}')
    selected = summary.loc[summary['method'].isin(experiments)]
    if selected.duplicated(['method', 'n_samples']).any():
        raise ValueError('Summary contains duplicate method/sample-size pairs.')

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for i, (name, (label, box_integral)) in enumerate(experiments.items()):
        result = selected.loc[selected['method'] == name].sort_values('n_samples')
        sample_sizes = result['n_samples'].tolist()

        # Both panels summarize the same stored repetitions for each box and N.
        plot_estimates(
            sample_sizes, result['mean_estimate'].to_numpy(), label=label,
            title='Mean Estimates vs Sample Size', ax=axes[0]
        )
        plot_rmse(
            sample_sizes, result['rmse_vs_pi'].to_numpy(), label=label,
            title=r'Empirical RMSE Relative to $\pi$', ax=axes[1], fit_slope=True
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
    fig.suptitle('Uniform Monte Carlo: Truncation Box Comparison')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, axes


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_path = PROJECT_ROOT / 'figures' / 'uniform_comparison.png'
    fig, axes = plot_uniform_comparison(PROJECT_ROOT / 'results' / 'summary', output_path)
    plt.close(fig)
    print(f'Figure saved to {output_path}')
