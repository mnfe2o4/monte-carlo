"""
Generate target and importance-weight figures for the report without sampling.

Run from the repository root with python -m experiments.visualize_target.
Proposal parameters match the current Gaussian experiment entry points.
"""

from pathlib import Path

import matplotlib

# Save figures without opening a GUI window or rerunning Monte Carlo experiments.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from src.plotting import plot_heatmap, plot_3d_surface, plot_weight_heatmaps
from src.proposals import GaussianProposal
from src.target import target_function, log_target_function


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    figure_path = PROJECT_ROOT / 'figures'
    xlim = (-4, 4)
    ylim = (-4, 4)

    # Both views evaluate the same unnormalized target on a deterministic grid.
    fig, ax = plot_heatmap(
        target_function, title=r'Target Density $\widetilde\rho(x)$',
        xlim=xlim, ylim=ylim, colorbar_label=r'$\widetilde\rho(x)$',
        output_path=figure_path / 'target_heatmap.png'
    )
    plt.close(fig)
    fig, ax = plot_3d_surface(
        target_function, title=r'Target Surface $\widetilde\rho(x)$',
        xlim=xlim, ylim=ylim, zlabel=r'$\widetilde\rho(x)$',
        output_path=figure_path / 'target_surface.png'
    )
    plt.close(fig)

    # These are fixed evaluation proposals, not the historical PDF parameters.
    proposals = {
        'gaussian': ('Initial Gaussian', GaussianProposal(
            np.array([0.0, 0.5]), np.diag([1.0, 0.25])
        )),
        'refined_gaussian': ('Refined Gaussian', GaussianProposal(
            np.array([0.0, 0.0]), np.diag([2.46006384, 1.40547353])
        )),
        'moment_matched_gaussian': ('Moment-matched Gaussian', GaussianProposal(
            np.array([0.0, 0.0]), np.diag([1.0, 0.75])
        )),
    }
    proposal_log_pdfs = {label: proposal.log_pdf for label, proposal in proposals.values()}
    fig, axes = plot_weight_heatmaps(
        log_target_function, proposal_log_pdfs, xlim=xlim, ylim=ylim,
        output_path=figure_path / 'weights_heatmap_comparison.png'
    )
    # Reuse the comparison limits so separate images also have comparable colors.
    norm = axes[0].collections[0].norm
    vmin, vmax = norm.vmin, norm.vmax
    plt.close(fig)

    for name, (label, proposal) in proposals.items():
        fig, axes = plot_weight_heatmaps(
            log_target_function, {label: proposal.log_pdf}, xlim=xlim, ylim=ylim,
            vmin=vmin, vmax=vmax,
            output_path=figure_path / f'{name}_weights_heatmap.png'
        )
        plt.close(fig)

    print(f'Target and weight figures saved to {figure_path}')
