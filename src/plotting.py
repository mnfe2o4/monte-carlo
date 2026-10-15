import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_estimator(
        samples: np.ndarray, target: Callable[[np.ndarray], float],
        title: str | None = None, xlim: tuple[float, float] = (-10, 10),
        ylim: tuple[float, float] = (-10, 10),
        proposal_pdf: Callable[[np.ndarray], float] | None = None,
        ax: Axes | None = None, output_path: str | Path | None = None
) -> tuple[Figure, Axes]:
    """
    Plots the target function along with previously generated samples.

    Args:
        samples: Array of samples with shape (num_samples, 2).
        target: Function accepting one sample of shape (2,) and returning
            the unnormalized target density, not its logarithm.
        title: Optional title for the plot.
        xlim: Lower and upper limits for the first coordinate.
        ylim: Lower and upper limits for the second coordinate.
        proposal_pdf: Optional proposal density to overlay as contour lines.
            Accepts one sample of shape (2,) and returns a scalar density.
        ax: Optional existing axes. If None, create a new figure and axes.
        output_path: Optional path for saving the figure as an image.

    Returns:
        fig, ax: Figure and axes for further customization or display.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    x = np.linspace(*xlim, 400)
    y = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([target(point) for point in points]).reshape(X.shape)

    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Target Function Value')
    if proposal_pdf is not None:
        Q = np.array([proposal_pdf(point) for point in points]).reshape(X.shape)
        ax.contour(X, Y, Q, levels=6, colors='orange', linestyles='--')
        ax.plot([], [], color='orange', linestyle='--', label='Proposal Contours')
    ax.scatter(samples[:, 0], samples[:, 1], color='red', s=12, alpha=0.5, label='Samples')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    if title:
        ax.set_title(title)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_estimates(
        sample_sizes: list[int], estimates: np.ndarray,
        label: str | None = None, title: str | None = None,
        reference_values: dict[str, float] | None = None,
        ax: Axes | None = None, output_path: str | Path | None = None
) -> tuple[Figure, Axes]:
    """
    Plots previously computed integral estimates against sample size.

    Args:
        sample_sizes: Number of samples used for each estimate.
        estimates: One estimate or mean across repetitions per sample size.
        label: Optional label identifying the estimator.
        title: Optional title for the plot.
        reference_values: Optional mapping of labels to horizontal reference
            values, such as the full-space and truncated integral truths.
        ax: Optional existing axes for comparing multiple estimators.
        output_path: Optional path for saving the figure as an image.

    Returns:
        fig, ax: Figure and axes for further customization or display.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(sample_sizes, estimates, marker='o', label=label)
    if reference_values is not None:
        for i, (reference_label, value) in enumerate(reference_values.items()):
            ax.axhline(value, color=f'C{i + 1}', linestyle='--', label=reference_label)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Integral Estimate')
    ax.set_xscale('log')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Integral Estimate vs Sample Size')
    ax.grid(True)
    if label is not None or reference_values:
        ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_rmse(
        sample_sizes: list[int], rmse_values: np.ndarray,
        label: str | None = None, title: str | None = None,
        reference_values: dict[str, float] | None = None,
        ax: Axes | None = None, output_path: str | Path | None = None
) -> tuple[Figure, Axes]:
    """
    Plots previously computed empirical RMSE values against sample size.

    Args:
        sample_sizes: Number of samples used in each configuration.
        rmse_values: Positive empirical RMSE values, one per sample size.
            Values must use an explicitly chosen integral truth.
        label: Optional label identifying the estimator and reference truth.
        title: Optional title, which can include the number of repetitions.
        reference_values: Optional mapping of labels to positive horizontal
            reference values, such as the absolute truncation bias.
        ax: Optional existing axes for comparing multiple curves.
        output_path: Optional path for saving the figure as an image.

    Returns:
        fig, ax: Figure and axes for further customization or display.
    """
    if np.any(np.asarray(rmse_values) <= 0):
        raise ValueError('RMSE values must be positive for a log-log plot.')
    if reference_values is not None and any(value <= 0 for value in reference_values.values()):
        raise ValueError('RMSE reference values must be positive for a log-log plot.')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(sample_sizes, rmse_values, marker='o', label=label)
    if reference_values is not None:
        for i, (reference_label, value) in enumerate(reference_values.items()):
            ax.axhline(value, color=f'C{i + 1}', linestyle='--', label=reference_label)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Empirical RMSE')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Empirical RMSE vs Sample Size')
    ax.grid(True)
    if label is not None or reference_values:
        ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax
