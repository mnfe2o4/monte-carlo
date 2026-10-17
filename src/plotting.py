"""
Plot stored Monte Carlo results and evaluate target geometry on fixed grids.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


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
        ax: Axes | None = None, output_path: str | Path | None = None,
        true_value: float | None = None
) -> tuple[Figure, Axes]:
    """
    Plot previously computed integral estimates against sample size.

    Args:
        sample_sizes: Number of samples used for each estimate.
        estimates: One estimate or mean across repetitions per sample size.
        label: Optional label identifying the estimator.
        title: Optional title for the plot.
        reference_values: Optional mapping of labels to additional dashed
            horizontal references, such as a truncated integral truth.
        ax: Optional existing axes for comparing multiple estimators.
        output_path: Optional path for saving the figure as an image.
        true_value: Optional integral truth shown as a dashed horizontal line.
            Pass the truth matching the comparison being presented.

    Returns:
        fig, ax: Figure and axes for further customization or display.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(sample_sizes, estimates, marker='o', label=label)
    if true_value is not None:
        ax.axhline(true_value, color='black', linestyle='--',
                   label=f'True Value = {true_value:.6g}')
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
    if label is not None or reference_values or true_value is not None:
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
        ax: Axes | None = None, output_path: str | Path | None = None,
        fit_slope: bool = False
) -> tuple[Figure, Axes]:
    """
    Plot previously computed empirical RMSE values against sample size.

    Args:
        sample_sizes: Number of samples used in each configuration.
        rmse_values: Positive empirical RMSE values, one per sample size.
            Values must use an explicitly chosen integral truth.
        label: Optional label identifying the estimator and reference truth.
        title: Optional title, which can include the number of repetitions.
        reference_values: Optional mapping of labels to positive dashed
            horizontal references, such as the absolute truncation bias.
        ax: Optional existing axes for comparing multiple curves.
        output_path: Optional path for saving the figure as an image.
        fit_slope: If True, fit a power law in log-log coordinates and show
            the fitted curve as a dashed line with its slope in the legend.
            Requires finite data and at least two distinct positive sample
            sizes. This empirical slope is not proof of a theoretical rate
            or of finite variance, especially for Gaussian IS in this study.

    Returns:
        fig, ax: Figure and axes for further customization or display.
    """
    if np.any(np.asarray(rmse_values) <= 0):
        raise ValueError('RMSE values must be positive for a log-log plot.')
    if reference_values is not None and any(value <= 0 for value in reference_values.values()):
        raise ValueError('RMSE reference values must be positive for a log-log plot.')

    if fit_slope:
        fit_sizes = np.asarray(sample_sizes, dtype=float)
        fit_errors = np.asarray(rmse_values, dtype=float)
        if fit_sizes.ndim != 1 or fit_errors.ndim != 1 or fit_sizes.shape != fit_errors.shape:
            raise ValueError('Sample sizes and RMSE values must be matching one-dimensional arrays.')
        if not np.all(np.isfinite(fit_sizes)) or not np.all(np.isfinite(fit_errors)):
            raise ValueError('Slope fitting requires finite sample sizes and RMSE values.')
        if np.any(fit_sizes <= 0) or len(np.unique(fit_sizes)) < 2:
            raise ValueError('Slope fitting requires at least two distinct positive sample sizes.')
        # Fit log(RMSE) = intercept + slope * log(N) across the supplied results.
        slope, intercept = np.polyfit(np.log(fit_sizes), np.log(fit_errors), 1)
        order = np.argsort(fit_sizes)
        fit_sizes = fit_sizes[order]
        fitted_rmse = np.exp(intercept + slope * np.log(fit_sizes))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    line, = ax.plot(sample_sizes, rmse_values, marker='o', label=label)
    if fit_slope:
        fit_label = f'{label}: Empirical Fit' if label is not None else 'Empirical Fit'
        ax.plot(fit_sizes, fitted_rmse, color=line.get_color(), linestyle='--',
                label=f'{fit_label} (slope = {slope:.2f})')
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
    if label is not None or reference_values or fit_slope:
        ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def _evaluate_grid(
        func: Callable[[np.ndarray], float], xlim: tuple[float, float],
        ylim: tuple[float, float], num_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a pointwise function on a regular two-dimensional grid.

    Args:
        func: Function accepting one point of shape (2,) and returning a scalar.
        xlim: Finite increasing bounds for the first coordinate.
        ylim: Finite increasing bounds for the second coordinate.
        num_points: Number of grid points along each coordinate, at least two.

    Returns:
        X, Y, values: Coordinate meshes and function values, each with shape
            (num_points, num_points).
    """
    if isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)) or num_points < 2:
        raise ValueError('num_points must be an integer of at least two.')
    for bounds in (xlim, ylim):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError('Coordinate limits must be finite increasing pairs.')

    x = np.linspace(*xlim, num_points)
    y = np.linspace(*ylim, num_points)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    values = np.array([func(point) for point in points], dtype=float).reshape(X.shape)
    return X, Y, values


def plot_heatmap(
        func: Callable[[np.ndarray], float], title: str | None = None,
        xlim: tuple[float, float] = (-4, 4),
        ylim: tuple[float, float] = (-4, 4), num_points: int = 200,
        colorbar_label: str = 'Function Value', cmap: str = 'viridis',
        vmin: float | None = None, vmax: float | None = None,
        ax: Axes | None = None, output_path: str | Path | None = None
) -> tuple[Figure, Axes]:
    """
    Plot a scalar function on a two-dimensional grid without drawing samples.

    Args:
        func: Function accepting one point of shape (2,). Returned values are
            plotted directly; pass a log-function to display logarithmic values.
        title: Optional title for the plot.
        xlim: Finite increasing limits for the first coordinate.
        ylim: Finite increasing limits for the second coordinate.
        num_points: Number of grid points along each coordinate, at least two.
        colorbar_label: Label identifying the plotted quantity and its scale.
        cmap: Matplotlib colormap name.
        vmin: Optional lower color limit. Lower values use the endpoint color.
        vmax: Optional upper color limit. Higher values use the endpoint color.
        ax: Optional existing axes for arranging multiple plots.
        output_path: Optional path for saving the figure as an image.

    Returns:
        fig, ax: Figure and axes for further customization or display.
            Nonfinite grid values are masked rather than replaced by zero.
    """
    X, Y, values = _evaluate_grid(func, xlim, ylim, num_points)
    if not np.any(np.isfinite(values)):
        raise ValueError('The heatmap requires at least one finite grid value.')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    else:
        fig = ax.figure

    # Rasterize the dense grid while retaining vector text in PDF exports.
    heatmap = ax.pcolormesh(
        X, Y, np.ma.masked_invalid(values), shading='auto', cmap=cmap,
        vmin=vmin, vmax=vmax, rasterized=True
    )
    fig.colorbar(heatmap, ax=ax, label=colorbar_label)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title if title is not None else 'Function Heatmap')

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_3d_surface(
        func: Callable[[np.ndarray], float], title: str | None = None,
        xlim: tuple[float, float] = (-4, 4),
        ylim: tuple[float, float] = (-4, 4), num_points: int = 150,
        zlabel: str = 'Function Value', cmap: str = 'viridis',
        ax: Axes3D | None = None, output_path: str | Path | None = None
) -> tuple[Figure, Axes3D]:
    """
    Plot the surface of a scalar function over a two-dimensional grid.

    Args:
        func: Function accepting one point of shape (2,) and returning a scalar.
        title: Optional title for the plot.
        xlim: Finite increasing limits for the first coordinate.
        ylim: Finite increasing limits for the second coordinate.
        num_points: Number of grid points along each coordinate, at least two.
        zlabel: Label for the vertical axis and colorbar.
        cmap: Matplotlib colormap name.
        ax: Optional existing axes created with projection='3d'.
        output_path: Optional path for saving the figure as an image.

    Returns:
        fig, ax: Figure and three-dimensional axes for customization or display.
            Nonfinite grid values are masked.
    """
    if ax is not None and not isinstance(ax, Axes3D):
        raise ValueError('Surface plotting requires axes with projection="3d".')
    X, Y, values = _evaluate_grid(func, xlim, ylim, num_points)
    if not np.any(np.isfinite(values)):
        raise ValueError('The surface requires at least one finite grid value.')

    if ax is None:
        fig = plt.figure(figsize=(10, 7), layout='constrained')
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Use the requested grid resolution instead of the default surface subsampling.
    surface = ax.plot_surface(
        X, Y, np.ma.masked_invalid(values), cmap=cmap, edgecolor='none',
        rcount=num_points, ccount=num_points, rasterized=True
    )
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.12, label=zlabel)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(zlabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title if title is not None else 'Function Surface')

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_weight_heatmaps(
        log_target: Callable[[np.ndarray], float],
        proposal_log_pdfs: dict[str, Callable[[np.ndarray], float]],
        title: str | None = None, xlim: tuple[float, float] = (-4, 4),
        ylim: tuple[float, float] = (-4, 4), num_points: int = 200,
        cmap: str = 'viridis', vmin: float | None = None,
        vmax: float | None = None, output_path: str | Path | None = None
) -> tuple[Figure, np.ndarray]:
    """
    Compare importance log-weights using one coordinate grid and color scale.

    Args:
        log_target: Natural log of the unnormalized target at one point of
            shape (2,). Evaluate it directly instead of logging a tiny density.
        proposal_log_pdfs: Nonempty mapping from panel titles to normalized
            proposal log-densities. Each callable accepts one point of shape (2,)
            and returns a natural logarithm. One entry produces a single panel.
        title: Optional title for the whole comparison.
        xlim: Finite increasing limits for the first coordinate.
        ylim: Finite increasing limits for the second coordinate.
        num_points: Number of grid points along each coordinate, at least two.
        cmap: Matplotlib colormap name shared by every panel.
        vmin: Optional common lower limit for log10 weights. If None, use the
            minimum finite value across all panels, without percentile clipping.
        vmax: Optional common upper limit for log10 weights. If None, use the
            maximum finite value across all panels.
        output_path: Optional path for saving the figure as an image.

    Returns:
        fig, axes: Figure and one-dimensional array containing one axes per
            proposal, with one shared colorbar. Nonfinite log-weights, including
            points outside proposal support, are masked. This finite-window
            visualization does not establish finite importance-weight variance.
    """
    if not proposal_log_pdfs:
        raise ValueError('Provide at least one proposal log-density.')

    X, Y, log_target_values = _evaluate_grid(log_target, xlim, ylim, num_points)
    log_weights = []
    for proposal_log_pdf in proposal_log_pdfs.values():
        _, _, log_proposal_values = _evaluate_grid(proposal_log_pdf, xlim, ylim, num_points)
        # Convert natural log-weights to log10 without exponentiating densities.
        with np.errstate(invalid='ignore'):
            values = (log_target_values - log_proposal_values) / np.log(10.0)
        if not np.any(np.isfinite(values)):
            raise ValueError('Each proposal requires at least one finite log-weight on the grid.')
        log_weights.append(np.ma.masked_invalid(values))

    # Every panel receives the same normalization object, not just the same cmap.
    norm = Normalize(vmin=vmin, vmax=vmax)
    norm.autoscale_None(np.ma.stack(log_weights))
    fig, axes = plt.subplots(
        1, len(log_weights), figsize=(6 * len(log_weights), 5),
        squeeze=False, layout='constrained'
    )
    axes = axes[0]
    for ax, label, values in zip(axes, proposal_log_pdfs, log_weights):
        heatmap = ax.pcolormesh(
            X, Y, values, shading='auto', cmap=cmap, norm=norm, rasterized=True
        )
        ax.set_title(label)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    fig.colorbar(
        heatmap, ax=axes.tolist(), shrink=0.85,
        label=r'$\log_{10} w(x)$'
    )
    fig.suptitle(title if title is not None else r'Importance Log-Weights: $w(x)=\widetilde\rho(x)/q(x)$')

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, axes
