import matplotlib.pyplot as plt
import numpy as np
from constants import *

'''def plot_heatmap(func, x_range, y_range, num_points=100, title="Heatmap", filename="heatmap.png"):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    print("Z min:", np.nanmin(Z), "max:", np.nanmax(Z))
    print("Z 1%:", np.nanpercentile(Z, 1), "99%:", np.nanpercentile(Z, 99))


    plt.figure(figsize=(8,6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(filename, dpi=300)
    plt.close()'''

def plot_heatmap(func, x_range, y_range, num_points=100,
                 title="Heatmap", filename="heatmap.png",
                 mode="linear"):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    Z = np.asarray(Z, dtype=float)

    if mode == "log":
        Z[Z <= 0] = np.nan
        Z = np.log10(Z)
        cb_label = "log10(weight)"
    elif mode == "clip":
        low = np.nanpercentile(Z, 1)
        high = np.nanpercentile(Z, 99)
        Z = np.clip(Z, low, high)
        cb_label = "weight (clipped)"
    else:
        cb_label = "Function Value"

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label=cb_label)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_3D_surface(func, x_range, y_range, num_points=150,
                    title="3D Surface Plot of $\\tilde{\\rho}(x)$",
                    filename="3d_surface.png"):
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(*x_range, num_points)
    y = np.linspace(*y_range, num_points)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="none",
        linewidth=0,
        antialiased=True,
    )

    fig.colorbar(surf, shrink=0.6, aspect=12, label="Function Value")

    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$\\tilde{\\rho}(x)$")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_weight_heatmaps_side_by_side(
    x_range=(-4, 4),
    y_range=(-4, 4),
    num_points=200,
    filename="weights_heatmap_comparison.png"
):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    W_init = INITIAL_WEIGHT_FUNCTION(X, Y)
    W_ref  = REFINED_WEIGHT_FUNCTION(X, Y)

    W_init = np.asarray(W_init, dtype=float)
    W_ref  = np.asarray(W_ref, dtype=float)

    W_init[W_init <= 0] = np.nan
    W_ref[W_ref <= 0] = np.nan

    Z_init = np.log10(W_init)
    Z_ref  = np.log10(W_ref)

    Z_init = np.clip(Z_init, -50, None)
    Z_ref  = np.clip(Z_ref,  -50, None)


    global_vmin = np.nanmin([np.nanmin(Z_init), np.nanmin(Z_ref)])
    global_vmax = np.nanmax([np.nanmax(Z_init), np.nanmax(Z_ref)])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    cf1 = axes[0].contourf(
        X, Y, Z_init, levels=50, cmap="viridis",
        vmin=global_vmin, vmax=global_vmax
    )
    axes[0].set_title("Initial proposal")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    cf2 = axes[1].contourf(
        X, Y, Z_ref, levels=50, cmap="viridis",
        vmin=global_vmin, vmax=global_vmax
    )
    axes[1].set_title("Refined proposal")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    cbar = fig.colorbar(
        cf1,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        aspect=25,
        label=r"$\log_{10} w(x)$"
    )

    fig.suptitle("Log-weight heatmaps $w(x)=\\tilde\\rho(x)/q(x)$", fontsize=12)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_heatmap(
        TARGET_FUNCTION,
        x_range=(-4, 4),
        y_range=(-4, 4),
        num_points=200,
        title="Heatmap of $\\tilde\\rho(x)$",
        filename="fuction_heatmap_log.png",
        mode="log"
    )
    plot_3D_surface(
        TARGET_FUNCTION,
        x_range=(-4, 4),
        y_range=(-4, 4),
        num_points=100,
        title="3D Surface Plot of rho",
        filename="function_3d_surface.png"
    )
    plot_heatmap(
        lambda x1, x2: INITIAL_WEIGHT_FUNCTION(x1, x2),
        x_range=(-4, 4),
        y_range=(-4, 4),
        num_points=200,
        title="Heatmap of Importance Weight $w(x)=\\tilde\\rho(x)/q(x)$",
        filename="initial_weights_heatmap.png",
        mode="log"
    )
    plot_heatmap(
        lambda x1, x2: REFINED_WEIGHT_FUNCTION(x1, x2),
        x_range=(-4, 4),
        y_range=(-4, 4),
        num_points=200,
        title="Heatmap of Importance Weight With Refined Proposal $w(x)=\\tilde\\rho(x)/q(x)$",
        filename="refined_weights_heatmap.png",
        mode="log"
    )
    plot_weight_heatmaps_side_by_side(
        x_range=(-4, 4),
        y_range=(-4, 4),
        num_points=200,
        filename="weights_heatmap_comparison.png"
    )
