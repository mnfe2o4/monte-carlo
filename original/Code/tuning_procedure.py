import numpy as np
from constants import *
from normal_sampling_MC import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    b_values = np.linspace(0.1, 0.3, 21)
    c_values = np.linspace(2.2, 2.4, 21)

    rmse_grid = np.zeros((len(b_values), len(c_values)))

    for i, b in enumerate(b_values):
        shifted_mean = [0, b]
        for j, c in enumerate(c_values):
            scaled_cov = np.array([[1, 0], [0, 0.25]]) * (c**2)

            rmse = calculate_RMSE(
                num_samples=10000,
                mean=shifted_mean,
                cov=scaled_cov,
                num_trials=200,
                true_value=np.pi,
            )
            if not (0 < rmse < 0.2):
                rmse = 0.2
            rmse_grid[i, j] = rmse

            if (j + 1) % 10 == 0:
                print(f"Completed b index {i+1}/{len(b_values)}, c index {j+1}/{len(c_values)}")

    min_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    best_b = b_values[min_idx[0]]
    best_c  = c_values[min_idx[1]]
    best_rmse = rmse_grid[min_idx]

    print("Best b =", best_b)
    print("Best c  =", best_c)
    print("Best RMSE =", best_rmse)

    np.save("rmse_grid.npy", rmse_grid)
    np.save("b_values.npy", b_values)
    np.save("c_values.npy", c_values)


    B, C = np.meshgrid(c_values, b_values)
    plt.figure(figsize=(8,6))
    plt.contourf(c_values, b_values, rmse_grid, levels=50, cmap="viridis")
    plt.colorbar(label="RMSE")
    plt.xlabel("scale c")
    plt.ylabel("mean b")
    plt.title("RMSE over (mean b, scale c)")
    plt.savefig("rmse_heatmap.png", dpi=300)
    plt.close()
