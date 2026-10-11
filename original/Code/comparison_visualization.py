import numpy as np
import matplotlib.pyplot as plt

est_uniform = np.load("Results\\uniform_estimates.npy", allow_pickle=True).item()
rmse_uniform = np.load("Results\\uniform_rmse.npy", allow_pickle=True).item()

est_init = np.load("Results\\normal_estimates.npy", allow_pickle=True).item()
rmse_init = np.load("Results\\normal_rmse.npy", allow_pickle=True).item()

est_ref = np.load("Results\\normal_estimates_refined.npy", allow_pickle=True).item()
rmse_ref = np.load("Results\\normal_rmse_refined.npy", allow_pickle=True).item()

TRUE_VALUE = np.pi

def plot_estimates_two(estA, estB, labelA, labelB, filename, true_value=TRUE_VALUE):
    NsA = np.array(sorted(estA.keys()))
    NsB = np.array(sorted(estB.keys()))
    valsA = np.array([estA[N] for N in NsA])
    valsB = np.array([estB[N] for N in NsB])

    plt.figure(figsize=(8,6))
    plt.plot(NsA, valsA, marker='o', label=labelA)
    plt.plot(NsB, valsB, marker='s', label=labelB)
    plt.axhline(true_value, linestyle='--', color='red', label="True value")

    plt.xscale("log")
    plt.xlabel("Sample size N (log scale)")
    plt.ylabel("Estimated Z")
    plt.title(f"Estimates Comparison: {labelA} vs {labelB}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_rmse_two(rmseA, rmseB, labelA, labelB, filename):

    NsA = np.array(sorted(rmseA.keys()))
    NsB = np.array(sorted(rmseB.keys()))
    rmsA = np.array([rmseA[N] for N in NsA])
    rmsB = np.array([rmseB[N] for N in NsB])

    plt.figure(figsize=(8,6))
    plt.loglog(NsA, rmsA, marker='o', label=labelA)
    plt.loglog(NsB, rmsB, marker='s', label=labelB)

    def add_slope(Ns, rms, label_prefix):
        mask = rms > 0
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(np.log(Ns[mask]), np.log(rms[mask]), 1)
            plt.loglog(Ns[mask], np.exp(intercept)*Ns[mask]**slope,
                       linestyle="--", label=f"{label_prefix} slope={slope:.2f}")

    add_slope(NsA, rmsA, labelA)
    add_slope(NsB, rmsB, labelB)

    plt.xlabel("Sample size N (log scale)")
    plt.ylabel("RMSE (log scale)")
    plt.title(f"RMSE Comparison: {labelA} vs {labelB}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_rmse_sqrtN_comparison(rmseA, rmseB, labelA, labelB, filename):

    NsA = np.array(sorted(rmseA.keys()))
    NsB = np.array(sorted(rmseB.keys()))

    adjA = np.array([rmseA[N] * np.sqrt(N) for N in NsA])
    adjB = np.array([rmseB[N] * np.sqrt(N) for N in NsB])

    plt.figure(figsize=(8,6))
    plt.loglog(NsA, adjA, marker='o', label=labelA)
    plt.loglog(NsB, adjB, marker='s', label=labelB)


    plt.xlabel("Sample size N (log scale)")
    plt.ylabel(r"Adjusted RMSE: RMSE $\times \sqrt{N}$")
    plt.title(f"Variance Constant Comparison: {labelA} vs {labelB}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_estimates_three_zoom(est_uniform, est_init, est_ref,
                              true_value=np.pi,
                              N_min=10**3,
                              filename="compare_estimates_three_zoom.png"):
    Ns_all = np.array(sorted(est_uniform.keys()))
    mask = Ns_all >= N_min
    Ns = Ns_all[mask]

    vals_u = np.array([est_uniform[N] for N in Ns])
    vals_i = np.array([est_init[N]    for N in Ns])
    vals_r = np.array([est_ref[N]     for N in Ns])

    plt.figure(figsize=(8,6))
    plt.plot(Ns, vals_u, marker='o', label="Uniform MC")
    plt.plot(Ns, vals_i, marker='s', label="Initial IS")
    plt.plot(Ns, vals_r, marker='^', label="Refined IS")
    plt.axhline(true_value, color='red', linestyle='--', label="True value")

    plt.xscale("log")
    plt.xlabel("Sample size N (log scale)")
    plt.ylabel("Estimated Z")
    plt.title(f"Monte Carlo Estimates vs N (N ≥ {N_min})")

    # 纵轴可以适当 zoom 一下，让差异更明显
    plt.ylim(3.0, 3.3)   # 视情况调：真值是 π≈3.14

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Initial IS vs Refined IS
    plot_estimates_two(
        est_init, est_ref,
        "Initial IS", "Refined IS",
        "compare_initial_vs_refined_estimates.png"
    )

    plot_rmse_two(
        rmse_init, rmse_ref,
        "Initial IS", "Refined IS",
        "compare_initial_vs_refined_rmse.png"
    )

    # Uniform MC vs Refined IS
    plot_estimates_two(
        est_uniform, est_ref,
        "Uniform MC", "Refined IS",
        "compare_uniform_vs_refined_estimates.png"
    )

    plot_rmse_two(
        rmse_uniform, rmse_ref,
        "Uniform MC", "Refined IS",
        "compare_uniform_vs_refined_rmse.png"
    )

    plot_rmse_sqrtN_comparison(
        rmse_uniform, rmse_ref,
        "Uniform MC", "Refined IS",
        "compare_uniform_vs_refined_rmse_sqrtN.png"
    )

    plot_estimates_three_zoom(
        est_uniform, est_init, est_ref,
        true_value=3.14,
        N_min=10**5,
        filename="compare_estimates_three_zoom_1e5.png"
    )



