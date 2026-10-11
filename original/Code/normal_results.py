import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from normal_sampling_MC import *
from constants import *

refined_mean = [0, 0.19]
refined_cov = np.array([[1, 0], [0, 0.25]]) * (2.28**2)

# Functions to compute estimates and RMSEs for various sample sizes
def normal_estimate_results(sample_sizes, mean=MEAN, cov=COV):
    results = {}
    for N in sample_sizes:
        estimate = normal_sampling_MC(N, mean=mean, cov=cov)
        results[N] = estimate
        print(f"Sample Size: {N}, Estimate: {estimate}")
    return results

def normal_rmse_results(sample_sizes, num_trials=1000, true_value=None, mean=MEAN, cov=COV):
    results = {}
    for N in sample_sizes:
        rmse = calculate_RMSE(N, num_trials=num_trials, true_value=true_value, mean=mean, cov=cov)
        results[N] = rmse
        print(f"Sample Size: {N}, RMSE: {rmse}")
    return results

# Plotting functions
def plot_estimates(estimates, true_value):
    Ns = np.array(list(estimates.keys()))
    vals = np.array(list(estimates.values()))

    plt.figure(figsize=(7,5))
    plt.plot(Ns, vals, marker='o', label="MC estimate")
    plt.axhline(true_value, color='red', linestyle='--', label="Reference value")
    plt.xscale("log")
    plt.xlabel("Sample size N (log scale)")
    plt.ylabel("Estimated Z")
    plt.title("Monte Carlo Estimates vs N (Importance Sampling)")
    plt.legend()
    plt.grid(True)
    plt.savefig("importance_estimates.png", dpi=300)
    plt.close()

def plot_rmse(rmse_results):
    Ns = np.array(list(rmse_results.keys()))
    rmses = np.array(list(rmse_results.values()))

    plt.figure(figsize=(7,5))
    plt.loglog(Ns, rmses, marker='o', label="Empirical RMSE")

    # Fit slope to log-log data (expect ≈ -1/2)
    slope, intercept = np.polyfit(np.log(Ns), np.log(rmses), 1)
    plt.loglog(Ns, np.exp(intercept) * Ns**slope, 
               label=f"Fit slope = {slope:.2f}", linestyle="--")

    plt.xlabel("Sample size N (log scale)")
    plt.ylabel("RMSE (log scale)")
    plt.title("RMSE vs N (Importance Sampling)")
    plt.legend()
    plt.grid(True)
    plt.savefig("importance_rmse.png", dpi=300)
    plt.close()

def plot_sample_cloud(num_samples=5000, mean=MEAN, cov=COV):
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=num_samples)

    plt.figure(figsize=(6,6))
    plt.scatter(samples[:,0], samples[:,1], s=5, alpha=0.3)
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Importance Sampling in $[-4,4]^2$")
    plt.grid(True)
    plt.savefig("importance_sample_cloud.png", dpi=300)
    plt.close()

def plot_histogram_of_estimator(N, num_trials=200, true_value=None):
    estimates = [normal_sampling_MC(N) for _ in range(num_trials)]

    plt.figure(figsize=(7,5))
    plt.hist(estimates, bins=30, alpha=0.7)
    if true_value is not None:
        plt.axvline(true_value, color='red', linestyle='--', label="True value")
    plt.title(f"Distribution of MC Estimator (N={N})")
    plt.xlabel("Estimated Z")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"normal_histogram_N{N}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Define sample sizes to test
    sample_sizes_estimate = [int(10**(i/4)) for i in range(8, 33)] # From 10^2 to 10^8
    sample_sizes_rmse = [int(10**(i/2)) for i in range(4, 15)] # From 10^2 to 10^7
    
    # Get true value using a large number of samples
    true_value = get_true_value(num_samples=10**8, mean=refined_mean, cov=refined_cov)
    print(f"True Value (approximated): {true_value}")
    
    # Compute estimates
    print("Normal Sampling MC Estimates:")
    estimates=normal_estimate_results(sample_sizes_estimate, mean=refined_mean, cov=refined_cov)

    # Compute RMSEs
    print("\nNormal Sampling MC RMSEs:")
    rmse_results=normal_rmse_results(
        sample_sizes_rmse, 
        num_trials=100, 
        true_value=true_value, 
        mean=refined_mean, 
        cov=refined_cov
    )

    # Plot all figures
    plot_estimates(estimates, true_value)
    plot_rmse(rmse_results)
    plot_sample_cloud(mean=refined_mean, cov=refined_cov)
    plot_histogram_of_estimator(N=5000, num_trials=300, true_value=true_value)

    # Save results to .npy files
    np.save("normal_estimates_refined.npy", estimates)
    np.save("normal_rmse_refined.npy", rmse_results)

    print("All results computed and plots saved.")