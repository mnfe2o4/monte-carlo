from scipy.optimize import minimize, OptimizeResult
import numpy as np
from tqdm import tqdm
from collections.abc import Callable
from src.estimators import MCEstimator
from src.proposals import GaussianProposal
from src.target import log_target_function, Z_TRUE

def calculate_gaussian_rmse(mean: np.ndarray, cov: np.ndarray, num_samples: int, num_repetitions: int, rng: np.random.Generator | None = None) -> float:
    """
    Compute the RMSE of Monte Carlo estimates for a Gaussian distribution.

    Args:
        mean (np.ndarray): Mean vector of the Gaussian distribution.
        cov (np.ndarray): Covariance matrix of the Gaussian distribution.
        num_samples (int): Number of samples to draw for each estimate.
        num_repetitions (int): Number of repetitions to compute RMSE.
        rng (np.random.Generator | None): Random number generator for reproducibility.

    Returns:
        float: The computed RMSE value.
    """

    estimator = MCEstimator(GaussianProposal(mean, cov))
    rmse = estimator.estimate_rmse(
        log_target_function, num_samples, num_repetitions, Z_TRUE, use_log=True, rng=rng
    )
    return rmse

def target_rmse_objective(params: np.ndarray, num_samples: int, num_repetitions: int, seed: int) -> float:
    """
    Objective function for optimization to minimize RMSE.

    Args:
        params (np.ndarray): Parameters for the Gaussian proposal (mean and covariance).
        num_samples (int): Number of samples to draw for each estimate.
        num_repetitions (int): Number of repetitions to compute RMSE.
        seed (int): Seed for the random number generator for reproducibility.

    Returns:
        float: The computed RMSE value for the given parameters.
    """
    mean = np.array([0, params[0]])
    cov = np.array([[params[1], 0], [0, params[2]]])  # Assuming diagonal covariance
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    return calculate_gaussian_rmse(mean, cov, num_samples, num_repetitions, rng)

def find_refined_gaussian_params(initial_params: np.ndarray, num_samples: int, num_repetitions: int, seed: int, callback: Callable[[OptimizeResult], None] | None = None) -> OptimizeResult:
    """
    Optimize Gaussian proposal parameters to minimize RMSE.

    Args:
        initial_params (np.ndarray): Initial guess for the parameters (mean_y, cov_xx, cov_yy).
        num_samples (int): Number of samples to draw for each estimate.
        num_repetitions (int): Number of repetitions to compute RMSE.
        seed (int): Seed for the random number generator for reproducibility.

    Returns:
        OptimizeResult: The result of the optimization.
    """
    result = minimize(
        target_rmse_objective,
        initial_params,
        args=(num_samples, num_repetitions, seed),
        callback=callback,
        bounds=[(0, 5), (0.01, 10), (0.01, 10)]  # Example bounds for mean_y and covariance values
    )
    return result

if __name__ == "__main__":
    # Initial guess for the parameters: mean_y, cov_xx, cov_yy
    initial_params = np.array([0.0, 1.0, 0.75])
    num_samples = 10000
    num_repetitions = 100
    seed = 20261008

    with tqdm(desc="Optimizing Gaussian parameters") as p_bar:

        def callback(intermediate_result: OptimizeResult) -> None:
            x = intermediate_result.x

            p_bar.set_postfix({
                "mean_y": f"{x[0]:.5f}",
                "cov_xx": f"{x[1]:.5f}",
                "cov_yy": f"{x[2]:.5f}",
                "fun": f"{intermediate_result.fun:.6f}",
            })
            p_bar.update(1)

        optimized_results = find_refined_gaussian_params(initial_params, num_samples, num_repetitions, seed, callback=callback)
    print(f"Optimized Gaussian parameters (mean_y, cov_xx, cov_yy): {optimized_results.x}")
    print(f"Optimization Success: {optimized_results.success}, Message: {optimized_results.message}")