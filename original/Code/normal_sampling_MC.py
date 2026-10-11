from constants import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def importance_sampling_weight(x, y, mean=MEAN, cov=COV):
    proposal_pdf = stats.multivariate_normal.pdf(
        np.column_stack((x, y)),
        mean=mean,
        cov=cov
    )
    target_pdf = TARGET_FUNCTION(x, y)
    weights = target_pdf / proposal_pdf
    return weights

def normal_sampling_MC(num_samples, mean=MEAN, cov=COV):
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=num_samples)
    weights = importance_sampling_weight(samples[:, 0], samples[:, 1], mean=mean, cov=cov)
    integral_estimate = np.mean(weights)
    return integral_estimate

def calculate_RMSE(num_samples, mean=MEAN, cov=COV, num_trials=1000, true_value=None):
    estimates = []
    for _ in range(num_trials):
        estimate = normal_sampling_MC(num_samples, mean=mean, cov=cov)
        estimates.append(estimate)
    if true_value is not None:
        deviations = np.array(estimates) - true_value
        rmse = np.sqrt(np.mean(deviations**2))
    else:
        rmse = np.std(estimates)
    return rmse

def get_true_value(num_samples=10**7, mean=MEAN, cov=COV):
    return normal_sampling_MC(num_samples, mean=mean, cov=cov)

if __name__ == "__main__":
    num_samples = 100000
    integral_estimate = normal_sampling_MC(num_samples)
    rmse = calculate_RMSE(num_samples)
    print(f"Estimated integral with {num_samples} samples: {integral_estimate}")
    print(f"RMSE over multiple trials: {rmse}")