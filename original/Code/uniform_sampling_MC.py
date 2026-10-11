from constants import *
import numpy as np
import matplotlib.pyplot as plt

def uniform_sampling_MC(num_samples):
    samples = np.random.uniform(low=-4, high=4, size=(num_samples, 2))
    weights = TARGET_FUNCTION(samples[:, 0], samples[:, 1])
    integral_estimate = np.mean(weights) * (8 * 8)
    return integral_estimate

def calculate_RMSE(num_samples, num_trials=1000, true_value=None):
    estimates = []
    for _ in range(num_trials):
        estimate = uniform_sampling_MC(num_samples)
        estimates.append(estimate)
    if true_value is not None:
        deviations = np.array(estimates) - true_value
        rmse = np.sqrt(np.mean(deviations**2))
    else:
        rmse = np.std(estimates)
    return rmse

def get_true_value(num_samples=10**8):
    return uniform_sampling_MC(num_samples)

if __name__ == "__main__":
    num_samples = 10000000
    integral_estimate = uniform_sampling_MC(num_samples)
    #rmse = calculate_RMSE(num_samples)
    print(f"Estimated integral with {num_samples} samples: {integral_estimate}")
    #print(f"RMSE over multiple trials: {rmse}")