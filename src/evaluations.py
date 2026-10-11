import numpy as np

def rmse(estimates: np.ndarray, true_value: float):
    """
    Calculate the Root Mean Square Error (RMSE) of the estimates compared to the true value.

    Args:
        estimates (list or np.ndarray): A list or array of estimated values.
        true_value (float): The true value to compare against.

    Returns:
        float: The RMSE of the estimates.
    """
    estimates = np.asarray(estimates)
    deviations = estimates - true_value
    return np.sqrt(np.mean(deviations**2))

def bias(estimates: np.ndarray, true_value: float):
    """
    Calculate the bias of the estimates compared to the true value.

    Args:
        estimates (list or np.ndarray): A list or array of estimated values.
        true_value (float): The true value to compare against.

    Returns:
        float: The bias of the estimates.
    """
    estimates = np.asarray(estimates)
    return np.mean(estimates) - true_value