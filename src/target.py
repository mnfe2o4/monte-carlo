import numpy as np
from scipy.integrate import nquad

# target function
def target_function(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return np.exp(-x1**2 / 2 - 0.5*(2*x2 + x1**2 - 1)**2)

# log-target function
def log_target_function(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return -x1**2 / 2 - 0.5*(2*x2 + x1**2 - 1)**2

Z_TRUE = np.pi

def compute_truncated_integral(bounds):
    result, _ = nquad(
        lambda x1, x2: target_function(np.array([x1, x2])),
        bounds,
    )
    return result

Z_TRUNCATED_4 = compute_truncated_integral([[-4, 4], [-4, 4]])
Z_TRUNCATED_6 = compute_truncated_integral([[-6, 6], [-6, 6]])