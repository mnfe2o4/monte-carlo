"""
Define the nonlinear target and reference integrals for the main case study.
"""

import numpy as np
from scipy.integrate import nquad

# target function
def target_function(x: np.ndarray) -> float:
    """
    Evaluate the unnormalized nonlinear target at one two-dimensional point.

    Args:
        x: Coordinate array of shape (2,).

    Returns:
        density: Nonnegative target value. Its full-space integral is pi.
    """
    x1, x2 = x[0], x[1]
    return np.exp(-x1**2 / 2 - 0.5*(2*x2 + x1**2 - 1)**2)

# log-target function
def log_target_function(x: np.ndarray) -> float:
    """
    Evaluate the natural logarithm of the target directly.

    Args:
        x: Coordinate array of shape (2,).

    Returns:
        log_density: Log-target value before exponentiation, preserving tail
            information that may be lost when the ordinary density underflows.
    """
    x1, x2 = x[0], x[1]
    return -x1**2 / 2 - 0.5*(2*x2 + x1**2 - 1)**2

# Exact full-space integral obtained by the nonlinear coordinate transform.
Z_TRUE = np.pi
Z_TRUNCATED_4 = 3.1317084117972485
Z_TRUNCATED_6 = 3.1404622456392373

def compute_truncated_integral(bounds):
    """
    Compute a reference target integral over a two-dimensional box.

    Args:
        bounds: Two (lower, upper) coordinate intervals passed to scipy nquad.

    Returns:
        integral: Adaptive numerical quadrature result, not a Monte Carlo
            estimate. The quadrature error estimate is not returned.
    """
    result, _ = nquad(
        lambda x1, x2: target_function(np.array([x1, x2])),
        bounds,
    )
    return result

if __name__ == "__main__":
    # Box references differ from Z_TRUE by the omitted mass outside each box.
    Z_TRUNCATED_4 = compute_truncated_integral([[-4, 4], [-4, 4]])
    Z_TRUNCATED_6 = compute_truncated_integral([[-6, 6], [-6, 6]])
    print(f"Z_TRUE = {Z_TRUE}", f"Z_TRUNCATED_4 = {Z_TRUNCATED_4}", f"Z_TRUNCATED_6 = {Z_TRUNCATED_6}")