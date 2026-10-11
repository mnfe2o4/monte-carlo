import numpy as np
from collections.abc import Callable
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def uniform_mc_estimator(
        num_samples, func: Callable[[np.ndarray], float], low: np.ndarray, high: np.ndarray, rng: np.random.Generator | None = None
):
    """
    Monte Carlo estimator using uniform sampling.

    Args:
        num_samples: Number of samples to draw.
        func: Function accepting one sample of shape (dim,) and returning a scalar function value.
        low: Lower bounds for each dimension (1D numpy array).
        high: Upper bounds for each dimension (1D numpy array).
        rng: Optional random number generator. If None, use np.random.default_rng().

    Returns:
        estimate: The Monte Carlo estimate of the integral.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(low) != len(high):
        raise ValueError("Low and high bounds must have the same dimension.")
    elif len(low) == 0:
        raise ValueError("Low and high bounds must not be empty.")

    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")

    if not all(low < high):
        raise ValueError("Each element of low must be less than the corresponding element of high.")
    
    dim = len(low)
    samples = rng.uniform(low=low, high=high, size=(num_samples, dim))
    weights = np.array([func(sample) for sample in samples])
    
    volume = np.prod(high - low)
    estimate = np.mean(weights) * volume
    return estimate

def importance_mc_estimator_gaussian(
        num_samples, func: Callable[[np.ndarray], float],
        mean: np.ndarray, cov: np.ndarray, use_log: bool = False, rng: np.random.Generator | None = None
):
    """
    Monte Carlo estimator using importance sampling with a Gaussian proposal distribution.

    Args:
        num_samples: Number of samples to draw.
        func: Function accepting one sample of shape (dim,) and returning a scalar. Pass the unnormalized target density when use_log=False, or its natural logarithm when use_log=True.
        mean: Mean vector of the Gaussian proposal distribution (1D numpy array).
        cov: Covariance matrix of the Gaussian proposal distribution (2D numpy array).
        use_log: If True, interpret func as the log-target function and aggregate log-weights using logsumexp.
        rng: Optional random number generator. If None, use np.random.default_rng().

    Returns:
        estimate: The Monte Carlo estimate of the integral.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(mean) != cov.shape[0] or len(mean) != cov.shape[1]:
        raise ValueError("Mean vector and covariance matrix dimensions do not match.")

    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")
    
    samples = rng.multivariate_normal(mean=mean, cov=cov, size=num_samples)

    if use_log:
        log_proposal_pdf = multivariate_normal.logpdf(samples, mean=mean, cov=cov)
        log_target_values = np.array([func(sample) for sample in samples])
        
        log_weights = log_target_values - log_proposal_pdf
        estimate = float(np.exp(logsumexp(log_weights) - np.log(num_samples)))
    else:
        proposal_pdf = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        target_values = np.array([func(sample) for sample in samples])
        
        weights = target_values / proposal_pdf 
        estimate = np.mean(weights)
    
    return estimate

def importance_mc_estimator_custom(
        samples, func: Callable[[np.ndarray], float], proposal_pdf: Callable[[np.ndarray], float],
        use_log: bool = False, rng: np.random.Generator | None = None
):
    """
    Monte Carlo estimator using importance sampling with a custom proposal distribution.

    Args:
        samples: Array of samples from the proposal distribution.
        func: Function accepting one sample of shape (dim,) and returning a scalar function value.
        proposal_pdf: Function accepting one sample of shape (dim,) and returning the probability density of the proposal distribution at that point.
        use_log: If True, interpret func and proposal_pdf as the log-target function and log-proposal PDF respectively, and aggregate log-weights using logsumexp.
        rng: Optional random number generator. If None, use np.random.default_rng().

    Returns:
        estimate: The Monte Carlo estimate of the integral.
    """
    if rng is None:
        rng = np.random.default_rng()

    if use_log:
        log_proposal_pdf = np.array([proposal_pdf(sample) for sample in samples])
        log_target_values = np.array([func(sample) for sample in samples])
        
        log_weights = log_target_values - log_proposal_pdf
        estimate = float(np.exp(logsumexp(log_weights) - np.log(len(samples))))
    else:
        proposal_pdf_values = np.array([proposal_pdf(sample) for sample in samples])
        target_values = np.array([func(sample) for sample in samples])
        
        weights = target_values / proposal_pdf_values
        estimate = np.mean(weights)

    return estimate
    