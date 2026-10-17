"""
Estimate integrals using matching proposal sampling and density evaluation.
"""

import numpy as np
from tqdm import tqdm
from collections.abc import Callable
from scipy.special import logsumexp
from .evaluations import rmse, bias
from .proposals import Proposal

class MCEstimator:
    """
    Importance sampling estimator using a supplied Proposal object.

    The proposal generates samples and evaluates its normalized density.
    The integrand is supplied separately to each estimation call. A bounded
    proposal estimates the integral over its support, not omitted regions.

    Attributes:
        proposal: Proposal providing matching sampling and density methods.
        pdf: Callable bound to the proposal's normalized density.
        log_pdf: Callable bound to the proposal's direct log-density.

    Methods:
        estimate(num_samples, func, use_log, rng): Estimate one integral.
        generate_samples(num_samples, rng): Draw samples from the proposal.
        estimate_rmse(func, num_samples, num_repetitions, true_value,
            use_log, rng): Compute empirical RMSE across repeated estimates.
    """

    def __init__(
            self, proposal: Proposal
    ):
        """
        Store a proposal and its matching density evaluation methods.

        Args:
            proposal: Proposal providing batch sample generation and normalized
                single-point PDF and log-PDF evaluation.
        """
        self.proposal = proposal
        self.pdf = proposal.pdf
        self.log_pdf = proposal.log_pdf

    def estimate(
            self, num_samples: int, func: Callable[[np.ndarray], float],
            use_log: bool = False, rng: np.random.Generator | None = None
    ) -> float:
        """
        Estimate an integral using fresh samples from the proposal.

        Args:
            num_samples: Positive number of samples for this estimate.
            func: Callable accepting one point of shape (dim,) and returning
                the integrand, or its natural logarithm when use_log=True.
                The log branch requires a nonnegative integrand.
            use_log: If True, use log-target values and proposal log-densities,
                and aggregate log-weights using logsumexp. Otherwise, evaluate
                ordinary target-to-proposal density ratios.
            rng: Optional NumPy generator passed to proposal sample generation.
                The built-in proposals create a new generator when None.

        Returns:
            estimate: Mean importance weight estimating the integral over the
                proposal support, provided the weighted integrand is integrable.
        """
        samples = self.generate_samples(num_samples, rng)
        
        if use_log:
            log_proposal_pdf = np.array([self.log_pdf(sample) for sample in samples])
            log_target_values = np.array([func(sample) for sample in samples])
            
            log_weights = log_target_values - log_proposal_pdf
            estimate = float(np.exp(logsumexp(log_weights) - np.log(len(samples))))
        else:
            proposal_pdf_values = np.array([self.pdf(sample) for sample in samples])
            target_values = np.array([func(sample) for sample in samples])
            
            weights = target_values / proposal_pdf_values
            estimate = np.mean(weights)
    
        return estimate

    def generate_samples(self, num_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Delegate sample generation to the configured proposal.

        Args:
            num_samples: Number of samples to generate.
            rng: Optional NumPy generator passed unchanged to the proposal.

        Returns:
            samples: Array of proposal samples with shape (num_samples, dim).
        """
        return self.proposal.generate_samples(num_samples, rng)

    def estimate_rmse(
              self, func: Callable[[np.ndarray], float], num_samples: int, num_repetitions: int, 
              true_value: float, use_log: bool = False, quiet: bool = True, rng: np.random.Generator | None = None
    ) -> float:
        """
        Compute empirical RMSE across fresh repeated estimates at a fixed budget.

        Args:
            func: Callable accepting one point of shape (dim,) and returning
                the integrand, or its natural logarithm when use_log=True.
            num_samples: Positive sample count for each estimate.
            num_repetitions: Positive number of estimates used for evaluation.
            true_value: Explicit reference integral. Choosing a full-space
                reference for a bounded proposal includes truncation error.
            use_log: If True, interpret func as a log-integrand for each call.
            quiet: If True, suppress the tqdm progress bar.
            rng: Optional NumPy generator advanced across repetitions. Pass
                a seeded generator to reproduce the sequence of estimates.

        Returns:
            rmse_value: Square root of the mean squared deviation from
                true_value. A finite empirical value does not establish that
                the estimator has finite theoretical variance or RMSE.
        """
        estimates = self.repeated_estimates(func=func, num_samples=num_samples, num_repetitions=num_repetitions, use_log=use_log, quiet=quiet, rng=rng)
        
        return rmse(estimates, true_value)

    def estimate_bias(
              self, func: Callable[[np.ndarray], float], num_samples: int, num_repetitions: int, 
              true_value: float, use_log: bool = False, quiet: bool = True, rng: np.random.Generator | None = None
    ) -> float:
        """
        Compute empirical bias across fresh repeated estimates at a fixed budget.

        Args:
            func: Callable accepting one point of shape (dim,) and returning
                the integrand, or its natural logarithm when use_log=True.
            num_samples: Positive sample count for each estimate.
            num_repetitions: Positive number of estimates used for evaluation.
            true_value: Explicit reference integral. Choosing a full-space
                reference for a bounded proposal includes truncation error.
            use_log: If True, interpret func as a log-integrand for each call.
            quiet: If True, suppress the tqdm progress bar.
            rng: Optional NumPy generator advanced across repetitions. Pass
                a seeded generator to reproduce the sequence of estimates.

        Returns:
            bias_value: Mean deviation from true_value. A finite empirical value
                does not establish that the estimator has finite theoretical bias.
        """
        estimates = self.repeated_estimates(func=func, num_samples=num_samples, num_repetitions=num_repetitions, use_log=use_log, quiet=quiet, rng=rng)
        
        return bias(estimates, true_value)

    def repeated_estimates(
              self, func: Callable[[np.ndarray], float], num_samples: int, num_repetitions: int, 
              use_log: bool = False, quiet: bool = True, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        Generate a batch of repeated estimates at a fixed budget.

        Args:
            func: Callable accepting one point of shape (dim,) and returning
                the integrand, or its natural logarithm when use_log=True.
            num_samples: Positive sample count for each estimate.
            num_repetitions: Positive number of estimates to generate.
            use_log: If True, interpret func as a log-integrand for each call.
            quiet: If True, suppress the tqdm progress bar.
            rng: Optional NumPy generator advanced across repetitions. Pass
                a seeded generator to reproduce the sequence of estimates.

        Returns:
            estimates: Array of repeated estimates with shape (num_repetitions,).
        """
        estimates = []
        for _ in tqdm(range(num_repetitions), disable=quiet, desc=f"Estimating with {num_samples} samples"):
            estimate = self.estimate(num_samples=num_samples, func=func, use_log=use_log, rng=rng)
            estimates.append(estimate)
        
        return np.array(estimates)