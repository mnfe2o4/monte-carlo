import numpy as np
from collections.abc import Callable
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from .evaluations import rmse, bias

class MCEstimator:
    """Base importance sampling estimator for a subclass-defined proposal.

    Subclasses implement generate_samples() using the same proposal as pdf
    and log_pdf. The target function is supplied to estimate() separately.

    Args:
        pdf: Proposal density accepting one sample of shape (dim,).
        log_pdf: Natural logarithm of the same proposal density, computed
            directly to avoid underflow from taking the logarithm of pdf.

    Attributes:
        pdf: Callable returning the proposal density at one sample.
        log_pdf: Callable returning the proposal log-density at one sample.
    """

    def __init__(
            self, pdf: Callable[[np.ndarray], float],
            log_pdf: Callable[[np.ndarray], float]
    ):
        self.pdf = pdf
        self.log_pdf = log_pdf

    def estimate(
            self, num_samples: int, func: Callable[[np.ndarray], float],
            use_log: bool = False, rng: np.random.Generator | None = None
    ) -> float:
        """
        Estimate an integral using samples from this estimator's proposal.
    
        Args:
            num_samples: Number of samples to draw.
            func: Function accepting one sample of shape (dim,) and returning
                the target value, or its natural logarithm when use_log=True.
            use_log: If True, use func as the log-target, self.log_pdf as the
                proposal log-density, and logsumexp to aggregate log-weights.
                Otherwise, use func as the target and self.pdf as the proposal
                density. This flag only specifies the target representation.
            rng: Optional random number generator. If None, use np.random.default_rng().
    
        Returns:
            estimate: The Monte Carlo estimate of the integral.
        """
        samples = self.generate_samples(num_samples, rng)
        if rng is None:
                rng = np.random.default_rng()
        
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
        Generate samples from a given proposal distribution.

        Args:
            num_samples: Number of samples to generate.
            rng: Optional random number generator. If None, use np.random.default_rng().

        Returns:
            samples: Array of generated samples.
        """
        pass  # This method should be implemented in subclasses to generate samples from the specific proposal distribution.

    def estimate_rmse(
              self, func: Callable[[np.ndarray], float], num_samples: int, num_repetitions: int, 
              true_value: float, rng: np.random.Generator | None = None, use_log: bool = False
    ) -> float:
        """
        Calculate empirical RMSE across repeated integral estimates.

        Args:
            func: Function accepting one sample of shape (dim,) and returning
                the target value, or its natural logarithm when use_log=True.
            num_samples: Number of samples to use in each repetition.
            num_repetitions: Number of repetitions to perform.
            true_value: The true value of the integral for RMSE calculation.
            rng: Optional random number generator. If None, use np.random.default_rng().
            use_log: If True, interpret func as the log-target and compute
                each estimate using proposal log-densities and logsumexp.

        Returns:
            rmse_value: The estimated RMSE of the Monte Carlo estimator.
        """
        estimates = []
        for _ in range(num_repetitions):
            estimate = self.estimate(num_samples=num_samples, func=func, use_log=use_log, rng=rng)
            estimates.append(estimate)
        
        return rmse(estimates, true_value)

class UniformMCEstimator(MCEstimator):
    """Monte Carlo integral estimator with a uniform proposal over a box.

    Estimates the target integral within bounds. When compared with a
    full-space integral, omitted mass outside the box is truncation bias.

    Args:
        bounds: Lower and upper bounds for each coordinate.

    Attributes:
        bounds: Box used for both proposal densities and sample generation.
    """

    def __init__(self, bounds: list[tuple[float, float]]):
        super().__init__(pdf=self.uniform_pdf(bounds), log_pdf=self.uniform_log_pdf(bounds))
        self.bounds = bounds

    def uniform_pdf(self, bounds: list[tuple[float, float]]) -> Callable[[np.ndarray], float]:
        """
        Create a uniform probability density function over the specified bounds.

        Args:
            bounds: List of tuples specifying the lower and upper bounds for each dimension.

        Returns:
            pdf: A callable that computes the uniform PDF for a given sample.
        """
        def pdf(x: np.ndarray) -> float:
            for i, (lower, upper) in enumerate(bounds):
                if not (lower <= x[i] <= upper):
                    return 0.0
            volume = np.prod([upper - lower for lower, upper in bounds])
            return 1.0 / volume
        return pdf

    def uniform_log_pdf(self, bounds: list[tuple[float, float]]) -> Callable[[np.ndarray], float]:
        """Create the uniform proposal log-density over the specified box.

        Args:
            bounds: Lower and upper bounds for each coordinate.

        Returns:
            Callable returning minus the log-volume inside the box and
            negative infinity outside it.
        """
        log_volume = np.sum(np.log([upper - lower for lower, upper in bounds]))

        def log_pdf(x: np.ndarray) -> float:
            for i, (lower, upper) in enumerate(bounds):
                if not (lower <= x[i] <= upper):
                    return -np.inf
            return float(-log_volume)
        return log_pdf

    def generate_samples(self, num_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Generate samples uniformly from the specified bounds.

        Args:
            num_samples: Number of samples to generate.
            rng: Optional random number generator. If None, use np.random.default_rng().

        Returns:
            samples: Array of generated samples.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        samples = np.array([rng.uniform(low=lower, high=upper, size=num_samples) for lower, upper in self.bounds]).T
        return samples

class GaussianMCEstimator(MCEstimator):
    """Importance sampling estimator with a multivariate Gaussian proposal.

    Samples over the full space and provides both proposal density forms.
    Whether the target is supplied as a density or log-density is specified
    when calling estimate() or estimate_rmse().

    Args:
        mean: Proposal mean vector, shape (dim,).
        cov: Positive-definite proposal covariance, shape (dim, dim).

    Attributes:
        mean: Mean used to generate proposal samples.
        cov: Covariance used to generate proposal samples.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        super().__init__(pdf=self.gaussian_pdf(mean, cov), log_pdf=self.gaussian_log_pdf(mean, cov))
        self.mean = mean
        self.cov = cov

    def gaussian_pdf(self, mean: np.ndarray, cov: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        Create a Gaussian probability density function with the specified mean and covariance.

        Args:
            mean: Mean vector of the Gaussian distribution.
            cov: Covariance matrix of the Gaussian distribution.

        Returns:
            pdf: A callable that computes the Gaussian PDF for a given sample.
        """
        def pdf(x: np.ndarray) -> float:
            return multivariate_normal.pdf(x, mean=mean, cov=cov)
        return pdf

    def gaussian_log_pdf(self, mean: np.ndarray, cov: np.ndarray) -> Callable[[np.ndarray], float]:
        """Create a directly evaluated Gaussian proposal log-density.

        Args:
            mean: Proposal mean vector, shape (dim,).
            cov: Proposal covariance matrix, shape (dim, dim).

        Returns:
            Callable returning the proposal log-density at one sample.
        """
        def log_pdf(x: np.ndarray) -> float:
            return multivariate_normal.logpdf(x, mean=mean, cov=cov)
        return log_pdf

    def generate_samples(self, num_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Generate samples from a Gaussian distribution with the specified mean and covariance.

        Args:
            num_samples: Number of samples to generate.
            rng: Optional random number generator. If None, use np.random.default_rng().

        Returns:
            samples: Array of generated samples.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        samples = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=num_samples)
        return samples