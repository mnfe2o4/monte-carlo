"""
Provide normalized proposal densities and their matching sample generators.
"""

import numpy as np
from collections.abc import Callable
from scipy.stats import multivariate_normal

class Proposal:
    """
    Proposal distribution combining sample generation and density evaluation.

    The density must be normalized and match the distribution used by
    sample_generator. Density callables accept one point, while the
    generator returns a batch of points.

    Attributes:
        pdf: Callable returning the normalized density at a point of
            shape (dim,).
        log_pdf: Callable directly returning the natural logarithm of
            the same density, with -np.inf outside its support.
        sample_generator: Callable accepting (n, rng) and returning
            samples of shape (n, dim).

    Methods:
        generate_samples(n, rng): Generate a batch of proposal samples.
    """

    def __init__(
            self, pdf: Callable[[np.ndarray], float], log_pdf: Callable[[np.ndarray], float],
            sample_generator: Callable[[int, np.random.Generator | None], np.ndarray]
    ):
        """
        Store matching density functions and a sample generator.

        Args:
            pdf: Normalized proposal density accepting one point of shape
                (dim,) and returning a scalar.
            log_pdf: Directly evaluated natural logarithm of the same density.
            sample_generator: Callable accepting a sample count and an optional
                NumPy generator, and returning an array of shape (n, dim).
        """
        self.pdf = pdf
        self.log_pdf = log_pdf
        self.sample_generator = sample_generator

    def generate_samples(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Generate a batch of samples using the stored sample generator.

        Args:
            n: Number of samples to generate.
            rng: Optional random number generator passed to sample_generator.
                The supplied generator defines how None is handled.

        Returns:
            samples: Array of proposal samples with shape (n, dim).
        """
        return self.sample_generator(n, rng)

class UniformProposal(Proposal):
    """
    Uniform proposal over a box defined by coordinate bounds.

    Its density is the reciprocal of the box volume inside the box and
    zero outside. Using it for integration estimates the integral over
    this box; omitted target mass remains a truncation bias relative to
    the full-space integral.

    Attributes:
        bounds: Lower and upper bounds for each of the dim coordinates.
        pdf: Callable bound to uniform_pdf.
        log_pdf: Callable bound to uniform_log_pdf.
        sample_generator: Callable bound to uniform_sample_generator.

    Methods:
        generate_samples(n, rng): Generate a batch through Proposal.
        uniform_pdf(x): Evaluate the normalized box density.
        uniform_log_pdf(x): Evaluate the box log-density directly.
        uniform_sample_generator(n, rng): Draw uniform samples in the box.
    """

    def __init__(self, bounds: list[tuple[float, float]]):
        """
        Configure the box used for sample generation and density evaluation.

        Args:
            bounds: One (lower, upper) pair per coordinate, with lower < upper.
                For example, [(-1, 1), (-2, 2)] defines a two-dimensional box.
        """
        self.bounds = bounds
        super().__init__(
            pdf=self.uniform_pdf,
            log_pdf=self.uniform_log_pdf,
            sample_generator=self.uniform_sample_generator
        )

    def uniform_pdf(self, x: np.ndarray) -> float:
        """
        Evaluate the normalized uniform density at one point.

        Args:
            x: Point of shape (dim,), matching the number of bounds.

        Returns:
            density: Reciprocal box volume inside the bounds, or zero outside.
        """
        for i in range(len(x)):
            if not (self.bounds[i][0] <= x[i] <= self.bounds[i][1]):
                return 0.0
        volume = np.prod([b[1] - b[0] for b in self.bounds])
        return 1.0 / volume

    def uniform_log_pdf(self, x: np.ndarray) -> float:
        """
        Evaluate the uniform log-density without first computing the density.

        Args:
            x: Point of shape (dim,), matching the number of bounds.

        Returns:
            log_density: Negative sum of log side lengths inside the box,
                or -np.inf outside its support.
        """
        for i in range(len(x)):
            if not (self.bounds[i][0] <= x[i] <= self.bounds[i][1]):
                return -np.inf
        return float(-np.sum(np.log([b[1] - b[0] for b in self.bounds])))

    def uniform_sample_generator(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Draw independent uniform samples within the coordinate bounds.

        Args:
            n: Number of samples to generate.
            rng: Optional NumPy generator. If None, create a new generator.

        Returns:
            samples: Array with shape (n, dim), one point per row.
        """
        if rng is None:
            rng = np.random.default_rng()
        samples = np.array([
            rng.uniform(low=b[0], high=b[1], size=n) for b in self.bounds
        ]).T
        return samples

class GaussianProposal(Proposal):
    """
    Multivariate Gaussian proposal with full-space support.

    Sample generation and both density forms use the same mean and
    positive-definite covariance. Full support alone does not guarantee
    finite importance-weight variance for a given target.

    Attributes:
        mean: Proposal mean vector with shape (dim,).
        cov: Positive-definite covariance matrix with shape (dim, dim).
        pdf: Callable bound to gaussian_pdf.
        log_pdf: Callable bound to gaussian_log_pdf.
        sample_generator: Callable bound to gaussian_sample_generator.

    Methods:
        generate_samples(n, rng): Generate a batch through Proposal.
        gaussian_pdf(x): Evaluate the normalized Gaussian density.
        gaussian_log_pdf(x): Evaluate the Gaussian log-density directly.
        gaussian_sample_generator(n, rng): Draw Gaussian samples.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Configure the Gaussian used for sample generation and density evaluation.

        Args:
            mean: Mean vector with shape (dim,).
            cov: Symmetric positive-definite covariance with shape (dim, dim).
            dist: Multivariate normal distribution object.
        """
        self.mean = mean
        self.cov = cov
        self.dist = multivariate_normal(mean=mean, cov=cov)
        super().__init__(
            pdf=self.gaussian_pdf,
            log_pdf=self.gaussian_log_pdf,
            sample_generator=self.gaussian_sample_generator
        )

    def gaussian_pdf(self, x: np.ndarray) -> float:
        """
        Evaluate the normalized Gaussian density at one point.

        Args:
            x: Point of shape (dim,), matching mean and cov.

        Returns:
            density: Exponential of the directly evaluated log-density.
                Very small values may underflow; use log_pdf for log-weights.
        """
        return float(self.dist.pdf(x))

    def gaussian_log_pdf(self, x: np.ndarray) -> float:
        """
        Evaluate the Gaussian log-density directly using a Cholesky factor.

        Args:
            x: Point of shape (dim,), matching mean and cov.

        Returns:
            log_density: Normalized log-density computed without taking the
                logarithm of a potentially underflowed PDF.
        """
        return float(self.dist.logpdf(x))

    def gaussian_sample_generator(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Draw independent samples from the configured Gaussian proposal.

        Args:
            n: Number of samples to generate.
            rng: Optional NumPy generator. If None, create a new generator.

        Returns:
            samples: Array with shape (n, dim), one point per row.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(mean=self.mean, cov=self.cov, size=n)