"""
Optimized implementations of common epidemiological computations.
Each function is mathematically equivalent to slow_epi_code.py but uses
vectorized operations for significant speedups.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d
from scipy.special import gammaln
from typing import List, Tuple


def compute_pairwise_distances(locations: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all locations.

    Mathematical form:
        d_{ij} = ||p_i - p_j|| = √[(x_i - x_j)² + (y_i - y_j)²]

    Optimization: scipy.spatial.distance.cdist
    """
    points = np.array(locations)
    return cdist(points, points, metric='euclidean')


def compute_serial_interval_convolution(incidence: np.ndarray,
                                         serial_interval: np.ndarray) -> np.ndarray:
    """
    Compute expected secondary infections using the renewal equation.

    Mathematical form:
        Λ_t = Σ_{s=1}^{t} I_{t-s} · w_s  (discrete convolution)

    Optimization: np.convolve
    """
    # Convolution computes sum of I[t-s] * w[s]
    # We need to shift by 1 since original indexes t-s-1
    padded = np.concatenate([[0], incidence[:-1]])
    convolved = np.convolve(padded, serial_interval, mode='full')[:len(incidence)]
    return convolved


def compute_rt_simple(incidence: np.ndarray,
                      serial_interval: np.ndarray,
                      window: int = 7) -> np.ndarray:
    """
    Estimate instantaneous reproduction number.

    Mathematical form:
        R_t = (Σ_{τ=t-w}^{t-1} I_τ) / (Σ_{τ=t-w}^{t-1} Λ_τ)

    where Λ_τ = Σ_s I_{τ-s-1} · w_s

    Optimization: convolution + cumsum for rolling windows
    """
    T = len(incidence)
    rt_estimates = np.zeros(T)

    # Compute lambda (expected) for all time points via convolution
    lambda_t = compute_serial_interval_convolution(incidence, serial_interval)

    # Use cumsum for efficient rolling sums
    cum_incidence = np.cumsum(incidence)
    cum_lambda = np.cumsum(lambda_t)

    # Rolling sum = cumsum[t] - cumsum[t-window]
    for t in range(window, T):
        observed = cum_incidence[t-1] - (cum_incidence[t-window-1] if t > window else 0)
        expected = cum_lambda[t-1] - (cum_lambda[t-window-1] if t > window else 0)
        if expected > 0:
            rt_estimates[t] = observed / expected

    return rt_estimates


def negative_binomial_likelihood_loop(observed: np.ndarray,
                                       expected: np.ndarray,
                                       overdispersion: float) -> float:
    """
    Compute negative binomial log-likelihood for epidemic model fitting.

    Mathematical form:
        ℓ = Σ_i [log Γ(y_i + r) - log Γ(y_i + 1) - log Γ(r)
                 + r·log(p_i) + y_i·log(1 - p_i)]

    where r = 1/overdispersion, p_i = r/(r + μ_i)

    Optimization: vectorized gammaln and numpy operations
    """
    r = 1.0 / overdispersion
    p = r / (r + expected)

    log_lik = (gammaln(observed + r) - gammaln(observed + 1) - gammaln(r) +
               r * np.log(p) + observed * np.log(1 - p))

    return np.sum(log_lik)


def compute_generation_matrix(contact_matrix: np.ndarray,
                               susceptibility: np.ndarray,
                               infectivity: np.ndarray,
                               recovery_rate: float) -> np.ndarray:
    """
    Compute next-generation matrix for age-structured epidemic model.

    Mathematical form:
        K_{ij} = (s_i · C_{ij} · f_j) / γ

    Optimization: numpy broadcasting
    """
    return (susceptibility[:, None] * contact_matrix * infectivity[None, :]) / recovery_rate


def moving_average_loop(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute centered moving average for smoothing epidemic curves.

    Mathematical form:
        MA_i = (1/|W_i|) Σ_{j ∈ W_i} x_j

    where W_i is the window centered at i (handles edges)

    Optimization: scipy.ndimage.uniform_filter1d
    """
    return uniform_filter1d(data.astype(float), size=window, mode='nearest')
