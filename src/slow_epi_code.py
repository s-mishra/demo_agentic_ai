"""
Naive implementations of common epidemiological computations.
These are mathematically correct but computationally inefficient.
Used for demonstrating code optimization capabilities.
"""

import numpy as np
from typing import List, Tuple

def compute_pairwise_distances(locations: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all locations.
    Used for spatial clustering in outbreak investigation.
    
    SLOW: O(n²) nested loops with Python overhead
    """
    n = len(locations)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x1, y1 = locations[i]
            x2, y2 = locations[j]
            distances[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return distances


def compute_serial_interval_convolution(incidence: np.ndarray, 
                                         serial_interval: np.ndarray) -> np.ndarray:
    """
    Compute the expected number of secondary infections using
    the renewal equation: Lambda_t = sum_{s=1}^{t} I_{t-s} * w_s
    
    SLOW: Explicit loop over time and serial interval
    """
    T = len(incidence)
    max_si = len(serial_interval)
    lambda_t = np.zeros(T)
    
    for t in range(T):
        total = 0.0
        for s in range(min(t, max_si)):
            if t - s - 1 >= 0:
                total += incidence[t - s - 1] * serial_interval[s]
        lambda_t[t] = total
    
    return lambda_t


def compute_rt_simple(incidence: np.ndarray, 
                      serial_interval: np.ndarray,
                      window: int = 7) -> np.ndarray:
    """
    Estimate instantaneous reproduction number using ratio of
    observed to expected cases (Wallinga-Teunis style).
    
    SLOW: Triple nested loop
    """
    T = len(incidence)
    rt_estimates = np.zeros(T)
    
    for t in range(window, T):
        # Compute expected infections in window
        expected = 0.0
        for tau in range(t - window, t):
            for s in range(len(serial_interval)):
                if tau - s - 1 >= 0:
                    expected += incidence[tau - s - 1] * serial_interval[s]
        
        # Compute observed in window
        observed = 0.0
        for tau in range(t - window, t):
            observed += incidence[tau]
        
        if expected > 0:
            rt_estimates[t] = observed / expected
    
    return rt_estimates


def negative_binomial_likelihood_loop(observed: np.ndarray,
                                       expected: np.ndarray,
                                       overdispersion: float) -> float:
    """
    Compute negative binomial log-likelihood for epidemic model fitting.
    
    SLOW: Loop over all observations with repeated scipy calls
    """
    from scipy.special import gammaln
    
    log_lik = 0.0
    r = 1.0 / overdispersion  # size parameter
    
    for i in range(len(observed)):
        y = observed[i]
        mu = expected[i]
        p = r / (r + mu)
        
        # Log-likelihood for single observation
        ll_i = (gammaln(y + r) - gammaln(y + 1) - gammaln(r) +
                r * np.log(p) + y * np.log(1 - p))
        log_lik += ll_i
    
    return log_lik


def compute_generation_matrix(contact_matrix: np.ndarray,
                               susceptibility: np.ndarray,
                               infectivity: np.ndarray,
                               recovery_rate: float) -> np.ndarray:
    """
    Compute next-generation matrix for age-structured epidemic model.
    K_ij = (susceptibility_i * contact_ij * infectivity_j) / recovery_rate
    
    SLOW: Explicit loops instead of broadcasting
    """
    n_groups = len(susceptibility)
    K = np.zeros((n_groups, n_groups))
    
    for i in range(n_groups):
        for j in range(n_groups):
            K[i, j] = (susceptibility[i] * contact_matrix[i, j] * 
                       infectivity[j]) / recovery_rate
    
    return K


def moving_average_loop(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute centered moving average for smoothing epidemic curves.
    
    SLOW: Explicit loop with slicing
    """
    n = len(data)
    result = np.zeros(n)
    half_window = window // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        result[i] = 0.0
        count = 0
        for j in range(start, end):
            result[i] += data[j]
            count += 1
        result[i] /= count
    
    return result


# Example usage showing the performance problem
if __name__ == "__main__":
    import time
    
    # Generate test data
    np.random.seed(42)
    n_locations = 500
    locations = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) 
                 for _ in range(n_locations)]
    
    incidence = np.random.poisson(50, size=365)
    serial_interval = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
    
    # Time the slow implementations
    print("Timing naive implementations...")
    
    start = time.time()
    distances = compute_pairwise_distances(locations)
    print(f"Pairwise distances ({n_locations} locations): {time.time() - start:.3f}s")
    
    start = time.time()
    lambda_t = compute_serial_interval_convolution(incidence, serial_interval)
    print(f"Serial interval convolution (365 days): {time.time() - start:.3f}s")
    
    start = time.time()
    rt = compute_rt_simple(incidence, serial_interval)
    print(f"Rt estimation (365 days): {time.time() - start:.3f}s")
