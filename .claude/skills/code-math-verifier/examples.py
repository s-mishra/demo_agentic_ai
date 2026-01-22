"""
Example: Using the code-math-verifier skill

This file shows how Claude would apply the skill to optimize
a real piece of epidemiological code.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import convolve
from verify_utils import test_equivalence, benchmark, print_verification_report


# ============================================================
# EXAMPLE 1: Pairwise Distance Matrix
# ============================================================

# --- Original Code (User Provides This) ---
def distance_naive(locations):
    """
    User's original slow code.
    """
    n = len(locations)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x1, y1 = locations[i]
            x2, y2 = locations[j]
            distances[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distances


# --- Claude's Analysis ---
"""
Step 1: Understanding
- Input: n×2 array of (x, y) coordinates
- Output: n×n distance matrix
- Core operation: Euclidean distance between all pairs

Step 2: Mathematical Form
    d_{ij} = √[(x_i - x_j)² + (y_i - y_j)²]
    
    Or in vector form: D = ||X - X^T||₂ (pairwise)

Step 3: Optimization
This is exactly what scipy.spatial.distance.cdist computes.
"""


# --- Optimized Code (Claude Produces This) ---
def distance_optimized(locations):
    """
    Compute pairwise Euclidean distance matrix.
    
    Mathematical form:
        d_{ij} = √[(x_i - x_j)² + (y_i - y_j)²]
    
    Uses scipy.spatial.distance.cdist for O(n²) but C-optimized.
    """
    return cdist(locations, locations, metric='euclidean')


# ============================================================
# EXAMPLE 2: Serial Interval Convolution (Epi-specific)
# ============================================================

# --- Original Code ---
def renewal_naive(cases, serial_interval):
    """
    Compute force of infection using renewal equation.
    Naive nested loop implementation.
    """
    T = len(cases)
    max_si = len(serial_interval)
    lambda_t = np.zeros(T)
    
    for t in range(T):
        total = 0.0
        for s in range(min(t, max_si)):
            if t - s - 1 >= 0:
                total += cases[t - s - 1] * serial_interval[s]
        lambda_t[t] = total
    
    return lambda_t


# --- Claude's Analysis ---
"""
Step 1: Understanding
- Input: case time series I_t, serial interval weights w_s
- Output: force of infection λ_t
- Core operation: weighted sum of past cases

Step 2: Mathematical Form
    λ_t = Σ_{s=0}^{S} I_{t-s-1} · w_s
    
    This IS convolution: λ = I * w (with appropriate indexing)

Step 3: Optimization
Use numpy.convolve or scipy.signal.convolve.
Need to handle the indexing carefully (mode='full', then slice).
"""


# --- Optimized Code ---
def renewal_optimized(cases, serial_interval):
    """
    Compute force of infection using renewal equation.
    
    Mathematical form:
        λ_t = Σ_{s=0}^{S} I_{t-s-1} · w_s
        
    This is a discrete convolution, implemented via scipy.
    """
    # Convolve and take first T elements
    # Shift by 1 to match the t-s-1 indexing
    result = convolve(cases, serial_interval, mode='full')[:len(cases)]
    # Shift to account for the -1 in original indexing
    return np.concatenate([[0], result[:-1]])


# ============================================================
# EXAMPLE 3: Gaussian Kernel (RBF)
# ============================================================

# --- Original Code ---
def rbf_kernel_naive(X, length_scale=1.0):
    """
    Compute RBF kernel matrix. Naive loops.
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sq_dist = np.sum((X[i] - X[j])**2)
            K[i, j] = np.exp(-sq_dist / (2 * length_scale**2))
    return K


# --- Claude's Analysis ---
"""
Step 1: Understanding
- Input: n×d data matrix, length scale parameter
- Output: n×n kernel matrix
- Core operation: Gaussian (RBF) kernel

Step 2: Mathematical Form
    K_{ij} = exp(-||x_i - x_j||² / 2ℓ²)

Step 3: Optimization
Use cdist for squared distances, then apply exp.
"""


# --- Optimized Code ---
def rbf_kernel_optimized(X, length_scale=1.0):
    """
    Compute RBF (Gaussian) kernel matrix.
    
    Mathematical form:
        K_{ij} = exp(-||x_i - x_j||² / 2ℓ²)
    """
    sq_dist = cdist(X, X, metric='sqeuclidean')
    return np.exp(-sq_dist / (2 * length_scale**2))


# ============================================================
# Run Verification
# ============================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Pairwise Distance")
    print("="*60)
    
    result = test_equivalence(
        naive_func=distance_naive,
        optimized_func=distance_optimized,
        generate_inputs=lambda: {'locations': np.random.randn(50, 2)},
        n_tests=100
    )
    bench = benchmark(
        distance_naive, 
        distance_optimized,
        {'locations': np.random.randn(300, 2)}
    )
    print_verification_report(result, bench)
    
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Renewal Equation (Convolution)")
    print("="*60)
    
    def gen_renewal_inputs():
        T = np.random.randint(50, 200)
        cases = np.random.poisson(50, size=T).astype(float)
        si = np.random.dirichlet(np.ones(14))  # 14-day serial interval
        return {'cases': cases, 'serial_interval': si}
    
    result = test_equivalence(
        naive_func=renewal_naive,
        optimized_func=renewal_optimized,
        generate_inputs=gen_renewal_inputs,
        n_tests=100,
        rtol=1e-8  # Slightly looser due to different algorithms
    )
    bench = benchmark(
        renewal_naive,
        renewal_optimized,
        {'cases': np.random.poisson(50, size=500).astype(float),
         'serial_interval': np.random.dirichlet(np.ones(14))}
    )
    print_verification_report(result, bench)
    
    
    print("\n" + "="*60)
    print("EXAMPLE 3: RBF Kernel")
    print("="*60)
    
    result = test_equivalence(
        naive_func=rbf_kernel_naive,
        optimized_func=rbf_kernel_optimized,
        generate_inputs=lambda: {'X': np.random.randn(50, 5), 'length_scale': 1.0},
        n_tests=100
    )
    bench = benchmark(
        rbf_kernel_naive,
        rbf_kernel_optimized,
        {'X': np.random.randn(200, 5), 'length_scale': 1.0}
    )
    print_verification_report(result, bench)
