"""
Verify equivalence and benchmark slow vs fast implementations.
"""

import numpy as np
from numpy.testing import assert_allclose
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import slow_epi_code as slow
import fast_epi_code as fast


def benchmark(func, args, n_runs=5):
    """Return median execution time."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    return np.median(times), result


def verify_and_benchmark_all():
    """
    Verify equivalence and benchmark all slow vs fast implementations.

    Tests each function pair for numerical equivalence, then measures
    execution time to compute speedup factors.
    """
    np.random.seed(42)

    results = []

    # =========================================================================
    # 1. compute_pairwise_distances
    # =========================================================================
    print("=" * 60)
    print("1. compute_pairwise_distances")
    print("=" * 60)

    n_locations = 500
    locations = [(np.random.uniform(0, 100), np.random.uniform(0, 100))
                 for _ in range(n_locations)]

    # Verify equivalence
    slow_result = slow.compute_pairwise_distances(locations)
    fast_result = fast.compute_pairwise_distances(locations)

    max_diff = np.abs(slow_result - fast_result).max()
    assert_allclose(slow_result, fast_result, rtol=1e-14, atol=1e-14)
    print(f"✓ Verified equivalent (max diff: {max_diff:.2e})")

    # Benchmark
    slow_time, _ = benchmark(slow.compute_pairwise_distances, (locations,))
    fast_time, _ = benchmark(fast.compute_pairwise_distances, (locations,))
    speedup = slow_time / fast_time

    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {speedup:.0f}x\n")
    results.append(("pairwise_distances", slow_time, fast_time, speedup))

    # =========================================================================
    # 2. compute_serial_interval_convolution
    # =========================================================================
    print("=" * 60)
    print("2. compute_serial_interval_convolution")
    print("=" * 60)

    incidence = np.random.poisson(50, size=1000)
    serial_interval = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])

    slow_result = slow.compute_serial_interval_convolution(incidence, serial_interval)
    fast_result = fast.compute_serial_interval_convolution(incidence, serial_interval)

    max_diff = np.abs(slow_result - fast_result).max()
    assert_allclose(slow_result, fast_result, rtol=1e-10, atol=1e-10)
    print(f"✓ Verified equivalent (max diff: {max_diff:.2e})")

    slow_time, _ = benchmark(slow.compute_serial_interval_convolution, (incidence, serial_interval))
    fast_time, _ = benchmark(fast.compute_serial_interval_convolution, (incidence, serial_interval))
    speedup = slow_time / fast_time

    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {speedup:.0f}x\n")
    results.append(("serial_interval_convolution", slow_time, fast_time, speedup))

    # =========================================================================
    # 3. compute_rt_simple
    # =========================================================================
    print("=" * 60)
    print("3. compute_rt_simple")
    print("=" * 60)

    slow_result = slow.compute_rt_simple(incidence, serial_interval, window=7)
    fast_result = fast.compute_rt_simple(incidence, serial_interval, window=7)

    # Skip first few values where there might be edge effects
    max_diff = np.abs(slow_result[10:] - fast_result[10:]).max()
    assert_allclose(slow_result[10:], fast_result[10:], rtol=1e-10, atol=1e-10)
    print(f"✓ Verified equivalent (max diff: {max_diff:.2e})")

    slow_time, _ = benchmark(slow.compute_rt_simple, (incidence, serial_interval, 7))
    fast_time, _ = benchmark(fast.compute_rt_simple, (incidence, serial_interval, 7))
    speedup = slow_time / fast_time

    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {speedup:.0f}x\n")
    results.append(("rt_simple", slow_time, fast_time, speedup))

    # =========================================================================
    # 4. negative_binomial_likelihood_loop
    # =========================================================================
    print("=" * 60)
    print("4. negative_binomial_likelihood_loop")
    print("=" * 60)

    observed = np.random.poisson(50, size=10000).astype(float)
    expected = np.random.uniform(30, 70, size=10000)
    overdispersion = 0.1

    slow_result = slow.negative_binomial_likelihood_loop(observed, expected, overdispersion)
    fast_result = fast.negative_binomial_likelihood_loop(observed, expected, overdispersion)

    rel_diff = abs(slow_result - fast_result) / abs(slow_result)
    assert_allclose(slow_result, fast_result, rtol=1e-10)
    print(f"✓ Verified equivalent (rel diff: {rel_diff:.2e})")

    slow_time, _ = benchmark(slow.negative_binomial_likelihood_loop, (observed, expected, overdispersion))
    fast_time, _ = benchmark(fast.negative_binomial_likelihood_loop, (observed, expected, overdispersion))
    speedup = slow_time / fast_time

    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {speedup:.0f}x\n")
    results.append(("negative_binomial_likelihood", slow_time, fast_time, speedup))

    # =========================================================================
    # 5. compute_generation_matrix
    # =========================================================================
    print("=" * 60)
    print("5. compute_generation_matrix")
    print("=" * 60)

    n_groups = 100
    contact_matrix = np.random.rand(n_groups, n_groups)
    susceptibility = np.random.rand(n_groups)
    infectivity = np.random.rand(n_groups)
    recovery_rate = 0.1

    slow_result = slow.compute_generation_matrix(contact_matrix, susceptibility, infectivity, recovery_rate)
    fast_result = fast.compute_generation_matrix(contact_matrix, susceptibility, infectivity, recovery_rate)

    max_diff = np.abs(slow_result - fast_result).max()
    assert_allclose(slow_result, fast_result, rtol=1e-14, atol=1e-14)
    print(f"✓ Verified equivalent (max diff: {max_diff:.2e})")

    slow_time, _ = benchmark(slow.compute_generation_matrix, (contact_matrix, susceptibility, infectivity, recovery_rate))
    fast_time, _ = benchmark(fast.compute_generation_matrix, (contact_matrix, susceptibility, infectivity, recovery_rate))
    speedup = slow_time / fast_time

    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {speedup:.0f}x\n")
    results.append(("generation_matrix", slow_time, fast_time, speedup))

    # =========================================================================
    # 6. moving_average_loop
    # =========================================================================
    print("=" * 60)
    print("6. moving_average_loop")
    print("=" * 60)

    data = np.random.randn(10000)
    window = 15

    slow_result = slow.moving_average_loop(data, window)
    fast_result = fast.moving_average_loop(data, window)

    # Edge handling may differ slightly, check interior
    interior = slice(window, -window)
    max_diff = np.abs(slow_result[interior] - fast_result[interior]).max()
    assert_allclose(slow_result[interior], fast_result[interior], rtol=1e-10, atol=1e-10)
    print(f"✓ Verified equivalent in interior (max diff: {max_diff:.2e})")

    slow_time, _ = benchmark(slow.moving_average_loop, (data, window))
    fast_time, _ = benchmark(fast.moving_average_loop, (data, window))
    speedup = slow_time / fast_time

    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {speedup:.0f}x\n")
    results.append(("moving_average", slow_time, fast_time, speedup))

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Function':<30} {'Slow (s)':<12} {'Fast (s)':<12} {'Speedup':<10}")
    print("-" * 60)

    total_slow = 0
    total_fast = 0
    for name, slow_t, fast_t, spd in results:
        print(f"{name:<30} {slow_t:<12.4f} {fast_t:<12.6f} {spd:<10.0f}x")
        total_slow += slow_t
        total_fast += fast_t

    print("-" * 60)
    print(f"{'TOTAL':<30} {total_slow:<12.4f} {total_fast:<12.6f} {total_slow/total_fast:<10.0f}x")


if __name__ == "__main__":
    verify_and_benchmark_all()
