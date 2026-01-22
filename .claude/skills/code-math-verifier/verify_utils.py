"""
Verification Utilities for Code Optimization

Simple helper functions for testing equivalence between implementations.
The actual understanding of math is done by Claude, not this code.
"""

import numpy as np
from numpy.testing import assert_allclose
import time
from typing import Callable, Dict, Any, List
from dataclasses import dataclass


@dataclass
class EquivalenceResult:
    """Result of equivalence testing."""
    passed: bool
    n_tests: int
    max_abs_diff: float
    max_rel_diff: float
    failed_seeds: List[int]
    notes: List[str]


@dataclass 
class BenchmarkResult:
    """Result of benchmarking."""
    naive_time: float
    optimized_time: float
    speedup: float
    input_size: int


def test_equivalence(
    naive_func: Callable,
    optimized_func: Callable,
    generate_inputs: Callable,
    n_tests: int = 100,
    rtol: float = 1e-10,
    atol: float = 1e-12
) -> EquivalenceResult:
    """
    Test that two functions produce equivalent outputs.
    
    Parameters
    ----------
    naive_func : Callable
        The reference (naive) implementation
    optimized_func : Callable
        The optimized implementation to verify
    generate_inputs : Callable
        Function that returns a dict of inputs when called.
        Will be called once per test with the seed set.
        Example: lambda: {'x': np.random.randn(100)}
    n_tests : int
        Number of random tests to run
    rtol, atol : float
        Tolerance for np.allclose
        
    Returns
    -------
    EquivalenceResult
        Contains pass/fail, max differences, failed cases
        
    Example
    -------
    >>> result = test_equivalence(
    ...     naive_func=slow_distance,
    ...     optimized_func=fast_distance,
    ...     generate_inputs=lambda: {'points': np.random.randn(50, 2)},
    ...     n_tests=100
    ... )
    >>> print(f"Passed: {result.passed}, Max diff: {result.max_abs_diff}")
    """
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    failed_seeds = []
    notes = []
    
    for seed in range(n_tests):
        np.random.seed(seed)
        inputs = generate_inputs()
        
        try:
            naive_out = naive_func(**inputs)
            opt_out = optimized_func(**inputs)
            
            # Convert to arrays for comparison
            naive_arr = np.asarray(naive_out)
            opt_arr = np.asarray(opt_out)
            
            # Check shapes match
            if naive_arr.shape != opt_arr.shape:
                failed_seeds.append(seed)
                notes.append(f"Seed {seed}: Shape mismatch {naive_arr.shape} vs {opt_arr.shape}")
                continue
            
            # Compute differences
            abs_diff = np.abs(naive_arr - opt_arr)
            max_abs_diff = max(max_abs_diff, np.max(abs_diff))
            
            # Relative diff (avoid div by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = abs_diff / (np.abs(naive_arr) + 1e-15)
                rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
            max_rel_diff = max(max_rel_diff, np.max(rel_diff))
            
            # Check if within tolerance
            if not np.allclose(naive_arr, opt_arr, rtol=rtol, atol=atol):
                failed_seeds.append(seed)
                notes.append(f"Seed {seed}: Max abs diff = {np.max(abs_diff):.2e}")
                
        except Exception as e:
            failed_seeds.append(seed)
            notes.append(f"Seed {seed}: Exception - {str(e)}")
    
    passed = len(failed_seeds) == 0
    
    if passed:
        notes.append(f"All {n_tests} tests passed")
        notes.append(f"Max absolute difference: {max_abs_diff:.2e}")
        notes.append(f"Max relative difference: {max_rel_diff:.2e}")
    
    return EquivalenceResult(
        passed=passed,
        n_tests=n_tests,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        failed_seeds=failed_seeds,
        notes=notes
    )


def benchmark(
    naive_func: Callable,
    optimized_func: Callable,
    inputs: Dict[str, Any],
    n_runs: int = 10,
    warmup: int = 2
) -> BenchmarkResult:
    """
    Benchmark naive vs optimized implementation.
    
    Parameters
    ----------
    naive_func, optimized_func : Callable
        Functions to benchmark
    inputs : Dict
        Input arguments to pass to both functions
    n_runs : int
        Number of timed runs (median is reported)
    warmup : int
        Number of warmup runs before timing
        
    Returns
    -------
    BenchmarkResult
        Contains times and speedup factor
        
    Example
    -------
    >>> inputs = {'points': np.random.randn(500, 2)}
    >>> result = benchmark(slow_distance, fast_distance, inputs)
    >>> print(f"Speedup: {result.speedup:.1f}x")
    """
    # Warmup
    for _ in range(warmup):
        naive_func(**inputs)
        optimized_func(**inputs)
    
    # Time naive
    naive_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        naive_func(**inputs)
        naive_times.append(time.perf_counter() - start)
    
    # Time optimized
    opt_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        optimized_func(**inputs)
        opt_times.append(time.perf_counter() - start)
    
    naive_time = np.median(naive_times)
    opt_time = np.median(opt_times)
    
    # Estimate input size
    input_size = 0
    for v in inputs.values():
        if hasattr(v, 'size'):
            input_size = max(input_size, v.size)
        elif hasattr(v, '__len__'):
            input_size = max(input_size, len(v))
    
    return BenchmarkResult(
        naive_time=naive_time,
        optimized_time=opt_time,
        speedup=naive_time / opt_time if opt_time > 0 else float('inf'),
        input_size=input_size
    )


def generate_pytest_file(
    func_name: str,
    naive_code: str,
    optimized_code: str,
    input_generator_code: str,
    output_path: str = None
) -> str:
    """
    Generate a pytest file for equivalence testing.
    
    Parameters
    ----------
    func_name : str
        Name for the test (e.g., 'pairwise_distance')
    naive_code : str
        Python code defining the naive function
    optimized_code : str  
        Python code defining the optimized function
    input_generator_code : str
        Python code that generates test inputs
        Example: "{'x': np.random.randn(100)}"
    output_path : str, optional
        Path to write the test file
        
    Returns
    -------
    str
        The generated test code
    """
    
    test_code = f'''"""
Equivalence tests for {func_name}
Auto-generated by code-math-verifier skill
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ============================================================
# Naive Implementation
# ============================================================

{naive_code}


# ============================================================
# Optimized Implementation  
# ============================================================

{optimized_code}


# ============================================================
# Tests
# ============================================================

class TestEquivalence:
    """Test equivalence between naive and optimized."""
    
    RTOL = 1e-10
    ATOL = 1e-12
    
    def generate_inputs(self):
        """Generate random test inputs."""
        return {input_generator_code}
    
    @pytest.mark.parametrize("seed", range(100))
    def test_random_inputs(self, seed):
        """Test with 100 different random inputs."""
        np.random.seed(seed)
        inputs = self.generate_inputs()
        
        naive_result = {func_name}_naive(**inputs)
        opt_result = {func_name}_optimized(**inputs)
        
        assert_allclose(
            naive_result, 
            opt_result,
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg=f"Failed at seed {{seed}}"
        )
    
    def test_empty_input(self):
        """Test edge case: empty/minimal input."""
        # TODO: Customize for your function
        pass
    
    def test_single_element(self):
        """Test edge case: single element."""
        # TODO: Customize for your function
        pass


class TestPerformance:
    """Benchmark naive vs optimized."""
    
    def test_speedup(self):
        """Verify optimized is faster."""
        import time
        
        np.random.seed(42)
        # Use larger inputs for timing
        inputs = {input_generator_code.replace('100', '500').replace('50', '250')}
        
        # Time naive
        start = time.perf_counter()
        for _ in range(3):
            {func_name}_naive(**inputs)
        naive_time = (time.perf_counter() - start) / 3
        
        # Time optimized
        start = time.perf_counter()
        for _ in range(3):
            {func_name}_optimized(**inputs)
        opt_time = (time.perf_counter() - start) / 3
        
        speedup = naive_time / opt_time
        print(f"\\nNaive: {{naive_time:.4f}}s")
        print(f"Optimized: {{opt_time:.4f}}s")
        print(f"Speedup: {{speedup:.1f}}x")
        
        # Should be at least 2x faster
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {{speedup:.1f}}x"
'''
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(test_code)
        print(f"Wrote test file to: {output_path}")
    
    return test_code


def print_verification_report(result: EquivalenceResult, bench: BenchmarkResult = None):
    """Print a formatted verification report."""
    
    print("=" * 60)
    print("EQUIVALENCE VERIFICATION REPORT")
    print("=" * 60)
    
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"\nStatus: {status}")
    print(f"Tests run: {result.n_tests}")
    print(f"Tests failed: {len(result.failed_seeds)}")
    print(f"\nMax absolute difference: {result.max_abs_diff:.2e}")
    print(f"Max relative difference: {result.max_rel_diff:.2e}")
    
    if result.failed_seeds:
        print(f"\nFailed seeds: {result.failed_seeds[:10]}{'...' if len(result.failed_seeds) > 10 else ''}")
    
    if bench:
        print(f"\n--- Performance ---")
        print(f"Naive time: {bench.naive_time:.4f}s")
        print(f"Optimized time: {bench.optimized_time:.4f}s")
        print(f"Speedup: {bench.speedup:.1f}x")
        print(f"Input size: {bench.input_size}")
    
    print("\n" + "=" * 60)


# ============================================================
# Example Usage
# ============================================================

if __name__ == '__main__':
    # Example: Verify pairwise distance optimization
    
    def distance_naive(points):
        n = len(points)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))
        return D
    
    def distance_optimized(points):
        from scipy.spatial.distance import cdist
        return cdist(points, points)
    
    # Test equivalence
    result = test_equivalence(
        naive_func=distance_naive,
        optimized_func=distance_optimized,
        generate_inputs=lambda: {'points': np.random.randn(50, 2)},
        n_tests=100
    )
    
    # Benchmark
    bench = benchmark(
        naive_func=distance_naive,
        optimized_func=distance_optimized,
        inputs={'points': np.random.randn(300, 2)}
    )
    
    # Report
    print_verification_report(result, bench)
