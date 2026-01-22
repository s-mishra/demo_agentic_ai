# Code-Math Verifier Skill

## Purpose

Guide Claude to analyze **arbitrary** code, understand its mathematical meaning, identify optimization opportunities, and verify equivalence. This skill is about **reasoning**, not pattern matching.

## When to Invoke

Use this skill when user asks to:
- "Optimize this code"
- "Vectorize these loops"
- "Speed up this function"
- "What does this code compute mathematically?"
- "Is this refactor equivalent?"

## Core Workflow

```
READ CODE → UNDERSTAND MATH → FIND OPTIMIZATION → VERIFY EQUIVALENCE → SHIP
```

---

## Step 1: Read and Understand the Code

Before doing anything, deeply understand what the code computes.

**Questions to answer:**
1. What are the inputs? (shapes, types, constraints)
2. What are the outputs? (shapes, types, meaning)
3. What is the core computation? (ignore boilerplate)
4. Are there any side effects?
5. What are the edge cases? (empty arrays, single elements, etc.)

**Do this:**
- Trace through with a small example mentally
- Identify the innermost operation
- Note any accumulation patterns (sum, product, max, etc.)

---

## Step 2: Extract the Mathematical Operation

Translate the code into mathematical notation. This is the critical step.

**Process:**
1. Identify loop variables and their ranges
2. Identify what gets computed at each iteration
3. Write the operation as a mathematical expression
4. Simplify if possible

**Template for nested loops:**
```
for i in range(n):
    for j in range(m):
        C[i,j] = f(A[i], B[j])

→ Mathematical form: C_{ij} = f(A_i, B_j)
```

**Template for reductions:**
```
result = 0
for i in range(n):
    result += f(x[i])

→ Mathematical form: result = Σᵢ f(xᵢ)
```

**Template for convolutions:**
```
for t in range(T):
    for s in range(S):
        y[t] += x[t-s] * h[s]

→ Mathematical form: yₜ = Σₛ x_{t-s} · hₛ  (this is convolution: y = x * h)
```

**Output the math as:**
- LaTeX: `$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$`
- Plain text: `d[i,j] = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2)`
- Description: "Pairwise Euclidean distance matrix"

---

## Step 3: Identify Optimization Opportunities

Now that you understand the math, ask: **Is there a library function that computes this?**

**Check these libraries (in order of preference):**

1. **NumPy broadcasting** — Can the operation be expressed without explicit loops?
   - Element-wise: `y[i] = f(x[i])` → `y = f(x)`
   - Outer operations: `C[i,j] = a[i] * b[j]` → `C = np.outer(a, b)`
   - Reductions: `sum(x[i] for i in range(n))` → `np.sum(x)`

2. **SciPy specialized functions:**
   - Distances: `scipy.spatial.distance.cdist`, `pdist`, `squareform`
   - Convolution: `scipy.signal.convolve`, `fftconvolve`
   - Linear algebra: `scipy.linalg.*`
   - Statistics: `scipy.stats.*`
   - Sparse matrices: `scipy.sparse.*`

3. **NumPy linear algebra:**
   - Matrix multiply: `A @ B` or `np.dot(A, B)`
   - Einsum for complex contractions: `np.einsum('ij,jk->ik', A, B)`

4. **Specialized libraries:**
   - `numba.jit` for loops that can't be vectorized
   - `jax` for autodiff + JIT
   - `torch` for GPU acceleration

**Red flags that suggest optimization is possible:**
- Nested `for` loops over array indices
- `append` inside loops (use pre-allocation)
- Repeated computation inside loops (hoist it out)
- Python `sum()` or `math.sqrt()` instead of `np.sum()`, `np.sqrt()`

**When NOT to optimize:**
- Code is already vectorized
- Loop body has complex control flow (early exits, conditionals)
- Code is I/O bound, not compute bound
- Readability matters more than speed

---

## Step 4: Write the Optimized Version

When writing optimized code:

1. **Preserve the interface** — Same inputs, same outputs
2. **Add docstring** — Include the mathematical formula
3. **Handle edge cases** — Empty arrays, single elements
4. **Match dtypes** — Don't accidentally convert float64 to float32

**Template:**
```python
def compute_optimized(inputs):
    """
    Compute [description].
    
    Mathematical form:
        [LaTeX or plain math notation]
    
    Parameters
    ----------
    inputs : array-like
        [description]
    
    Returns
    -------
    result : ndarray
        [description]
    
    Notes
    -----
    Equivalent to naive loop implementation but O(n) instead of O(n²).
    Verified equivalent via numerical testing.
    """
    # Implementation using vectorized operations
    return result
```

---

## Step 5: Verify Equivalence

**CRITICAL: Never ship optimized code without verification.**

### 5.1 Symbolic Reasoning (Quick Check)

Ask yourself:
- Does the optimized version compute the same mathematical expression?
- Are there any edge cases where they might differ?
- Are there numerical precision differences? (e.g., order of summation)

### 5.2 Numerical Testing (Authoritative Check)

Generate test cases and compare outputs:

```python
import numpy as np
from numpy.testing import assert_allclose

def test_equivalence():
    # Test with multiple random inputs
    for seed in range(100):
        np.random.seed(seed)
        
        # Generate random inputs matching expected shapes/ranges
        x = np.random.randn(100)
        
        # Run both implementations
        naive_result = compute_naive(x)
        optimized_result = compute_optimized(x)
        
        # Check equivalence with tolerance
        assert_allclose(
            naive_result, 
            optimized_result,
            rtol=1e-10,  # relative tolerance
            atol=1e-12,  # absolute tolerance
            err_msg=f"Failed at seed {seed}"
        )

# Also test edge cases
def test_edge_cases():
    # Empty input
    assert_allclose(compute_naive([]), compute_optimized([]))
    
    # Single element
    assert_allclose(compute_naive([1.0]), compute_optimized([1.0]))
    
    # Large values
    x = np.array([1e10, 1e-10])
    assert_allclose(compute_naive(x), compute_optimized(x), rtol=1e-8)
```

### 5.3 Tolerance Guidelines

| Scenario | rtol | atol |
|----------|------|------|
| Exact same algorithm | 1e-15 | 1e-15 |
| Different summation order | 1e-10 | 1e-12 |
| Different algorithm (e.g., FFT vs direct) | 1e-8 | 1e-10 |
| Single vs double precision | 1e-5 | 1e-6 |

### 5.4 Report Results

After verification, report:
- Number of test cases run
- Maximum absolute difference observed
- Maximum relative difference observed
- Any edge cases that needed special handling

---

## Step 6: Benchmark (Optional but Recommended)

Show the speedup:

```python
import time
import numpy as np

def benchmark(func, inputs, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*inputs)
        times.append(time.perf_counter() - start)
    return np.median(times)

naive_time = benchmark(compute_naive, test_inputs)
optimized_time = benchmark(compute_optimized, test_inputs)

print(f"Naive: {naive_time:.4f}s")
print(f"Optimized: {optimized_time:.4f}s")
print(f"Speedup: {naive_time / optimized_time:.1f}x")
```

---

## Example Interaction

**User:** "Optimize this code"
```python
def slow_func(x, y):
    n = len(x)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = np.exp(-0.5 * ((x[i] - x[j])**2 + (y[i] - y[j])**2))
    return result
```

**Claude's response following this skill:**

### Step 1: Understanding
- Inputs: Two 1D arrays `x` and `y` of length `n`
- Output: `n × n` matrix
- Core computation: Gaussian kernel based on squared Euclidean distance

### Step 2: Mathematical Form
$$K_{ij} = \exp\left(-\frac{1}{2}\left[(x_i - x_j)^2 + (y_i - y_j)^2\right]\right)$$

This is a **radial basis function (RBF) kernel** or **Gaussian kernel matrix**.

### Step 3: Optimization
The squared distance computation can use `scipy.spatial.distance.cdist` or broadcasting:

```python
from scipy.spatial.distance import cdist

def fast_func(x, y):
    """
    Compute Gaussian kernel matrix.
    
    K_{ij} = exp(-0.5 * ||p_i - p_j||²)
    
    where p_i = (x_i, y_i)
    """
    points = np.column_stack([x, y])
    sq_distances = cdist(points, points, metric='sqeuclidean')
    return np.exp(-0.5 * sq_distances)
```

### Step 4: Verification
```python
# Ran 100 random tests
# Max absolute difference: 2.2e-16
# Max relative difference: 1.1e-15
# All edge cases passed
```

### Step 5: Benchmark
```
Naive: 0.847s (n=500)
Optimized: 0.003s
Speedup: 282x
```

---

## What This Skill Is NOT

This skill does **NOT**:
- Pattern match with regex
- Have a fixed list of "supported" operations
- Replace your understanding with lookup tables

This skill **IS**:
- A reasoning framework for Claude
- A checklist to ensure nothing is missed
- A guide for verification rigor

---

## What NOT To Do

1. **Don't pattern match blindly** — Understand the math first
2. **Don't optimize prematurely** — Profile first, then optimize hotspots
3. **Don't skip verification** — Always test equivalence numerically
4. **Don't sacrifice readability** — Sometimes the naive version is better
5. **Don't ignore edge cases** — Empty arrays break many vectorized ops

---

## Summary Checklist

Before declaring "optimization complete":

- [ ] I understand what the code computes mathematically
- [ ] I can write the operation in mathematical notation
- [ ] The optimized version computes the same expression
- [ ] I ran 100+ random test cases
- [ ] Maximum difference is within acceptable tolerance
- [ ] Edge cases are handled
- [ ] I benchmarked and confirmed speedup
- [ ] The optimized code has a clear docstring with the math
