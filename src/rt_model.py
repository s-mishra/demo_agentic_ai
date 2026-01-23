"""
Bayesian Rt estimation using Bambi (per CLAUDE.md conventions)
"""
import bambi as bmb
import pandas as pd
import numpy as np
from scipy.stats import gamma as gamma_dist
from pathlib import Path

# Load data
root = Path(__file__).parent.parent
df = pd.read_csv(root / 'data/dengue_cases.csv', parse_dates=['date'])

# Serial interval distribution: Gamma(mean=14.5, sd=4.5)
si_mean, si_sd = 14.5, 4.5
si_shape = (si_mean / si_sd) ** 2
si_scale = si_sd ** 2 / si_mean
si_weights = gamma_dist.pdf(np.arange(1, 22) * 7, a=si_shape, scale=si_scale)  # weekly
si_weights = si_weights / si_weights.sum()

# Compute total infectiousness (weighted sum of past cases)
def compute_infectiousness(cases, weights):
    """
    Compute total infectiousness using the renewal equation.

    Parameters
    ----------
    cases : array-like
        Time series of observed case counts.
    weights : array-like
        Serial interval distribution weights.

    Returns
    -------
    np.ndarray
        Infectiousness at each time point, computed as weighted sum of past cases.
    """
    n = len(cases)
    infectiousness = np.zeros(n)
    for t in range(1, n):
        for s, w in enumerate(weights):
            if t - s - 1 >= 0:
                infectiousness[t] += cases[t - s - 1] * w
    return infectiousness

df['infectiousness'] = compute_infectiousness(df['cases'].values, si_weights)

# Filter to rows with positive infectiousness
model_df = df[df['infectiousness'] > 0].copy()
model_df['log_infectiousness'] = np.log(model_df['infectiousness'])

# Bambi model: cases ~ Poisson with log-link
# log(E[cases]) = log(Rt) + log(infectiousness)
# Include log_infectiousness with fixed coefficient of 1 (offset)
model = bmb.Model(
    "cases ~ 1 + offset(log_infectiousness)",
    data=model_df,
    family="poisson",
    link="log",
    priors={"Intercept": bmb.Prior("LogNormal", mu=0, sigma=0.5)}
)

print("Model specification:")
print(model)

# Fit model
print("\nFitting model...")
trace = model.fit(draws=1000, tune=500, chains=2, random_seed=42)

# Extract Rt estimate (exp of intercept)
intercept_samples = trace.posterior["Intercept"].values.flatten()
rt_samples = np.exp(intercept_samples)

print(f"\nRt estimate (constant model):")
print(f"  Mean: {rt_samples.mean():.3f}")
print(f"  95% CI: [{np.percentile(rt_samples, 2.5):.3f}, {np.percentile(rt_samples, 97.5):.3f}]")
