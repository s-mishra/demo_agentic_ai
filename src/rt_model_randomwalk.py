"""
Time-varying Rt estimation using PyMC with Gaussian Random Walk
"""
import pymc as pm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from pathlib import Path
import arviz as az

sns.set_theme(style="whitegrid", palette="viridis")

# Load data
root = Path(__file__).parent.parent
df = pd.read_csv(root / 'data/dengue_cases.csv', parse_dates=['date'])

# Serial interval distribution: Gamma(mean=14.5, sd=4.5)
si_mean, si_sd = 14.5, 4.5
si_shape = (si_mean / si_sd) ** 2
si_scale = si_sd ** 2 / si_mean
si_weights = gamma_dist.pdf(np.arange(1, 22) * 7, a=si_shape, scale=si_scale)
si_weights = si_weights / si_weights.sum()

# Compute total infectiousness (vectorized)
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
        lookback = min(t, len(weights))
        infectiousness[t] = np.dot(cases[t-lookback:t][::-1], weights[:lookback])
    return infectiousness

df['infectiousness'] = compute_infectiousness(df['cases'].values, si_weights)

# Filter to valid rows
model_df = df[df['infectiousness'] > 0].copy().reset_index(drop=True)
n_times = len(model_df)
cases = model_df['cases'].values
infectiousness = model_df['infectiousness'].values

print(f"Modeling {n_times} time points")

# PyMC model with Gaussian Random Walk for log(Rt)
with pm.Model() as rt_model:
    # Prior on initial log(Rt) - LogNormal(0, 0.5) means log(Rt) ~ Normal(0, 0.5)
    log_rt_init = pm.Normal("log_rt_init", mu=0, sigma=0.5)

    # Random walk innovation standard deviation
    rw_sigma = pm.HalfNormal("rw_sigma", sigma=0.1)

    # Gaussian random walk for log(Rt) after initial value
    log_rt_innovations = pm.Normal("log_rt_innovations", mu=0, sigma=rw_sigma, shape=n_times - 1)

    # Build log(Rt) trajectory: cumulative sum of innovations
    log_rt = pm.Deterministic(
        "log_rt",
        pm.math.concatenate([[log_rt_init], log_rt_init + pm.math.cumsum(log_rt_innovations)])
    )

    # Rt = exp(log_rt)
    rt = pm.Deterministic("rt", pm.math.exp(log_rt))

    # Expected cases from renewal equation
    expected_cases = rt * infectiousness

    # Observation model: Negative Binomial for overdispersion
    phi = pm.HalfNormal("phi", sigma=5)  # overdispersion
    pm.NegativeBinomial("cases_obs", mu=expected_cases, alpha=phi, observed=cases)

print("\nModel structure:")
print(rt_model)

# Fit model
print("\nFitting model...")
with rt_model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=42,
        target_accept=0.95,
        return_inferencedata=True
    )

# Extract Rt estimates
rt_samples = trace.posterior["rt"].values.reshape(-1, n_times)
rt_mean = rt_samples.mean(axis=0)
rt_lower = np.percentile(rt_samples, 2.5, axis=0)
rt_upper = np.percentile(rt_samples, 97.5, axis=0)

# Results dataframe
results = model_df[['date', 'cases']].copy()
results['rt_mean'] = rt_mean
results['rt_lower'] = rt_lower
results['rt_upper'] = rt_upper

print("\nRt estimates:")
print(results[['date', 'cases', 'rt_mean', 'rt_lower', 'rt_upper']].to_string(index=False))

# Diagnostics
print("\nModel diagnostics:")
print(az.summary(trace, var_names=["log_rt_init", "rw_sigma", "phi"]))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
colors = sns.color_palette("viridis", 3)

# Cases
ax1 = axes[0]
ax1.fill_between(results['date'], results['cases'], alpha=0.4, color=colors[0])
ax1.plot(results['date'], results['cases'], linewidth=1.5, color=colors[0])
ax1.set_ylabel('Weekly Cases', fontsize=11)
ax1.set_title('Dengue Cases and Time-Varying Rt (Random Walk Model)', fontsize=14, fontweight='bold')

# Rt
ax2 = axes[1]
ax2.fill_between(results['date'], rt_lower, rt_upper, alpha=0.3, color=colors[1], label='95% CI')
ax2.plot(results['date'], rt_mean, linewidth=2, color=colors[1], label='Rt mean')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Rt = 1')
ax2.set_ylabel('Effective Reproduction Number (Rt)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 2.5)

plt.tight_layout()
output_path = root / 'output/rt_randomwalk.png'
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to {output_path}")

# Save results
results.to_csv(root / 'output/rt_estimates.csv', index=False)
print(f"Results saved to output/rt_estimates.csv")
