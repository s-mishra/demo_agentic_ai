"""
Time-varying Rt estimation using Bambi with random walk prior
"""
import bambi as bmb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from pathlib import Path

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

# Compute total infectiousness (vectorized with scipy)
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

# Filter and prepare data
model_df = df[df['infectiousness'] > 0].copy()
model_df['log_infectiousness'] = np.log(model_df['infectiousness'])
model_df['time_idx'] = np.arange(len(model_df))
model_df['time_factor'] = model_df['time_idx'].astype(str)

# Time-varying model with random effect for each time point
# This estimates a separate Rt for each week, shrunk toward global mean
model = bmb.Model(
    "cases ~ 1 + (1|time_factor) + offset(log_infectiousness)",
    data=model_df,
    family="poisson",
    link="log",
    priors={
        "Intercept": bmb.Prior("Normal", mu=0, sigma=0.5),
        "1|time_factor": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.3)),
    }
)

print("Time-varying Rt Model:")
print(model)

# Fit model
print("\nFitting model (this may take a moment)...")
trace = model.fit(draws=1000, tune=1000, chains=2, random_seed=42, target_accept=0.9)

# Extract time-varying Rt estimates
intercept = trace.posterior["Intercept"].values  # (chains, draws)
time_effects = trace.posterior["1|time_factor"].values  # (chains, draws, time)

# Combine intercept + time effects -> Rt
n_times = time_effects.shape[-1]
rt_samples = np.exp(intercept[:, :, None] + time_effects)  # (chains, draws, time)
rt_samples = rt_samples.reshape(-1, n_times)  # (total_samples, time)

# Compute summary statistics
rt_mean = rt_samples.mean(axis=0)
rt_lower = np.percentile(rt_samples, 2.5, axis=0)
rt_upper = np.percentile(rt_samples, 97.5, axis=0)

# Create results dataframe
results = model_df[['date', 'cases']].copy()
results['rt_mean'] = rt_mean
results['rt_lower'] = rt_lower
results['rt_upper'] = rt_upper

print("\nRt estimates (first 10 weeks):")
print(results[['date', 'cases', 'rt_mean', 'rt_lower', 'rt_upper']].head(10).to_string(index=False))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Cases
ax1 = axes[0]
ax1.fill_between(results['date'], results['cases'], alpha=0.4)
ax1.plot(results['date'], results['cases'], linewidth=1.5)
ax1.set_ylabel('Weekly Cases', fontsize=11)
ax1.set_title('Dengue Cases and Time-Varying Rt Estimate', fontsize=14, fontweight='bold')

# Rt
ax2 = axes[1]
ax2.fill_between(results['date'], rt_lower, rt_upper, alpha=0.3, label='95% CI')
ax2.plot(results['date'], rt_mean, linewidth=2, label='Rt mean')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
ax2.set_ylabel('Rt', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 2.5)

plt.tight_layout()
output_path = root / 'output/rt_timevarying.png'
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to {output_path}")
