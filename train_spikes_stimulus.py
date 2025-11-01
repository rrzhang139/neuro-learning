import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
dt = 0.001  # Time step (s)
T = 10    # Total time (s)
time = np.arange(0, T, dt)
n_time_bins = len(time)

# Create a random stimulus (e.g., Gaussian noise)
stimulus = np.random.randn(n_time_bins)

# Simulate a spike train (for demonstration, let's say spikes occur at specific times)
spike_times = np.array([1.5, 3.2, 5.8, 7.1, 8.9])  # seconds
spikes = np.zeros(n_time_bins)
for s_time in spike_times:
    spike_idx = int(s_time / dt)
    if 0 <= spike_idx < n_time_bins:
        spikes[spike_idx] = 1

# Next step: Now that we have our synthetic data, we need to define the parameters
# for our STA calculation, such as the time window before a spike to consider.

sta_window_s = 0.5
sta_n_bins = int(sta_window_s / dt)
sta_time_lags = np.arange(-sta_window_s, 0, dt)

# Next step: With our parameters defined, we can now proceed to implement the STA calculation itself.
sta = np.zeros(sta_n_bins)
spike_indices = np.nonzero(spikes)[0]
for spike_idx in spike_indices:
    if spike_idx >= sta_n_bins:
        sta += stimulus[spike_idx - sta_n_bins: spike_idx]

sta /= len(spike_indices)

# Now calculate the estimated firing rates
estimated_firing_rates = np.zeros_like(spikes)
for t in range(n_time_bins):
    for k in range(sta_n_bins):
        if t >= k:
            estimated_firing_rates[t] += sta[k] * stimulus[t - k]

print(f"estimated firing rates: {estimated_firing_rates}")
