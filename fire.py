import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters for a leaky integrate-and-fire neuron
T_TOTAL_MS = 1000.0  # Total simulation time (ms)
DT_MS = 0.1         # Time step (ms)
V_REST_MV = -65.0   # Resting membrane potential (mV)
V_RESET_MV = -65.0  # Reset potential after spike (mV)
V_TH_MV = -50.0     # Spike threshold (mV)
J_EXC_MV = 1.5      # Excitation potential increase per input (mV)
RATE_EXC_HZ = 1000    # Rate of excitatory inputs (Hz)
SEED = 42           # Random seed for reproducibility
time_constant = 5
alpha = np.exp(-DT_MS / time_constant)

# TODO: Initialize neuron state variables (voltage, spikes)
timesteps = int(T_TOTAL_MS / DT_MS)
voltages = np.full(timesteps, V_RESET_MV)
spikes = np.zeros(timesteps)
rng = np.random.default_rng(SEED)
p_exc = RATE_EXC_HZ * (DT_MS / 1000.0)

# TODO: Implement the simulation loop logic
for i in range(1, timesteps):
    if rng.random() < p_exc:
        voltages[i-1] += J_EXC_MV
    voltages[i] = voltages[i-1] + (V_RESET_MV - voltages[i-1]) * (1 - alpha)
    if voltages[i] >= V_TH_MV:
        spikes[i] = 1
        voltages[i] = V_RESET_MV

print(f"total spikes {np.sum(spikes)}")

time = np.arange(timesteps) * DT_MS

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane Potential (mV)', color=color)
ax1.plot(time, voltages, color=color, label='Voltage')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Spike', color=color)
ax2.plot(time, spikes, color=color, linestyle='None',
         marker='|', markersize=8, label='Spikes')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Leaky Integrate-and-Fire Neuron Simulation')
plt.show()
