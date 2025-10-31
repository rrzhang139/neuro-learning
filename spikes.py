import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def extract_spike_times(ts_field):
    """Extract spike times from TS field, handling empty, single, or array values."""
    if isinstance(ts_field, np.ndarray):
        if ts_field.size == 0:
            return np.array([])
        if ts_field.ndim == 0:
            return np.array([float(ts_field)])
        if ts_field.dtype == object:
            all_spikes = []
            for item in ts_field.flat:
                if isinstance(item, np.ndarray):
                    if item.size > 0:
                        all_spikes.extend(item.astype(float).tolist())
                else:
                    all_spikes.append(float(item))
            return np.array(all_spikes)
        return ts_field.astype(float)
    return np.array([float(ts_field)])


def load_spike_trains(filepath):
    """Load spike train data from MATLAB file and extract all spike times.

    Returns:
        np.ndarray: Array of shape (n_neurons, n_trials) with dtype=object.
                    Each element is a 1D array of spike times in seconds.
    """
    mat = scipy.io.loadmat(filepath, squeeze_me=True)
    spike_data = mat['MatData']
    neuron_data = spike_data['class'].item()
    # breakpoint()

    neuron_spikes = []

    for trial in neuron_data:
        trial = trial.item()[0]
        spike_times = extract_spike_times(trial['TS'])
        neuron_spikes.append(spike_times)

    return np.array(neuron_spikes, dtype=object)


if __name__ == '__main__':

    neuron_spikes = load_spike_trains("pfc-3/ELV133_3_2271.mat")
    print(f"Shape: {neuron_spikes.shape}")
    print(f"Dtype: {neuron_spikes.dtype}")

    trial_fr = []
    bin_size = 0.2
    max_time = 8
    all_trials_spike_counts = []
    for trial_idx, trial in enumerate(neuron_spikes):
        spikes = len(trial)
        if spikes == 0:
            continue
        trial.sort()

        # Calculate simple firing rate
        trial_duration = trial[-1] - trial[0]
        firing_rate = spikes / trial_duration
        # print(f"firing rate {firing_rate}")
        trial_fr.append(firing_rate)

        # Calculate the time-dependent spikes per count.
        # NOTE This depends on an arbitrary bin size and uniformly distributed spike count
        # create edges of the time bins
        bin_edges = np.arange(0, max_time, bin_size)
        spikes_per_bin, _ = np.histogram(trial, bins=bin_edges)
        all_trials_spike_counts.append(spikes_per_bin)

        # Calculate ISI (inter-spike intervals)

    avg_fr = np.mean(trial_fr)
    std_fr = np.std(trial_fr)
    print(f"average firing rate: {avg_fr}")
    print(f"std firing rate: {std_fr}")

    print(np.stack(all_trials_spike_counts, axis=0).shape)

    all_trials_spike_counts = np.array(
        all_trials_spike_counts)  # (N_trials x N_bins)
    mean_spike_counts_per_bin = np.mean(all_trials_spike_counts, axis=0)
    time_dependent_firing_rate = mean_spike_counts_per_bin / bin_size
    print(f"time_dependent_firing_rate: {time_dependent_firing_rate}")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, time_dependent_firing_rate)
    plt.show()
