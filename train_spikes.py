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
    window_size = 0.5
    dt = 0.01
    all_trials_spike_counts = []
    all_trials_isi_means = []
    all_trials_isi_stds = []
    all_trials_sliding_rates = []  # To store sliding window rates for each trial
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
        # Calculate sliding window firing rate for the current trial
        t_points = np.arange(0, max_time, dt)
        trial_sliding_rates = []
        half_window = window_size / 2
        for t in t_points:
            window_start = t - half_window
            window_end = t + half_window

            # Efficiently count spikes within the window using searchsorted
            # Assumes 'trial' (spike times) is sorted
            start_idx = np.searchsorted(trial, window_start, side='left')
            end_idx = np.searchsorted(trial, window_end, side='left')

            spikes_in_window = end_idx - start_idx

            # Firing rate for this window
            rate = spikes_in_window / window_size
            trial_sliding_rates.append(rate)
        all_trials_sliding_rates.append(trial_sliding_rates)

        # Calculate ISI (inter-spike intervals)
        inter_spike_intervals = np.diff(trial)
        isi_mean = np.mean(inter_spike_intervals)
        isi_std = np.std(inter_spike_intervals)
        all_trials_isi_means.append(isi_mean)
        all_trials_isi_stds.append(isi_std)
        # print(f"inter_spike_intervals: {inter_spike_intervals[:3]}")

    # PROBLEM 1: Average Firing rate overall
    avg_fr = np.mean(trial_fr)
    std_fr = np.std(trial_fr)
    print(f"average firing rate: {avg_fr}")
    print(f"std firing rate: {std_fr}")

    # PROBLEM 2: Time-dependent firing rate
    all_trials_spike_counts = np.array(
        all_trials_spike_counts)  # (N_trials x N_bins)
    mean_spike_counts_per_bin = np.mean(all_trials_spike_counts, axis=0)
    time_dependent_firing_rate = mean_spike_counts_per_bin / bin_size
    print(f"time_dependent_firing_rate: {time_dependent_firing_rate}")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, time_dependent_firing_rate)
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Peri-Stimulus Time Histogram (PSTH) for Neuron ELV133_3_2271")
    plt.show()

    # PROBLEM 3: Inter-spike Intervals
    # One property of ISIs are regularity of spikes. We can measure this through Coefficient of Variation
    # how much intervals vary relative to the mean
    all_trials_isi_means = np.array(
        all_trials_isi_means)
    all_trials_isi_stds = np.array(
        all_trials_isi_stds)
    cv_per_trial = all_trials_isi_stds / all_trials_isi_means
    print(f"cv_per_trial: {cv_per_trial}")
    # Analysis: CVs are ~1, so firing patterns are random, not bursty.

    # PROBLEM 4: Time-dependent firing rate using sliding window
    all_trials_sliding_rates = np.array(all_trials_sliding_rates)
    mean_sliding_rate = np.mean(all_trials_sliding_rates, axis=0)

    plt.figure()
    plt.plot(t_points, mean_sliding_rate)
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Sliding Window Firing Rate for Neuron ELV133_3_2271")
    plt.show()
