import numpy as np
from scipy.signal import resample

def longest_true_runs(arr: np.ndarray) -> np.ndarray:
    """
    For each row in a 2D boolean array, finds the longest consecutive run of True values
    and returns a boolean array of the same shape with only that run set to True.
    """
    out = np.zeros_like(arr, dtype=bool)
    for i, row in enumerate(arr):
        # Find start/end indices of True runs
        padded = np.r_[False, row, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[::2], edges[1::2]
        if len(starts) == 0:
            continue
        lengths = ends - starts
        j = np.argmax(lengths)
        out[i, starts[j]:ends[j]] = True
    return out

def compute_collision_features(
    peaks_traces: np.ndarray,
    sampling_frequency: float,
    width_threshold_amplitude: float = 0.5
) -> dict:
    """
    Compute collision features (amplitude, width, energy) for each peak.
    """
    amplitudes = np.abs(peaks_traces).max(axis=1)
    energies = np.sum(peaks_traces**2, axis=1)
    
    # Resample for width precision if 1D trace per peak
    if peaks_traces.ndim == 2:
        num_samples = peaks_traces.shape[1]
        resample_factor = 8
        peaks_traces_resampled = resample(peaks_traces, num=num_samples * resample_factor, axis=1)
        # Compute widths on resampled traces
        under_width_threshold = peaks_traces_resampled < -width_threshold_amplitude * amplitudes[:, np.newaxis]
        longest_under = longest_true_runs(under_width_threshold)
        widths_samples = np.sum(longest_under, axis=1) / resample_factor
    else:
        under_width_threshold = peaks_traces < -width_threshold_amplitude * amplitudes[:, np.newaxis]
        widths_samples = np.sum(longest_true_runs(under_width_threshold), axis=1)
        
    widths_ms = widths_samples * (1000 / sampling_frequency)
    
    return {
        "amplitude": amplitudes,
        "width": widths_ms,
        "energy": energies
    }

def detect_temporal_collisions(
    sample_indices: np.ndarray,
    channel_indices: np.ndarray,
    sampling_frequency: float,
    refractory_period_ms: float = 2.0
) -> np.ndarray:
    """
    Identify spikes that are too close to each other on the same channel.
    """
    prev_diffs = np.full_like(sample_indices, np.inf, dtype=float)
    next_diffs = np.full_like(sample_indices, np.inf, dtype=float)
    
    for ch in np.unique(channel_indices):
        mask = channel_indices == ch
        idx = np.where(mask)[0]
        samples = sample_indices[mask]
        
        # Sort by time within this channel
        order = np.argsort(samples)
        sorted_idx = idx[order]
        sorted_samples = samples[order]
        
        # Compute diffs
        if len(sorted_samples) > 1:
            diffs = np.diff(sorted_samples)
            prev_diffs[sorted_idx[1:]] = diffs
            next_diffs[sorted_idx[:-1]] = diffs
            
    closest_sample_diff = np.minimum(np.abs(prev_diffs), np.abs(next_diffs))
    closest_ms = closest_sample_diff * 1000 / sampling_frequency
    return closest_ms < refractory_period_ms

def optimize_collision_thresholds(
    features: dict,
    temporal_collisions: np.ndarray,
    false_positive_tolerance: float = 0.05
) -> dict:
    """
    Find optimal thresholds for collision features based on temporal collisions.
    """
    non_collision_count = (~temporal_collisions).sum()
    max_fp = false_positive_tolerance * non_collision_count
    
    optimized_thresholds = {}
    
    for criterion, values in features.items():
        unique_vals = np.sort(np.unique(values))
        best_thr = unique_vals[-1] # Default to max (nothing flagged)
        best_tp = -1
        
        # Binary search or scan for best threshold
        # For simplicity, we scan a subset of values if too many
        if len(unique_vals) > 1000:
            candidates = unique_vals[::len(unique_vals)//1000]
        else:
            candidates = unique_vals
            
        for thr in candidates:
            flagged = values > thr
            tp = np.count_nonzero(flagged & temporal_collisions)
            fp = np.count_nonzero(flagged & ~temporal_collisions)
            
            if fp <= max_fp:
                if tp > best_tp:
                    best_tp = tp
                    best_thr = thr
        
        optimized_thresholds[criterion] = best_thr
        
    return optimized_thresholds
