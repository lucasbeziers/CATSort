import numpy as np

def get_snippet(
    traces: np.ndarray,
    index: int,
    n_before: int, n_after: int
    ) -> np.ndarray:
    """
    Get a snippet of the traces around a specific index.
    Fill the snippet with zeros if the index is out of bounds.
    """
    n_channels = traces.shape[1]
    snippet = np.zeros((n_before + n_after, n_channels))

    start = index - n_before
    end = index + n_after

    # If the snippet is fully within bounds, extract it directly
    if start >= 0 and end <= traces.shape[0]:
        snippet = traces[start:end, :]

    # If the snippet is partially out of bounds, fill with zeros where necessary
    else:
        valid_start = max(start, 0)
        valid_end = min(end, traces.shape[0])
        insert_start = valid_start - start
        insert_end = insert_start + (valid_end - valid_start)
        snippet[insert_start:insert_end, :] = traces[valid_start:valid_end, :]

    return snippet # shape (n_before+n_after, n_channels)

def get_peaks_traces_all_channels(
    peaks: np.ndarray,
    traces: np.ndarray,
    n_before: int, n_after: int
    ) -> np.ndarray:
    """
    Extract snippets of traces around detected peaks.

    Output shape: (n_peaks, n_before + n_after, n_channels)
    """
    n_channels = traces.shape[1]
    complete_peaks = np.zeros((len(peaks), n_before+n_after, n_channels))

    for i, peak in enumerate(peaks):
        sample_index = peak['sample_index']
        snippet = get_snippet(traces, sample_index, n_before, n_after)
        complete_peaks[i] = snippet
    return complete_peaks


def get_peaks_traces_best_channel(
    peaks: np.ndarray,
    traces: np.ndarray,
    n_before: int, n_after: int
    ) -> np.ndarray:
    """
    Extract snippets of traces around detected peaks.
    Keep only the channel with the highest amplitude

    Output shape: (n_peaks, n_before + n_after)
    """
    complete_peaks = np.zeros((len(peaks), n_before+n_after))

    for i, peak in enumerate(peaks):
        sample_index = peak['sample_index']
        snippet = get_snippet(traces, sample_index, n_before, n_after) # shape (n_before+n_after, n_channels)
        best_channel = np.argmax(np.abs(snippet).max(axis=0))
        complete_peaks[i] = snippet[:, best_channel] # shape (n_before+n_after)
    return complete_peaks
