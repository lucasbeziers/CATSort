import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from catsort.core.utils import get_snippet, get_peaks_traces_all_channels, get_peaks_traces_best_channel

@pytest.fixture
def dummy_recording():
    num_channels = 4
    num_samples = 1000
    sampling_frequency = 30000
    traces = np.zeros((num_samples, num_channels), dtype='float32')
    
    # Add some data to distinguish channels
    for ch in range(num_channels):
        traces[:, ch] = np.arange(num_samples) + ch * 1000
        
    recording = NumpyRecording([traces], sampling_frequency=sampling_frequency)
    return recording, traces

def test_get_snippet_in_bounds(dummy_recording):
    recording, traces = dummy_recording
    index = 500
    n_before, n_after = 10, 10
    
    snippet = get_snippet(recording, index, n_before, n_after)
    
    assert snippet.shape == (20, 4)
    assert np.allclose(snippet, traces[490:510, :])

def test_get_snippet_at_start(dummy_recording):
    recording, traces = dummy_recording
    index = 5  # n_before = 10 -> starts at -5
    n_before, n_after = 10, 10
    
    snippet = get_snippet(recording, index, n_before, n_after)
    
    assert snippet.shape == (20, 4)
    # First 5 samples should be 0
    assert np.all(snippet[:5, :] == 0)
    # Next 15 samples should match traces[0:15]
    assert np.allclose(snippet[5:, :], traces[0:15, :])

def test_get_snippet_at_end(dummy_recording):
    recording, traces = dummy_recording
    num_samples = recording.get_num_samples()
    index = num_samples - 5 # n_after = 10 -> ends at num_samples + 5
    n_before, n_after = 10, 10
    
    snippet = get_snippet(recording, index, n_before, n_after)
    
    assert snippet.shape == (20, 4)
    # First 15 samples should match traces[num_samples-15:num_samples]
    assert np.allclose(snippet[:15, :], traces[num_samples-15:num_samples, :])
    # Last 5 samples should be 0
    assert np.all(snippet[15:, :] == 0)

def test_get_peaks_traces_all_channels(dummy_recording):
    recording, traces = dummy_recording
    peaks = np.array([
        (500, 0, -10.0), # sample_index, channel_index, amplitude
        (100, 1, -5.0)
    ], dtype=[('sample_index', 'int64'), ('channel_index', 'int64'), ('amplitude', 'float32')])
    
    n_before, n_after = 5, 5
    peaks_traces = get_peaks_traces_all_channels(peaks, recording, n_before, n_after)
    
    assert peaks_traces.shape == (2, 10, 4)
    assert np.allclose(peaks_traces[0], traces[495:505, :])
    assert np.allclose(peaks_traces[1], traces[95:105, :])

def test_get_peaks_traces_best_channel(dummy_recording):
    recording, _ = dummy_recording
    sampling_freq = recording.get_sampling_frequency()
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()
    
    # Create new traces with clear best channels
    traces = np.zeros((num_samples, num_channels), dtype='float32')
    # Peak 1: Channel 2 is strongest
    traces[500, 2] = -100.0
    # Peak 2: Channel 0 is strongest
    traces[100, 0] = -50.0
    
    new_recording = NumpyRecording([traces], sampling_frequency=sampling_freq)
    
    peaks = np.array([
        (500, 2, -100.0),
        (100, 0, -50.0)
    ], dtype=[('sample_index', 'int64'), ('channel_index', 'int64'), ('amplitude', 'float32')])
    
    n_before, n_after = 5, 5
    peaks_best = get_peaks_traces_best_channel(peaks, new_recording, n_before, n_after)
    
    # Extract expected traces directly from the new recording to be sure
    expected_peak1 = new_recording.get_traces(start_frame=495, end_frame=505)[:, 2]
    expected_peak2 = new_recording.get_traces(start_frame=95, end_frame=105)[:, 0]
    
    assert peaks_best.shape == (2, 10)
    assert np.allclose(peaks_best[0], expected_peak1)
    assert np.allclose(peaks_best[1], expected_peak2)
