import pytest
import numpy as np
from spikeinterface.core import NumpyRecording
from catsort.core.preprocessing import PreprocessingParameters, preprocess_recording

@pytest.fixture
def dummy_recording():
    num_channels = 4
    num_samples = 30000
    sampling_frequency = 30000
    traces = np.random.randn(num_samples, num_channels).astype('float32')
    recording = NumpyRecording([traces], sampling_frequency=sampling_frequency)
    # Necessary for whitening
    recording.set_channel_locations(np.zeros((num_channels, 2)))
    return recording

def test_preprocessing_parameters_defaults():
    """Test that default parameters are set correctly."""
    params = PreprocessingParameters()
    assert params.normalize is True
    assert params.freq_min == 300
    assert params.freq_max == 3000
    assert params.whitening is False

def test_preprocess_recording_default(dummy_recording):
    """Test preprocessing with default parameters."""
    processed = preprocess_recording(dummy_recording)
    assert processed is not None
    assert processed.get_num_channels() == dummy_recording.get_num_channels()
    assert processed.get_num_samples() == dummy_recording.get_num_samples()

def test_preprocess_recording_no_op(dummy_recording):
    """Test preprocessing with all options disabled (no-op)."""
    processed = preprocess_recording(
        dummy_recording,
        normalize=False,
        freq_min=None,
        freq_max=None,
        whitening=False
    )
    
    # In spikeinterface, if no preprocessing is applied, it might return the same object 
    # or a thin wrapper. We check if the data is identical.
    raw_data = dummy_recording.get_traces()
    processed_data = processed.get_traces()
    assert np.allclose(raw_data, processed_data)

def test_preprocess_recording_full(dummy_recording):
    """Test preprocessing with all options enabled."""
    processed = preprocess_recording(
        dummy_recording,
        normalize=True,
        freq_min=300,
        freq_max=3000,
        whitening=True
    )
    assert processed is not None
    assert processed.get_num_channels() == dummy_recording.get_num_channels()
    
    # Check that data has changed
    raw_data = dummy_recording.get_traces()
    processed_data = processed.get_traces()
    assert not np.allclose(raw_data, processed_data)

def test_preprocess_recording_individual_steps(dummy_recording):
    """Test individual preprocessing steps independently."""
    
    # Only normalization
    processed_norm = preprocess_recording(dummy_recording, normalize=True, freq_min=None, freq_max=None, whitening=False)
    assert not np.allclose(dummy_recording.get_traces(), processed_norm.get_traces())
    
    # Only bandpass
    processed_bp = preprocess_recording(dummy_recording, normalize=False, freq_min=300, freq_max=3000, whitening=False)
    assert not np.allclose(dummy_recording.get_traces(), processed_bp.get_traces())
    
    # Only whitening
    processed_white = preprocess_recording(dummy_recording, normalize=False, freq_min=None, freq_max=None, whitening=True)
    assert not np.allclose(dummy_recording.get_traces(), processed_white.get_traces())
