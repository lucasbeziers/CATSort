from dataclasses import dataclass
from typing import Optional
from spikeinterface.preprocessing import normalize_by_quantile, bandpass_filter, whiten
from spikeinterface.core import BaseRecording

@dataclass
class PreprocessingParameters:
    """
    Parameters for the preprocessing pipeline.
    """
    normalize: bool = True
    freq_min: Optional[float] = 300
    freq_max: Optional[float] = 3000
    ftype: str = 'bessel'
    filter_order: int = 3
    margin_ms: float = 10.0
    whitening: bool = False

def preprocess_recording(
    recording: BaseRecording, 
    **kwargs
) -> BaseRecording:
    """
    Apply preprocessing steps to a recording.
    
    Args:
        recording: The spikeinterface recording.
        **kwargs: Preprocessing parameters (see PreprocessingParameters).
    """
    # Use PreprocessingParameters for defaults and validation
    params = PreprocessingParameters(**kwargs)

    if params.normalize:
        recording = normalize_by_quantile(recording)
    
    if params.freq_min is not None or params.freq_max is not None:
        recording = bandpass_filter(
            recording, 
            freq_min=params.freq_min, 
            freq_max=params.freq_max,
            ftype=params.ftype,
            filter_order=params.filter_order,
            margin_ms=params.margin_ms,
        )
        
    if params.whitening:
        recording = whiten(recording, mode='local', regularize=True, apply_mean=True)

    return recording