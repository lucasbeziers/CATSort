import numpy as np
from typing import Optional
from spikeinterface.core import BaseRecording, NumpySorting, SortingAnalyzer, Templates, ChannelSparsity, create_sorting_analyzer
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import spikeinterface.sortingcomponents.matching as sm
from sklearn.decomposition import PCA

from catsort.core.utils import get_peaks_traces_best_channel, get_peaks_traces_all_channels
from catsort.core.collision import (
    compute_collision_features, 
    detect_temporal_collisions, 
    optimize_collision_thresholds,
    compute_fixed_thresholds
)
from catsort.core.clustering import isosplit6_subdivision_method

DEFAULT_PARAMS = {
    # Detection
    'detect_threshold': 5,
    'exclude_sweep_ms': 0.2,
    'radius_um': 100,
    
    # Collision analysis
    'ms_before_spike_detected': 1.0,
    'ms_after_spike_detected': 1.0,
    'refractory_period': 2.0,
    'scheme': 'original',  # 'original' or 'adaptive'
 
    # Original scheme parameters
    'mad_multiplier_amplitude': 7.0,
    'mad_multiplier_width': 10.0,
    'mad_multiplier_energy': 15.0,   
    # Adaptive scheme parameters
    'false_positive_tolerance': 0.05,  # Used when scheme='adaptive'
    
    # Clustering
    'n_pca_components': 10,
    'npca_per_subdivision': 10,
    
    # Template matching
    'tm_method': 'wobble', # 'wobble' only for now
    
    # Template matching (Wobble)
    'threshold_wobble': 5000,
    'jitter_factor_wobble': 24,
    'refractory_period_ms_wobble': 2.0,
}

def get_sorting_analyzer_with_computations(
    sorting: NumpySorting,
    recording: BaseRecording,
    ms_before: float, ms_after: float
) -> SortingAnalyzer:
    sorting_analyzer = create_sorting_analyzer(sorting, recording, return_in_uV=True)
    sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=np.inf)
    sorting_analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after)
    sorting_analyzer.compute("templates", operators=["average", "median", "std"])
    return sorting_analyzer

def run_catsort(recording: BaseRecording, params: Optional[dict] = None) -> NumpySorting:
    """
    Main entry point for CATSort (Collision-Aware using Templates Sort).
    
    Args:
        recording: spikeinterface recording object
        params: dictionary of parameters (optional)
        
    Returns:
        sorting: spikeinterface sorting object
    """
    if params is None:
        params = DEFAULT_PARAMS
    else:
        # Merge with defaults
        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        params = full_params

    print("Step 1: Detecting spikes...")
    peaks_detected = detect_peaks(
        recording=recording,
        method="locally_exclusive",
        method_kwargs={
            "peak_sign": "neg",
            "detect_threshold": params['detect_threshold'],
            "exclude_sweep_ms": params['exclude_sweep_ms'],
            "radius_um": params['radius_um'],
        },
    )
    
    sampling_freq = recording.get_sampling_frequency()
    
    n_before = int(sampling_freq * params['ms_before_spike_detected'] * 0.001)
    n_after = int(sampling_freq * params['ms_after_spike_detected'] * 0.001)
    
    print("Step 2: Collision handling...")
    # Extract best channel traces for collision feature computation
    traces_best = get_peaks_traces_best_channel(peaks_detected, recording, n_before, n_after)
    
    # Temporal collisions
    too_close = detect_temporal_collisions(
        peaks_detected['sample_index'], 
        peaks_detected['channel_index'], 
        sampling_freq, 
        params['refractory_period']
    )
    
    # Compute features and thresholds based on scheme
    collision_features = compute_collision_features(traces_best, sampling_freq)
    
    if params['scheme'] == 'adaptive':
        thresholds = optimize_collision_thresholds(
            collision_features, 
            too_close, 
            params['false_positive_tolerance']
        )
    elif params['scheme'] == 'original':
        mad_multipliers = {
            'amplitude': params['mad_multiplier_amplitude'],
            'width': params['mad_multiplier_width'],
            'energy': params['mad_multiplier_energy']
        }
        thresholds = compute_fixed_thresholds(collision_features, mad_multipliers)
    else:
        raise ValueError(f"Unknown scheme: {params['scheme']}. Must be 'adaptive' or 'original'.")
    
    print(f"  Scheme: {params['scheme']}")
    
    # Flag collisions
    is_collision = too_close.copy()
    print(f"  Temporal collisions: {np.sum(too_close)}")
    for crit, val in collision_features.items():
        flagged_by_crit = val > thresholds[crit]
        flagged_by_crit_not_too_close = flagged_by_crit & ~too_close
        print(f"  {crit}: {np.sum(flagged_by_crit_not_too_close)} additional collisions")
        is_collision |= flagged_by_crit
        
    print(f"  Total flagged: {np.sum(is_collision)} collisions out of {len(peaks_detected)} spikes")
    
    print("Step 3: Clustering non-collided spikes...")
    mask_not_collided = ~is_collision
    traces_all_not_collided = get_peaks_traces_all_channels(peaks_detected[mask_not_collided], recording, n_before, n_after)
    
    # PCA and Clustering
    num_spikes, num_samples, num_channels = traces_all_not_collided.shape
    concatenated = traces_all_not_collided.reshape(num_spikes, -1)
    pca = PCA(n_components=params['n_pca_components'])
    features_not_collided = pca.fit_transform(concatenated)
    
    labels_not_collided = isosplit6_subdivision_method(
        features_not_collided, 
        npca_per_subdivision=params['npca_per_subdivision']
    )
    
    # Create sorting with clusters for template computation
    samples_clean = peaks_detected[mask_not_collided]['sample_index']
    labels_clean = labels_not_collided

    sorting_clean = NumpySorting.from_samples_and_labels(
        samples_list=[samples_clean],
        labels_list=[labels_clean],
        sampling_frequency=sampling_freq
    )
    
    print("Step 4: Template Matching...")
    # Compute templates from clean clusters
    analyzer = get_sorting_analyzer_with_computations(
        sorting_clean, recording, 
        params['ms_before_spike_detected'], 
        params['ms_after_spike_detected']
    )
    
    templates_ext = analyzer.get_extension('templates')
    sparsity = ChannelSparsity.create_dense(analyzer)
    
    templates = Templates(
        templates_array=templates_ext.data['average'],
        sampling_frequency=sampling_freq,
        nbefore=templates_ext.nbefore,
        is_in_uV=True,
        sparsity_mask=sparsity.mask,
        channel_ids=analyzer.channel_ids,
        unit_ids=analyzer.unit_ids,
        probe=analyzer.get_probe()
    )
    
    if params['tm_method'] == 'wobble':
        spikes_tm = sm.find_spikes_from_templates(
            recording=recording,
            templates=templates,
            method='wobble',
            
            method_kwargs={
                "parameters": {
                    "threshold": params['threshold_wobble'],
                    "jitter_factor": params['jitter_factor_wobble'],
                    "refractory_period_frames": int(sampling_freq * params['refractory_period_ms_wobble'] * 0.001),
                    "scale_amplitudes": True
                },
            }
        )
    else:
        raise ValueError(f"Unknown template matching method: {params['tm_method']}")
        
    final_sorting = NumpySorting.from_samples_and_labels(
        samples_list=[spikes_tm['sample_index']],
        labels_list=[spikes_tm['cluster_index']],
        sampling_frequency=sampling_freq
    )
    
    return final_sorting