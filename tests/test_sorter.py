from spikeinterface.core import generate_recording, generate_sorting, generate_snippets
from spikeinterface.extractors import toy_example
from spikeinterface.comparison import compare_sorter_to_ground_truth

from catsort import sorter


def test_tetrode():
     recording, sorting_gt = toy_example(
        duration=10,
        num_channels=4,
        num_units=5, 
        sampling_frequency=30000,
        num_segments=1
    )
    
     default_parameters = sorter.DEFAULT_PARAMS
     sorting = sorter.run_catsort(recording, params=default_parameters)
     assert True

def test_performance_tetrode():
    recording, sorting_gt = toy_example(
            duration=10,
            num_channels=4,
            num_units=5, 
            sampling_frequency=30000,
            num_segments=1
        )
        
    default_parameters = sorter.DEFAULT_PARAMS
    sorting = sorter.run_catsort(recording, params=default_parameters)
    comparison = compare_sorter_to_ground_truth(sorting, sorting_gt, match_score=0.01)
    perf = comparison.get_performance()
    
    assert perf['accuracy'].mean() > 0.5
    assert perf['recall'].mean() > 0.5
    assert perf['precision'].mean() > 0.5

def test_monotrode():
     recording, sorting_gt = toy_example(
        duration=10,
        num_channels=1,
        num_units=5, 
        sampling_frequency=30000,
        num_segments=1
    )
    
     default_parameters = sorter.DEFAULT_PARAMS
     sorting = sorter.run_catsort(recording, params=default_parameters)
     assert True

def test_performance_monotrode():
    recording, sorting_gt = toy_example(
            duration=10,
            num_channels=1,
            num_units=5, 
            sampling_frequency=30000,
            num_segments=1
        )
        
    default_parameters = sorter.DEFAULT_PARAMS
    sorting = sorter.run_catsort(recording, params=default_parameters)
    comparison = compare_sorter_to_ground_truth(sorting, sorting_gt, match_score=0.01)
    perf = comparison.get_performance()
    
    assert perf['accuracy'].mean() > 0.25
    assert perf['recall'].mean() > 0.25
    assert perf['precision'].mean() > 0.25