import pytest
from spikeinterface.extractors import toy_example
from spikeinterface.comparison import compare_sorter_to_ground_truth

from catsort import sorter

@pytest.fixture(scope="module")
def tetrode_data():
    recording, sorting_gt = toy_example(
        duration=10,
        num_channels=4,
        num_units=5, 
        sampling_frequency=30000,
        num_segments=1
    )
    return recording, sorting_gt

@pytest.fixture(scope="module")
def monotrode_data():
    recording, sorting_gt = toy_example(
        duration=10,
        num_channels=1,
        num_units=5, 
        sampling_frequency=30000,
        num_segments=1
    )
    return recording, sorting_gt

def test_tetrode(tetrode_data):
    recording, _ = tetrode_data
    default_parameters = sorter.DEFAULT_PARAMS
    sorting = sorter.run_catsort(recording, params=default_parameters)
    assert len(sorting.get_unit_ids()) > 0

def test_performance_tetrode(tetrode_data):
    recording, sorting_gt = tetrode_data
    default_parameters = sorter.DEFAULT_PARAMS
    sorting = sorter.run_catsort(recording, params=default_parameters)
    comparison = compare_sorter_to_ground_truth(sorting_gt, sorting, match_score=0.1)
    perf = comparison.get_performance()
    
    assert perf['accuracy'].mean() > 0.5
    assert perf['recall'].mean() > 0.5
    assert perf['precision'].mean() > 0.5

def test_monotrode(monotrode_data):
    recording, _ = monotrode_data
    default_parameters = sorter.DEFAULT_PARAMS
    sorting = sorter.run_catsort(recording, params=default_parameters)
    assert len(sorting.get_unit_ids()) > 0

def test_performance_monotrode(monotrode_data):
    recording, sorting_gt = monotrode_data
    default_parameters = sorter.DEFAULT_PARAMS
    sorting = sorter.run_catsort(recording, params=default_parameters)
    comparison = compare_sorter_to_ground_truth(sorting_gt, sorting, match_score=0.1)
    perf = comparison.get_performance()
    
    assert perf['accuracy'].mean() > 0.19
    assert perf['recall'].mean() > 0.19
    assert perf['precision'].mean() > 0.19

def test_scheme_adaptive(tetrode_data):
    recording, _ = tetrode_data
    adaptive_parameters = sorter.DEFAULT_PARAMS.copy()
    adaptive_parameters['scheme'] = 'adaptive'
    sorting = sorter.run_catsort(recording, params=adaptive_parameters)
    assert len(sorting.get_unit_ids()) > 0