from pathlib import Path
import sys

from bunch import Bunch
import numpy as np

sys.path.append('./src')
import set_up_dummy_objects


def test_bunch_object_to_json_object():
    """Test to check if we get json object"""
    # Setup
    result_generation = set_up_dummy_objects.get_dummy_result_generation_object()
    bunch_object = Bunch(a=1, b=2)
    # Exercise
    result = result_generation.bunch_object_to_json_object(bunch_object)

    # Verify
    truth = {'a': 1, 'b': 2}
    assert result == truth
    assert isinstance(truth, dict)


def test_get_input_path():
    """Test to check if we get input path"""
    # Setup
    result_generation = set_up_dummy_objects.get_dummy_result_generation_object()
    directory = 'app/scans'
    filename = 'workflow.json'

    # Exercise
    result = result_generation.get_input_path(directory, filename)

    # Verify
    truth = 'app/scans/workflow.json'
    assert Path(result) == Path(truth)


def test_get_mean_scan_results():
    """Test to check if we get mean results"""
    # Setup
    result_generation = set_up_dummy_objects.get_dummy_result_generation_object()
    a = np.array([[4], [6], [5], [7]])

    # Exercise
    result = result_generation.get_mean_scan_results(a)

    # Verify
    truth = str(5.5)
    assert result == truth

def test_get_median_scan_results():
    """Test to check if we get correct median results"""
    # Setup
    result_generation = set_up_dummy_objects.get_dummy_result_generation_object()
    a = np.array([[4], [6], [5], [7], [8]])

    # Exercise
    result = result_generation.get_median_scan_results(a)

    # Verify
    truth = str(6.0)
    assert result == truth
