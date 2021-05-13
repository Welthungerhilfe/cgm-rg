from pathlib import Path
from datetime import datetime
import sys
sys.path.append('./src')

from bunch import Bunch
import numpy as np

import set_up_dummy_objects


def test_bunch_object_to_json_object():
    """Test to check if we get json object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    bunch_object = Bunch(a=1, b=2)
    # Exercise
    result = weightflow.bunch_object_to_json_object(bunch_object)

    # Verify
    truth = {'a': 1, 'b': 2}
    assert result == truth
    assert isinstance(truth, dict)

    # Cleanup - none required


def test_get_input_path():
    """Test to check if we get input path"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    directory = 'app/scans'
    filename = 'workflow.json'

    # Exercise
    result = weightflow.get_input_path(directory, filename)

    # Verify
    truth = 'app/scans/workflow.json'
    assert Path(result) == Path(truth)

    # Cleanup - none required


def test_get_mean_scan_results():
    """Test to check if we get mean results"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    a = np.array([[4], [6], [5], [7]])

    # Exercise
    result = weightflow.get_mean_scan_results(a)

    # Verify
    truth = str(5.5)
    assert result == truth


def test_process_depthmaps():
    """Test to check proper processing of depthmaps"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()

    # Exercise
    result = weightflow.process_depthmaps()

    # Verify

    assert isinstance(result, np.ndarray)


def test_artifact_level_weight_result_object():
    """Test creation of artifact level weight object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = weightflow.artifact_level_weight_result_object(predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)


'''
def test_scan_level_weight_result_object():
    """Test creation of scan level weight object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = weightflow.scan_level_weight_result_object(predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)
'''
