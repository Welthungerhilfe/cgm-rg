import sys
from datetime import datetime

import numpy as np
from bunch import Bunch
import set_up_dummy_objects


sys.path.append('./src')  # noqa: E402

import utils.preprocessing as preprocessing


def test_bunch_object_to_json_object():
    """
    Test to check if we get json object
    """
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    bunch_object = Bunch(a=1, b=2)
    # Exercise
    result = heightflow.bunch_object_to_json_object(bunch_object)

    # Verify
    truth = {'a': 1, 'b': 2}
    assert result == truth
    assert isinstance(truth, dict)

    # Cleanup - none required


def test_get_input_path():
    """
    Test to check if we get input path
    """
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    directory = 'app/scans'
    filename = 'workflow.json'

    # Exercise
    result = heightflow.get_input_path(directory, filename)

    # Verify
    truth = 'app/scans/workflow.json'
    assert result == truth

    # Cleanup - none required


def test_get_mean_scan_results():
    """
    Test to check if we get mean results
    """
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    a = np.array([[4], [6], [5], [7]])

    # Exercise
    result = heightflow.get_mean_scan_results(a)

    # Verify
    truth = str(5.5)
    assert result == truth


def test_process_depthmaps():
    """
    Test to check proper processing of depthmaps
    """
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    preprocessing.set_width(int(240))
    preprocessing.set_height(int(180))

    # Exercise
    result = heightflow.process_depthmaps()

    # Verify

    assert isinstance(result, np.ndarray)


def test_artifact_level_height_result_object():
    """
    Test creation of artifact level height object
    """
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = heightflow.artifact_level_height_result_object(
        predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)


'''
def test_scan_level_height_result_object():
    """
    Test creation of scan level height object
    """
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = heightflow.scan_level_height_result_object(predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)
'''
