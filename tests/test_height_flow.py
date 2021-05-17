import sys
from datetime import datetime

import numpy as np
from bunch import Bunch

sys.path.append('./src')  # noqa: E402
import set_up_dummy_objects


def test_process_depthmaps():
    """Test to check proper processing of depthmaps"""
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()

    # Exercise
    result = heightflow.process_depthmaps()

    # Verify
    assert isinstance(result, np.ndarray)


def test_artifact_level_height_result_object():
    """Test creation of artifact level height object"""
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
    """Test creation of scan level height object"""
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = heightflow.scan_level_height_result_object(predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)
'''
