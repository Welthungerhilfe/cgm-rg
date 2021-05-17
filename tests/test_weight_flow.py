from datetime import datetime
import sys
sys.path.append('./src')

from bunch import Bunch
import numpy as np

import set_up_dummy_objects


def test_process_depthmaps():
    """Test to check proper processing of depthmaps"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()

    # Exercise
    result = weightflow.process_depthmaps()

    # Verify

    assert isinstance(result, np.ndarray)


def test_artifact_level_result():
    """Test creation of artifact level weight object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = weightflow.artifact_level_result(predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)


'''
def test_scan_level_result():
    """Test creation of scan level weight object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result = weightflow.scan_level_result(predictions, generated_timestamp)

    # Verify

    assert isinstance(result, Bunch)
'''
