import sys
from datetime import datetime

from bunch import Bunch
import numpy as np

sys.path.append('./src')  # noqa: E402
import set_up_dummy_objects
import utils.preprocessing as preprocessing  # noqa: E402


def test_process_depthmaps():
    """Test to check proper processing of depthmaps"""
    # Setup
    flow = set_up_dummy_objects.get_dummy_height_flow_object()

    # Exercise
    depthmaps = preprocessing.process_depthmaps(flow.artifacts, flow.scan_directory, flow.result_generation)

    # Verify
    assert isinstance(depthmaps, np.ndarray)


def test_artifact_level_result():
    """Test creation of artifact level height object"""
    # Setup
    flow = set_up_dummy_objects.get_dummy_height_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result, _, _, _ = flow.artifact_level_result(predictions, generated_timestamp, start_time)

    # Verify
    assert isinstance(result, Bunch)



'''
def test_scan_level_height_result_object():
    """Test creation of scan level height object"""
    # Setup
    heightflow = set_up_dummy_objects.get_dummy_height_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    pos_percentile_error_99 = 50
    neg_percentile_error_99 = 50
    mae_scan = 0.8

    # Exercise
    result = heightflow.scan_level_height_result_object(predictions, generated_timestamp, workflow_obj, start_time, pos_percentile_error_99, neg_percentile_error_99, mae_scan)

    # Verify

    assert isinstance(result, Bunch)
'''
