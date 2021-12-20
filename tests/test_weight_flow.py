from datetime import datetime
import sys

from bunch import Bunch
import numpy as np

sys.path.append('./src')
import set_up_dummy_objects
import utils.preprocessing as preprocessing  # noqa: E402


def test_process_depthmaps():
    """Test to check proper processing of depthmaps"""
    # Setup
    flow = set_up_dummy_objects.get_dummy_weight_flow_object()

    # Exercise
    depthmaps = preprocessing.process_depthmaps(flow.artifacts, flow.scan_directory, flow.result_generation)
    # Verify
    assert isinstance(depthmaps, np.ndarray)


def test_artifact_level_result():
    """Test creation of artifact level weight object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Exercise
    result, _, _, _ = weightflow.artifact_level_result(predictions, generated_timestamp, start_time)

    # Verify
    assert isinstance(result, Bunch)


'''
def test_scan_level_result():
    """Test creation of scan level weight object"""
    # Setup
    weightflow = set_up_dummy_objects.get_dummy_weight_flow_object()
    predictions = np.random.uniform(70, 80, [26, 1])
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    pos_percentile_error_99 = 50
    neg_percentile_error_99 = 50
    mae_scan = 0.3

    # Exercise
    result = weightflow.scan_level_result(
        predictions,
        generated_timestamp,
        start_time,
        pos_percentile_error_99,
        neg_percentile_error_99,
        mae_scan)
    # Verify
    assert isinstance(result, Bunch)
'''
