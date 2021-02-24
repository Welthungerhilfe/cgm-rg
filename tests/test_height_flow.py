import sys
sys.path.append('./src')
import pytest
from bunch import Bunch
import json
import os
# from src.result_gen_with_api import BlurFlow
from result_gen_with_api import BlurFlow, ProcessWorkflows
# from api_endpoints import ApiEndpoints
import set_up_dummy_objects


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
    blurflow = set_up_dummy_objects.get_dummy_blur_flow_object()
    directory = 'app/scans'
    filename = 'workflow.json'

    # Exercise
    result = blurflow.get_input_path(directory, filename)

    # Verify
    truth = 'app/scans/workflow.json'
    assert result == truth

    # Cleanup - none required

