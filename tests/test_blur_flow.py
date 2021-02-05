import sys
sys.path.append('./src')
import pytest
from bunch import Bunch
import json
#from src.result_gen_with_api import BlurFlow
import result_gen_with_api

blurflow = BlurFlow(cgm_api, workflow, blur_workflow_path, rgb_artifacts, scan_parent_dir, scan_metadata)

def test_bunch_object_to_json_object():
    """
    Test to check if we get json object .
    """
    # Setup
    bunch_object = Bunch(a=1, b=2)
    # Exercise
    result = blurflow.bunch_object_to_json_object(bunch_object)

    # Verify
    truth = {'a': 1, 'b': 2}
    assert result == truth
    assert isinstance(truth, dict)

    # Cleanup - none required
