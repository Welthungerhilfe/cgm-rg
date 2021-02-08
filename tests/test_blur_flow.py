import sys
sys.path.append('./src')
import pytest
from bunch import Bunch
import json
import os
#from src.result_gen_with_api import BlurFlow
from result_gen_with_api import BlurFlow, ProcessWorkflows
from api_endpoints import ApiEndpoints

os.environ['APP_ENV'] = 'LOCAL'

scan_endpoint = '/api/scans/unprocessed?limit=1'
get_file_endpoint = '/api/files/'
post_file_endpoint = '/api/files'
result_endpoint = '/api/results'
workflow_endpoint = '/api/workflows'
url = "http://localhost:5001"

cgm_api = ApiEndpoints(
        url,
        scan_endpoint,
        get_file_endpoint,
        post_file_endpoint,
        result_endpoint,
        workflow_endpoint)

workflow = ProcessWorkflows(cgm_api)

blurflow = BlurFlow(cgm_api, workflow, blur_workflow_path, rgb_artifacts={}, scan_parent_dir='', scan_metadata={})

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
