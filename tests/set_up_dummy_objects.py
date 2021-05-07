import sys

import set_up_dummy_variables as sdv
sys.path.append('./src')  # noqa: E402
from api_endpoints import ApiEndpoints
from mock import patch
from result_gen_with_api import ProcessWorkflows
from result_generation.blur import BlurFlow
from result_generation.height.height_plaincnn import HeightFlowPlainCnn
from result_generation.weight import WeightFlow

# @patch('workflows.get_workflow_id')
# @patch('BlurFlow.get_workflow_id', return_value="44af5600-69d2-11eb-9498-8ffe0e3b2017")


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_blur_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return BlurFlow(
        get_dummy_api_endpoint_object(),
        get_dummy_process_workflows_object(),
        'src/workflows/blur-workflow.json',
        sdv.rgb_artifacts,
        sdv.scan_parent_dir,
        sdv.scan_metadata,
        sdv.scan_version)


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_height_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return HeightFlowPlainCnn(
        get_dummy_api_endpoint_object(),
        get_dummy_process_workflows_object(),
        'src/workflows/height-plaincnn-workflow-artifact.json',
        'src/workflows/height-plaincnn-workflow-scan.json',
        sdv.depth_artifacts,
        sdv.rgb_artifacts,
        sdv.scan_parent_dir,
        sdv.scan_metadata,
        sdv.person_details)


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_weight_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return WeightFlow(
        get_dummy_api_endpoint_object(),
        get_dummy_process_workflows_object(),
        'src/workflows/weight-workflow-artifact.json',
        'src/workflows/weight-workflow-scan.json',
        sdv.depth_artifacts,
        sdv.scan_parent_dir,
        sdv.scan_metadata,
        sdv.person_details)


def get_dummy_get_scan_metadata_object():
    pass


def get_dummy_api_endpoint_object():
    return ApiEndpoints(sdv.url)


def get_dummy_prepare_artifacts_object():
    pass


def get_dummy_process_workflows_object():
    return ProcessWorkflows(get_dummy_api_endpoint_object())
