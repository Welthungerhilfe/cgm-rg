import sys

import set_up_dummy_variables
sys.path.append('./src')  # noqa: E402
from api_endpoints import ApiEndpoints
from mock import patch
from result_gen_with_api import ProcessWorkflows
from result_generation.blur import BlurFlow
from result_generation.height.height_plaincnn import HeightFlowPlainCnn
from result_generation.weight import WeightFlow
from result_generation.result_generation import ResultGeneration


sdv = set_up_dummy_variables.create_dummy_vars()

'''
@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_blur_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return BlurFlow(
        get_dummy_result_generation_object_for_subclass(),
        'src/workflows/blur-workflow.json',
        'src/workflows/blur-faces-worklows.json',
        sdv.rgb_artifacts,
        sdv.scan_version,
        sdv.scan_type)
'''


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_height_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return HeightFlowPlainCnn(
        get_dummy_result_generation_object_for_subclass(),
        'src/workflows/height-plaincnn-workflow-artifact.json',
        'src/workflows/height-plaincnn-workflow-scan.json',
        sdv.depth_artifacts,
        sdv.rgb_artifacts,
        sdv.person_details)


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_weight_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return WeightFlow(
        get_dummy_result_generation_object_for_subclass(),
        'src/workflows/weight-workflow-artifact.json',
        'src/workflows/weight-workflow-scan.json',
        sdv.depth_artifacts,
        sdv.person_details)


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_result_generation_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return ResultGeneration(
        get_dummy_api_endpoint_object(),
        get_dummy_process_workflows_object(),
        sdv.scan_metadata,
        sdv.scan_parent_dir)


def get_dummy_result_generation_object_for_subclass():
    return ResultGeneration(
        get_dummy_api_endpoint_object(),
        get_dummy_process_workflows_object(),
        sdv.scan_metadata,
        sdv.scan_parent_dir)


def get_dummy_get_scan_metadata_object():
    pass


def get_dummy_api_endpoint_object():
    return ApiEndpoints(sdv.url)


def get_dummy_prepare_artifacts_object():
    pass


def get_dummy_process_workflows_object():
    return ProcessWorkflows(get_dummy_api_endpoint_object())
