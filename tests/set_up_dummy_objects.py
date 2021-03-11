import set_up_dummy_variables
import sys
sys.path.append('./src')
from result_generation.blur import BlurFlow
from result_generation.height import HeightFlow
from result_gen_with_api import ProcessWorkflows
from api_endpoints import ApiEndpoints
from mock import patch


# @patch('workflows.get_workflow_id')
# @patch('BlurFlow.get_workflow_id', return_value="44af5600-69d2-11eb-9498-8ffe0e3b2017")

@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_blur_flow_object(mock_some_fn):
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return BlurFlow(
        get_dummy_api_endpoint_object(),
        get_dummy_process_workflows_object(),
        'src/workflows/blur-workflow.json',
        set_up_dummy_variables.rgb_artifacts,
        set_up_dummy_variables.scan_parent_dir,
        set_up_dummy_variables.scan_metadata)


@patch.object(ProcessWorkflows, 'get_workflow_id')
def get_dummy_height_flow_object():
    mock_some_fn.return_value = '44af5600-69d2-11eb-9498-8ffe0e3b2017'
    return HeightFlow(
            get_dummy_api_endpoint_object(),
            get_dummy_process_workflows_object(),
            'src/workflows/height-workflow-artifact.json',
            'src/workflows/height-workflow-scan.json',
            set_up_dummy_variables.depth_artifacts,
            set_up_dummy_variables.scan_parent_dir,
            set_up_dummy_variables.scan_metadata,
            set_up_dummy_variables.person_details)


def get_dummy_weight_flow_object():
    pass


def get_dummy_get_scan_metadata_object():
    pass


def get_dummy_api_endpoint_object():
    return ApiEndpoints(
        set_up_dummy_variables.url,
        set_up_dummy_variables.scan_endpoint,
        set_up_dummy_variables.get_file_endpoint,
        set_up_dummy_variables.post_file_endpoint,
        set_up_dummy_variables.result_endpoint,
        set_up_dummy_variables.workflow_endpoint,
        set_up_dummy_variables.person_detail_endpoint)


def get_dummy_prepare_artifacts_object():
    pass


def get_dummy_process_workflows_object():
    return ProcessWorkflows(get_dummy_api_endpoint_object())
