# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import pickle
from datetime import datetime
import uuid

from bunch import Bunch

from utils.rest_api import CgmApi
from utils.preprocessing import process_depthmaps
from utils.inference import get_json_prediction
from utils.constants import PLAINCNN_WEIGHT_WORKFLOW_NAME, PLAINCNN_WEIGHT_WORKFLOW_VERSION, MEAN_PLAINCNN_WEIGHT_WORKFLOW_NAME, MEAN_PLAINCNN_WEIGHT_WORKFLOW_VERSION
from utils.result_utils import bunch_object_to_json_object, get_mean_scan_results, get_workflow, check_if_results_exists


rgb_format = ["rgb", "image/jpeg"]
depth_format = ["depth", "application/zip"]


cgm_api = CgmApi()


def get_scan_by_format(artifacts, file_format):
    return [artifact for artifact in artifacts if artifact['format'] in file_format]


def scan_level_weight_result_object(artifacts, predictions, scan_id, workflow_id, generated_timestamp):
    res = Bunch(dict(results=[]))
    result = Bunch(dict(
        id=f"{uuid.uuid4()}",
        scan=scan_id,
        workflow=workflow_id,
        source_artifacts=[artifact['id'] for artifact in artifacts],
        source_results=[],
        data={'mean_weight': get_mean_scan_results(predictions)},
        start_time=generated_timestamp,
        end_time=generated_timestamp,
        generated=generated_timestamp
    ))
    res.results.append(result)
    return res


def artifact_level_result(artifacts, predictions, scan_id, workflow_id, generated_timestamp):
    """Prepare artifact level height result object"""
    res = Bunch(dict(results=[]))
    for artifact, prediction in zip(artifacts, predictions):
        result = Bunch(dict(
            id=str(uuid.uuid4()),
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            data={'weight': str(prediction[0])},
            start_time=generated_timestamp,
            end_time=generated_timestamp,
            generated=generated_timestamp
        ))
        res.results.append(result)
    return res


def post_weight_result_object(artifacts, predictions, scan_id, artifact_workflow_id, scan_workflow_id, generated_timestamp):
    res = artifact_level_result(artifacts, predictions, scan_id, artifact_workflow_id, generated_timestamp)
    res_object = bunch_object_to_json_object(res)
    cgm_api.post_results(res_object)
    scan_res = scan_level_weight_result_object(artifacts, predictions, scan_id, scan_workflow_id, generated_timestamp)
    scan_res_object = bunch_object_to_json_object(scan_res)
    cgm_api.post_results(scan_res_object)


def main(payload):
    try:
        scan_metadata = payload['scan_metadata']
        workflows = payload['workflows']
        scan_id = scan_metadata['id']
        artifacts = scan_metadata['artifacts']
        version = scan_metadata['version']
        scan_type = scan_metadata['type']
        results = scan_metadata['results']
        artifact_level_workflow = get_workflow(workflows, PLAINCNN_WEIGHT_WORKFLOW_NAME, PLAINCNN_WEIGHT_WORKFLOW_VERSION)
        scan_level_workflow = get_workflow(workflows, MEAN_PLAINCNN_WEIGHT_WORKFLOW_NAME, MEAN_PLAINCNN_WEIGHT_WORKFLOW_VERSION)
        if not (check_if_results_exists(results, artifact_level_workflow['id']) and check_if_results_exists(results, scan_level_workflow['id'])):
            service_name = scan_level_workflow['data']['service_name']
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            depth_artifacts = get_scan_by_format(artifacts, depth_format)
            depthmaps = process_depthmaps(depth_artifacts, cgm_api)
            p_depthmaps = pickle.dumps(depthmaps)
            predictions = get_json_prediction(p_depthmaps, service_name)
            post_weight_result_object(depth_artifacts, predictions, scan_id, artifact_level_workflow['id'], scan_level_workflow['id'], generated_timestamp)
        return f"Hello!"
    except Exception as e:
        logging.error(e)
