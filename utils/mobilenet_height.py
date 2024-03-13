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
import itertools

from bunch import Bunch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from utils.rest_api import CgmApi
from utils.preprocessing import mobilenet_process_depthmaps
from utils.inference import get_json_prediction
from utils.constants import MOBILENET_HEIGHT_WORKFLOW_NAME, MOBILENET_HEIGHT_WORKFLOW_VERSION, MEAN_MOBILENET_HEIGHT_WORKFLOW_NAME, MEAN_MOBILENET_HEIGHT_WORKFLOW_VERSION
from utils.result_utils import bunch_object_to_json_object, get_mean_scan_results, get_workflow, check_if_results_exists

MAX_BATCH_SIZE = 9

rgb_format = ["rgb", "image/jpeg"]
depth_format = ["depth", "application/zip"]


cgm_api = CgmApi()


def remove_outliers(arr):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    result = [nums for nums in arr if lower <= nums <= upper]
    arr_new = np.array(result).reshape(len(result), 1)
    return arr_new


# Convert the list to an array of subarrays
def LOF(predictions):
    # data = rows['prediction_heights']
    if len(predictions)>8:
        # Apply the Local Outlier Factor algorithm
        lof = LocalOutlierFactor(n_neighbors=5)
        outlier_labels = lof.fit_predict(predictions)
        # Filter out the outliers
        inliers = predictions[outlier_labels == 1]
        inliers_data = list(itertools.chain.from_iterable(inliers))
        return inliers_data
    return []


def get_scan_by_format(artifacts, file_format):
    return [artifact for artifact in artifacts if artifact['format'] in file_format]


def scan_level_height_result_object(artifacts, predictions, scan_id, workflow_id, generated_timestamp):
    predictions = remove_outliers(predictions)
    lof_predictions = LOF(predictions)
    res = Bunch(dict(results=[]))
    result = Bunch(dict(
        id=f"{uuid.uuid4()}",
        scan=scan_id,
        workflow=workflow_id,
        source_artifacts=[artifact['id'] for artifact in artifacts],
        source_results=[],
        data={
            'mean_height': get_mean_scan_results(predictions),
            'lof_mean_height': get_mean_scan_results(lof_predictions)
        },
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
            data={'height': str(prediction[0])},
            start_time=generated_timestamp,
            end_time=generated_timestamp,
            generated=generated_timestamp
        ))
        res.results.append(result)
    return res


def post_height_result_object(cgm_api, artifacts, predictions, scan_id, artifact_workflow_id, scan_workflow_id, generated_timestamp):
    res = artifact_level_result(artifacts, predictions, scan_id, artifact_workflow_id, generated_timestamp)
    res_object = bunch_object_to_json_object(res)
    # logging.info(f"posting artifact result {res_object}")
    cgm_api.post_results(res_object)
    scan_res = scan_level_height_result_object(artifacts, predictions, scan_id, scan_workflow_id, generated_timestamp)
    scan_res_object = bunch_object_to_json_object(scan_res)
    # logging.info(f"posting scan result {scan_res_object}")
    cgm_api.post_results(scan_res_object)


def run_mobilenet_height_flow(cgm_api, scan_id, artifacts, workflows, results):
    artifact_level_workflow = get_workflow(workflows, MOBILENET_HEIGHT_WORKFLOW_NAME, MOBILENET_HEIGHT_WORKFLOW_VERSION)
    scan_level_workflow = get_workflow(workflows, MEAN_MOBILENET_HEIGHT_WORKFLOW_NAME, MEAN_MOBILENET_HEIGHT_WORKFLOW_VERSION)
    if not (check_if_results_exists(results, artifact_level_workflow['id']) and check_if_results_exists(results, scan_level_workflow['id'])):
        service_name = scan_level_workflow['data']['service_name']
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        # depth_artifacts = get_scan_by_format(artifacts, depth_format)
        depthmaps = mobilenet_process_depthmaps(artifacts, cgm_api)
        total_data = len(depthmaps)
        # p_depthmaps = pickle.dumps(depthmaps)
        i = 0
        predictions = []
        for i in range(0, total_data, MAX_BATCH_SIZE):
            pickled_data = pickle.dumps(depthmaps[i:i + MAX_BATCH_SIZE])
            predictions.extend(get_json_prediction(pickled_data, service_name))
        post_height_result_object(cgm_api, artifacts, predictions, scan_id, artifact_level_workflow['id'], scan_level_workflow['id'], generated_timestamp)
