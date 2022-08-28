import logging
from os import environ

import azure.functions as func
import requests
from os import getenv
from datetime import datetime
import uuid
from bunch import Bunch
import json
import numpy as np

from utils.preprocessing import process_depthmaps
from utils.rest_api import MlApi, ErrorStatsApi
from utils.inference import get_height_prediction
from utils.result_object_utils import bunch_object_to_json_object, calculate_age


ml_api = MlApi()
error_stats_api = ErrorStatsApi()


def artifact_level_result(artifacts, predictions, workflow_id, scan_id, generated_timestamp):
    """Prepare artifact level height result object"""
    res = Bunch(dict(results=[]))
    for artifact, prediction in zip(artifacts, predictions):
        result = Bunch(dict(
            id=str(uuid.uuid4()),
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=generated_timestamp,
            data={'height': str(prediction[0]), 'pos_pe': artifact['percentile']['99_percentile_neg_error'],
                      'neg_pe': artifact['percentile']['99_percentile_pos_error'], 'mae': artifact['percentile']['mae']}
                if 'percentile' in artifact and bool(artifact['percentile']) else
                {'height': str(prediction[0]), 'pos_pe': None, 'neg_pe': None, 'mae': None},
        ))
        res.results.append(result)
    return res


def post_height_result_object(artifacts, predictions, workflow_id, scan_id, generated_timestamp):
    res = artifact_level_result(artifacts, predictions, workflow_id, scan_id, generated_timestamp)
    res_object = bunch_object_to_json_object(res)
    ml_api.post_results(res_object)


def get_mean_scan_results(predictions):
    return str(np.mean(predictions))


def scan_level_height_result_object(artifact_ids, predictions, workflow_id, scan_id, generated_timestamp):
    res = Bunch(dict(results=[]))
    result = Bunch(dict(
        id=f"{uuid.uuid4()}",
        scan=scan_id,
        workflow=workflow_id,
        source_artifacts=[id for id in artifact_ids],
        source_results=[],
        generated=generated_timestamp,
        data={'mean_height': get_mean_scan_results(predictions)}
    ))
    res.results.append(result)
    return res


def post_results(artifact_ids, predictions, workflow_id, scan_id, generated_timestamp):
    res = scan_level_height_result_object(artifact_ids, predictions, workflow_id, scan_id, generated_timestamp)
    res_object = bunch_object_to_json_object(res)
    ml_api.post_results(res_object)


def get_standing_laying_data(scan_id, standing_laying_workflow_id):
    ml_api.get_results(scan_id, standing_laying_workflow_id)


def main(req: func.HttpRequest,
         context: func.Context) -> str:
    logging.info('Python HTTP trigger function processed a request.')
    response_object = {
        'invocation_id' : context.invocation_id,
        'operation_id' : context.trace_context.trace_parent.split('-')[1],
        'id' : context.trace_context.trace_parent.split('-')[2]
    }

    scan_id = req.params.get('scan_id')
    if not scan_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_id = req_body.get('scan_id')
            workflow_name = req_body.get('workflow_name') if req_body.get('workflow_name') else environ['HEIGHT_PLAINCNN_WORKFLOW_NAME']
            workflow_version = req_body.get('workflow_version') if req_body.get('workflow_version') else environ['HEIGHT_PLAINCNN_WORKFLOW_VERSION']
            mean_workflow_name = req_body.get('mean_workflow_name') if req_body.get('mean_workflow_name') else environ['HEIGHT_PLAINCNN_MEAN_WORKFLOW_NAME']
            mean_workflow_version = req_body.get('mean_workflow_version') if req_body.get('mean_workflow_version') else environ['HEIGHT_PLAINCNN_MEAN_WORKFLOW_VERSION']
            # standing_laying_workflow_name = req_body.get('standing_laying_workflow_name') if req_body.get('standing_laying_workflow_name') else environ['STANDING_LAYING_WORKFLOW_NAME']
            # standing_laying_workflow_version = req_body.get('standing_laying_workflow_version') if req_body.get('standing_laying_workflow_version') else environ['STANDING_LAYING_WORKFLOW_VERSION']
            # service_name = req_body.get('service_name')
            standing_laying_result_for_order = req_body.get('results')['standing_laying_results']
    try:
        if scan_id:
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            scan_metadata = ml_api.get_scan_metadata(scan_id)
            # scan_id = scan_metadata['id']
            height_plaincnn_workflow_id, service_name = ml_api.get_workflow_id_and_service_name(workflow_name, workflow_version, get_service_name=True)
            height_plaincnn_mean_workflow_id = ml_api.get_workflow_id_and_service_name(mean_workflow_name, mean_workflow_version)
            # standing_laying_workflow_id, _ = ml_api.get_workflow_id_and_service_name(standing_laying_workflow_name, standing_laying_workflow_version)
            logging.info(f"starting height RG for scan id {scan_id}, {height_plaincnn_workflow_id}")

            dob, sex = ml_api.get_basic_person_info(scan_metadata['person'])
            # age_in_days = calculate_age(dob, scan_metadata['scan_start'])
            scan_version = scan_metadata['version']
            scan_type = scan_metadata['type']
            depth_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'depth']

            depthmaps = process_depthmaps(depth_artifacts, ml_api)
            predictions = get_height_prediction(depthmaps, service_name)
            artifact_ids = [depth_artifact['id'] for depth_artifact in depth_artifacts]
            error_stats_dict = {
                "age" : calculate_age(dob, scan_metadata['scan_start']),
                "scan_type": scan_type,
                "scan_version": scan_version,
                "workflow_name": workflow_name,
                "workflow_ver": workflow_version,
                "percentile_value": 99
            }
            # standing_laying_result_for_order = get_standing_laying_data(scan_id, standing_laying_workflow_id)
            logging.info(standing_laying_result_for_order)
            for artifact in depth_artifacts:
                logging.info(f"value for order no {artifact['order']} is {standing_laying_result_for_order.get(str(artifact['order']))}")
                error_stats_dict['standing_laying'] = standing_laying_result_for_order.get(str(artifact['order']))
                artifact['percentile'] = error_stats_api.get_percentile_from_error_stats(error_stats_dict)

            post_height_result_object(depth_artifacts, predictions, height_plaincnn_workflow_id, scan_id, generated_timestamp)

            post_results(artifact_ids, predictions, height_plaincnn_mean_workflow_id, scan_id, generated_timestamp)

            # keys_wanted = ['id'] # , 'blurred_image']
            # mean_workflow_input = {
            #     "artifact_ids" : [depth_artifact['id'] for depth_artifact in depth_artifacts],
            #     "predictions": predictions,
            #     "scan_id" : scan_id,
            #     "scan_version" : scan_version,
            #     "scan_type" : scan_type
            # }
            response_object["status"] = 'Success'
            # response_object["results"] = mean_workflow_input
            logging.info(f"response object is {response_object}")
            response_json = json.dumps(response_object)
            return response_json
        else:
            response_object["status"] = 'Failed'
            response_object["exception"] = 'scan metadata required'
            logging.info(f"response object is {response_object}")
            return json.dumps(response_object)
    except Exception as error:
        response_object["status"] = 'Failed'
        response_object["exception"] = str(error)
        logging.info(f"response object is {response_object}")
        return json.dumps(response_object)
