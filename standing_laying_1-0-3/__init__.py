import logging

import azure.functions as func
import requests
from os import getenv
import numpy as np
from datetime import datetime
import json
from utils.rest_api import MlApi
from utils.inference import get_standing_laying_prediction
from utils.result_object_utils import bunch_object_to_json_object, prepare_result_object
from utils.preprocessing import standing_laying_data_preprocessing_tf

ml_api = MlApi()


def post_result_object(rgb_artifacts, predictions, generated_timestamp, scan_id, workflow_id):
    """Post the result object to the API"""
    res = prepare_result_object(rgb_artifacts, predictions, generated_timestamp, scan_id, workflow_id, 'standing')
    res_object = bunch_object_to_json_object(res)
    # print(res_object)
    ml_api.post_results(res_object)


def main(req: func.HttpRequest,
         context: func.Context) -> str:
    logging.info('Python HTTP trigger function processed a request.')
    response_object = {
        'invocation_id' : context.invocation_id,
        'operation_id' : context.trace_context.trace_parent.split('-')[1],
        'id' : context.trace_context.trace_parent.split('-')[2]
    }
    scan_id = req.params.get('scan_id')
    # standing_laying_workflow_id = req.params.get('standing_laying_workflow_id')
    if not scan_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_id = req_body.get('scan_id')
            workflow_name = req_body.get('workflow_name')
            workflow_version = req_body.get('workflow_version')
            # service_name = req_body.get('service_name')
    try:
        if scan_id:
            scan_metadata = ml_api.get_scan_metadata(scan_id)
            standing_laying_workflow_id, service_name = ml_api.get_workflow_id_and_service_name(workflow_name, workflow_version, get_service_name=True)
            # scan_id = scan_metadata['id']
            scan_type = scan_metadata['type']
            logging.info(f"starting standing laying for scan id {scan_id}, {standing_laying_workflow_id}")
            rgb_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'rgb']
            predictions = []
            standing_laying_result_for_order = {}
            for rgb_artifact in rgb_artifacts:
                rgb_artifact['start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                img = standing_laying_data_preprocessing_tf(rgb_artifact['file'], ml_api)
                prediction = get_standing_laying_prediction(img, service_name)
                standing_laying_result_for_order[rgb_artifact['order']] = prediction[0]
                predictions.append(prediction)
            predictions = np.array(predictions)
            # logging.info(f"predictions are {predictions}")
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            post_result_object(rgb_artifacts, predictions, generated_timestamp, scan_id, standing_laying_workflow_id)

            response_object['standing_laying_result_for_order'] = standing_laying_result_for_order
            response_object["status"] = 'Success'
            logging.info(f"response object is {response_object}")
            return json.dumps(response_object)
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
