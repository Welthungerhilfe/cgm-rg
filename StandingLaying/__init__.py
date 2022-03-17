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
from utils.preprocessing import standing_laying_data_preprocessing

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
    scan_metadata = req.params.get('scan_metadata')
    # standing_laying_workflow_id = req.params.get('standing_laying_workflow_id')
    if not scan_metadata:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_metadata = req_body.get('scan_metadata')
            workflow_name = req_body.get('workflow_name')
            workflow_version = req_body.get('workflow_version')
            service_name = req_body.get('service_name')
    try:
        if scan_metadata:
            standing_laying_workflow_id = ml_api.get_workflow_id(workflow_name, workflow_version)
            scan_id = scan_metadata['id']
            scan_type = scan_metadata['type']
            logging.info(f"starting standing laying for scan id {scan_id}, {standing_laying_workflow_id}")
            rgb_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'rgb']
            predictions = []
            for rgb_artifact in rgb_artifacts:
                rgb_artifact['start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                img = standing_laying_data_preprocessing(rgb_artifact['file'], scan_type)
                prediction = get_standing_laying_prediction(img, service_name)
                predictions.append(prediction)
            predictions = np.array(predictions)
            # logging.info(f"predictions are {predictions}")
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            post_result_object(rgb_artifacts, predictions, generated_timestamp, scan_id, standing_laying_workflow_id)

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
