import logging
from os import environ

import azure.functions as func
import json
from utils.rest_api import MlApi
from utils.result_object_utils import bunch_object_to_json_object
import uuid
from bunch import Bunch
import numpy as np


ml_api = MlApi()


def get_mean_scan_results(predictions):
    return str(np.mean(predictions))


def scan_level_height_result_object(artifact_ids, predictions, workflow_id, scan_id):
    res = Bunch(dict(results=[]))
    result = Bunch(dict(
        id=f"{uuid.uuid4()}",
        scan=scan_id,
        workflow=workflow_id,
        source_artifacts=[id for id in artifact_ids],
        source_results=[],
        data={'mean_height': get_mean_scan_results(predictions)}
    ))
    res.results.append(result)
    return res


def post_results(artifact_ids, predictions, workflow_id, scan_id):
    res = scan_level_height_result_object(artifact_ids, predictions, workflow_id, scan_id)
    res_object = bunch_object_to_json_object(res)
    ml_api.post_results(res_object)


def main(req: func.HttpRequest,
         context: func.Context) -> str:
    logging.info('Python HTTP trigger function processed a request.')
    response_object = {
        'invocation_id' : context.invocation_id,
        'operation_id' : context.trace_context.trace_parent.split('-')[1],
        'id' : context.trace_context.trace_parent.split('-')[2]
    }

    results = req.params.get('results')
    if not results:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            results = req_body.get('results')
            workflow_name = req_body.get('workflow_name') if req_body.get('workflow_name') else environ['HEIGHT_PLAINCNN_MEAN_WORKFLOW_NAME']
            workflow_version = req_body.get('workflow_version') if req_body.get('workflow_version') else environ['HEIGHT_PLAINCNN_MEAN_WORKFLOW_VERSION']
    try:
        if results:
            scan_id = results['scan_id']
            height_plaincnn_mean_workflow_id = ml_api.get_workflow_id(workflow_name, workflow_version)
            logging.info(f"starting height RG for scan id {scan_id}, {height_plaincnn_mean_workflow_id}")
            artifact_ids = results['artifact_ids']
            predictions = results['predictions']
            post_results(artifact_ids, predictions, height_plaincnn_mean_workflow_id, scan_id)
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
        # logging.info(f"response object is {response_object}")
        return json.dumps(response_object)
