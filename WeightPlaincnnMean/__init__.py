import json
import uuid
import logging
import numpy as np
from os import environ
from bunch import Bunch
import azure.functions as func
from utils.rest_api import MlApi
from utils.result_object_utils import bunch_object_to_json_object

ml_api = MlApi()


def get_mean_scan_results(predictions):
    return str(np.mean(predictions))


def scan_level_weight_result_object(artifact_ids, predictions, workflow_id, scan_id):
    res = Bunch(dict(results=[]))
    result = Bunch(dict(
        id=f"{uuid.uuid4()}",
        scan=scan_id,
        workflow=workflow_id,
        source_artifacts=[id for id in artifact_ids],
        source_results=[],
        data={'mean_weight': get_mean_scan_results(predictions)}
    ))
    res.results.append(result)
    return res

def post_results(artifact_ids, predictions, workflow_id, scan_id):
    res = scan_level_weight_result_object(artifact_ids, predictions, workflow_id, scan_id)
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
            workflow_name = req_body.get('workflow_name') if req_body.get('workflow_name') else environ['WEIGHT_PLAINCNN_MEAN_WORKFLOW_NAME']
            workflow_version = req_body.get('workflow_version') if req_body.get('workflow_version') else environ['WEIGHT_PLAINCNN_MEAN_WORKFLOW_VERSION']
    try:
        if results:
            scan_id = results['scan_id']
            weight_plaincnn_mean_workflow_id = ml_api.get_workflow_id(workflow_name, workflow_version)
            logging.info(f"starting weight RG for scan id {scan_id}, {weight_plaincnn_mean_workflow_id}")
            artifact_ids = results['artifact_ids']
            predictions = results['predictions']
            post_results(artifact_ids, predictions, weight_plaincnn_mean_workflow_id, scan_id)
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