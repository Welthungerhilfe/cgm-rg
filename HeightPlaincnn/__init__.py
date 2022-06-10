import logging

import azure.functions as func
import requests
from os import getenv
from datetime import datetime
import uuid
from bunch import Bunch
import json
from utils.preprocessing import process_depthmaps
from utils.rest_api import MlApi
from utils.inference import get_height_prediction
from utils.result_object_utils import bunch_object_to_json_object

ml_api = MlApi()


def artifact_level_result(artifacts, predictions, workflow_id, scan_id):
    """Prepare artifact level height result object"""
    res = Bunch(dict(results=[]))
    for artifact, prediction in zip(artifacts, predictions):
        result = Bunch(dict(
            id=str(uuid.uuid4()),
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            data={'height': str(prediction[0])}
        ))
        res.results.append(result)
    return res


def post_height_result_object(artifacts, predictions, workflow_id, scan_id):
    res = artifact_level_result(artifacts, predictions, workflow_id, scan_id)
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

    scan_id = req.params.get('scan_id')
    if not scan_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_id = req_body.get('scan_id')
            workflow_name = req_body.get('workflow_name')
            workflow_version = req_body.get('workflow_version')
            service_name = req_body.get('service_name')
    try:
        if scan_id:
            scan_metadata = ml_api.get_scan_metadata(scan_id)
            # scan_id = scan_metadata['id']
            height_plaincnn_workflow_id = ml_api.get_workflow_id(workflow_name, workflow_version)
            logging.info(f"starting height RG for scan id {scan_id}, {height_plaincnn_workflow_id}")

            scan_version = scan_metadata['version']
            scan_type = scan_metadata['type']
            depth_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'depth']

            depthmaps = process_depthmaps(depth_artifacts, ml_api)
            predictions = get_height_prediction(depthmaps, service_name)

            post_height_result_object(depth_artifacts, predictions, height_plaincnn_workflow_id, scan_id)
            # keys_wanted = ['id'] # , 'blurred_image']
            mean_workflow_input = {
                "artifact_ids" : [depth_artifact['id'] for depth_artifact in depth_artifacts],
                "predictions": predictions,
                "scan_id" : scan_id,
                "scan_version" : scan_version,
                "scan_type" : scan_type
            }
            response_object["status"] = 'Success'
            response_object["results"] = mean_workflow_input
            # logging.info(f"response object is {response_object}")
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
        # logging.info(f"response object is {response_object}")
        return json.dumps(response_object)
