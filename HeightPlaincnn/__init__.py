import logging

import azure.functions as func
import requests
from os import getenv
import numpy as np
from PIL import Image
import io
import cv2
import face_recognition
from datetime import datetime
import uuid
from bunch import Bunch
import json


url = getenv('URL')
headers = {'X-API-Key': getenv('API_KEY')}


def get_workflow_id(workflow_name, workflow_version):
    response = requests.get(url + f"/api/workflows", headers=headers)
    if response.status_code != 200:
        logging.info(f"error getting workflows {response.content}")
    workflows = response.json()['workflows']
    workflow = [workflow for workflow in workflows if workflow['name'] == workflow_name and workflow['version'] == workflow_version]

    return workflow[0]['id']


def post_height_result_object():
    pass


def main(req: func.HttpRequest,
         context: func.Context) -> str:
    logging.info('Python HTTP trigger function processed a request.')
    response_object = {
        'invocation_id' : context.invocation_id,
        'operation_id' : context.trace_context.trace_parent.split('-')[1],
        'id' : context.trace_context.trace_parent.split('-')[2]
    }

    scan_metadata = req.params.get('scan_metadata')
    if not scan_metadata:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_metadata = req_body.get('scan_metadata')
    try:
        if scan_metadata:
            scan_id = scan_metadata['id']
            height_plaincnn_workflow_id = get_workflow_id(getenv("HEIGHT_PLAINCNN_WORKFLOW_NAME"), getenv("HEIGHT_PLAINCNN_WORKFLOW_VERSION"))
            logging.info(f"starting face blur for scan id {scan_id}, {height_plaincnn_workflow_id}")

            scan_version = scan_metadata['version']
            scan_type = scan_metadata['type']
            depth_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'rgb']



            post_height_result_object(depth_artifacts, scan_id, height_plaincnn_workflow_id)

            keys_wanted = ['id', 'blur_id_from_post_request'] # , 'blurred_image']
            pose_input = {
                "blur_artifacts" : [{k: rgb_artifact[k] for k in keys_wanted} for rgb_artifact in rgb_artifacts],
                "scan_id" : scan_id,
                "scan_version" : scan_version,
                "scan_type" : scan_type
            }
            # for blur_arti in pose_input['blur_artifacts']:
            #     blur_arti['blurred_image'] = blur_arti['blurred_image'].tolist()

            response_object["status"] = 'Success'
            response_object["results"] = pose_input
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
