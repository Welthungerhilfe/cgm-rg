import logging
import re

import azure.functions as func
import requests
from os import getenv
import numpy as np
from PIL import Image
import io
import cv2
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


def orient_image_using_scan_type(original_image, scan_type):
    if scan_type in [100, 101, 102]:
        rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)  # Standing
    elif scan_type in [200, 201, 202]:
        rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Laying
    else:
        logging.info("%s %s %s", "Provided scan type", scan_type, "not supported")
        logging.info("Keeping the image in the same orientation as provided")
        rotated_image = original_image
    return rotated_image


def pose_result_generation(file_id, scan_type):
    response = requests.get(url + f"/api/files/{file_id}", headers=headers)
    rgb_image = np.asarray(Image.open(io.BytesIO(response.content)))
    rotated_image = orient_image_using_scan_type(rgb_image, scan_type)


def main(req: func.HttpRequest,
         context: func.Context) -> str:
    logging.info('Python HTTP trigger function processed a request.')
    response_object = {
        'invocation_id' : context.invocation_id,
        'operation_id' : context.trace_context.trace_parent.split('-')[1],
        'id' : context.trace_context.trace_parent.split('-')[2]
    }
    results = req.params.get('results')
    # pose_prediction_workflow_id = req.params.get('pose_prediction_workflow_id')
    # pose_visualization_workflow_id = req.params.get('pose_visualization_workflow_id')
    # face_recognition_workflow_id = req.params.get('face_recognition_workflow_id')
    if not results:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            results = req_body.get('results')
            # pose_prediction_workflow_id = req_body.get('pose_prediction_workflow_id')
            # pose_visualization_workflow_id = req_body.get('pose_visualization_workflow_id')
            # face_recognition_workflow_id = req_body.get('face_recognition_workflow_id')

    if results:
        pose_prediction_workflow_id = get_workflow_id(getenv("POSE_PREDICTION_WORKFLOW_NAME"), getenv("POSE_PREDICTION_WORKFLOW_VERSION"))
        pose_visualization_workflow_id = get_workflow_id(getenv("POSE_VISUALIZATION_WORKFLOW_NAME"), getenv("POSE_VISUALIZATION_WORKFLOW_VERSION"))
        scan_id = results['scan_id']
        scan_type = results['scan_type']
        blurred_artifacts = results['blur_artifacts']

        # response = requests.get(url + f"/api/scans/{scan_id}", headers=headers)
        # scan_metadata = response.json()['scans']
        # scan_version = scan_metadata[0]['version']
        # scan_type = scan_metadata[0]['type']
        # results = scan_metadata[0]['results']

        # face_recognition_results = [result for result in results if result['workflow'] == face_recognition_workflow_id]
        # for face_recognition_result in face_recognition_results:
        #     result = pose_result_generation(face_recognition_results['file'], scan_type)

        for arti in blurred_artifacts:
            rgb_image = np.asarray(Image.open(io.BytesIO(arti['blurred_image'])))
            logging.info(f"successfully read image from bytes {rgb_image.shape}")

        return func.HttpResponse(f"Hello, {scan_id}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
