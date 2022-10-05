import logging
import re
from turtle import pos

import azure.functions as func
import concurrent.futures
import requests
from os import getenv
import cv2
from datetime import datetime
import uuid
from bunch import Bunch
import json
from PIL import Image
import io
import numpy as np
from utils.rest_api import MlApi
from utils.result_object_utils import bunch_object_to_json_object # , prepare_result_object
from utils.preprocessing import blur_face
from utils.inference import get_face_locations
from itertools import repeat

ml_api = MlApi()

standing_scan_type = ["101", "102", "100"]
laying_scan_type = ["201", "202", "200"]


def post_blur_files(artifacts, results_dict):
    for artifact in artifacts:
        _, bin_file = cv2.imencode('.JPEG', results_dict[artifact['id']]['blurred_image'])
        bin_file = bin_file.tostring()
        blur_id_from_post_request = ml_api.post_files(bin_file, 'rgb')        
        artifact['blur_id_from_post_request'] = blur_id_from_post_request
        artifact['generated_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')


def prepare_result_object(artifacts, scan_id, workflow_id, start_time):
    """Prepare result object for results generated"""
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            file=artifact['blur_id_from_post_request'],
            generated=artifact['generated_timestamp'],
            start_time=start_time,
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res


def prepare_faces_result_object(artifacts, scan_id, workflow_id, results_dict, start_time):
    """Prepare result object for results generated"""
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=artifact['generated_timestamp'],
            data={'faces_detected': str(results_dict[artifact['id']]['faces_detected'])},
            start_time=start_time,
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res


def post_blur_result_object(artifacts, scan_id, face_recognition_workflow_id, face_detection_workflow_id, results_dict, start_time):
    """Post the result object to the API"""
    res = prepare_result_object(artifacts, scan_id, face_recognition_workflow_id, start_time)
    res_object = bunch_object_to_json_object(res)
    ml_api.post_results(res_object)

    faces_res = prepare_faces_result_object(artifacts, scan_id, face_detection_workflow_id, results_dict, start_time)
    faces_res_object = bunch_object_to_json_object(faces_res)
    ml_api.post_results(faces_res_object)


def run_face_blurring(rgb_artifact, scan_version, scan_type, service_name):
    blur_img_binary, blur_status, faces_detected = blur_face(rgb_artifact['file'], scan_version, scan_type, ml_api, service_name)
    print(blur_status, faces_detected)
    if blur_status:
        return {'blurred_image': blur_img_binary, 'faces_detected': faces_detected}
        # rgb_artifact['blurred_image'] = blur_img_binary
        # rgb_artifact['faces_detected'] = faces_detected
    # return rgb_artifact


def get_blurred_face(artifact, scan_type, scan_version, service_name):
    response = ml_api.get_files(artifact['file'])
    rgb_image = np.asarray(Image.open(io.BytesIO(response)))

    faces_detected, blur_img_binary = get_face_locations(rgb_image, scan_type, scan_version, service_name)

    return {artifact['id']: {'faces_detected': faces_detected, 'blurred_image': blur_img_binary}}


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
    try:
        if scan_id:
            scan_metadata = ml_api.get_scan_metadata(scan_id)
            face_recognition_workflow_id = ml_api.get_workflow_id_and_service_name(getenv("FACE_RECOGNITION_WORKFLOW_NAME"), getenv("FACE_RECOGNITION_WORKFLOW_VERSION"))
            face_detection_workflow_id, service_name = ml_api.get_workflow_id_and_service_name(getenv("FACE_DETECTION_WORKFLOW_NAME"), getenv("FACE_DETECTION_WORKFLOW_VERSION"), get_service_name=True)
            logging.info(f"starting face blur for scan id {scan_id}, {face_recognition_workflow_id}, {face_detection_workflow_id}")

            scan_version = scan_metadata['version']
            scan_type = scan_metadata['type']
            start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            rgb_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'rgb']

            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
                result = executor.map(get_blurred_face, rgb_artifacts, repeat(scan_type), repeat(scan_version), repeat(service_name))
            result_di = { k: v for r in result for k, v in r.items()}

            post_blur_files(rgb_artifacts, result_di)
            post_blur_result_object(rgb_artifacts, scan_id, face_recognition_workflow_id, face_detection_workflow_id, result_di, start_time)

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
        logging.info(f"response object is {response_object}")
        return json.dumps(response_object)
