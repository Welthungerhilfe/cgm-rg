import logging
import re

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


standing_scan_type = ["101", "102", "100"]
laying_scan_type = ["201", "202", "200"]


def get_workflow_id(workflow_name, workflow_version):
    response = requests.get(url + f"/api/workflows", headers=headers)
    if response.status_code != 200:
        logging.info(f"error getting workflows {response.content}")
    workflows = response.json()['workflows']
    workflow = [workflow for workflow in workflows if workflow['name'] == workflow_name and workflow['version'] == workflow_version]

    return workflow[0]['id']


def blur_img_transformation_using_scan_version_and_scan_type(rgb_image, scan_version, scan_type):
    if scan_version in ["v0.7"]:
        # Make the image smaller, The limit of cgm-api to post an image is 500 KB.
        # Some of the images of v0.7 is greater than 500 KB
        rgb_image = cv2.resize(
            rgb_image, (0, 0), fx=1.0 / 1.3, fy=1.0 / 1.3)

    # print("scan_version is ", self.scan_version)
    image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

    # if self.scan_version in ["v0.1", "v0.2", "v0.4", "v0.5", "v0.6", "v0.7", "v0.8", "v0.9", "v1.0"]:
    # The images are provided in 90degrees turned. Here we rotate 90 degress to
    # the right.
    if scan_type in standing_scan_type:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif scan_type in laying_scan_type:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


    return image


def orient_img(image, scan_type):
    # The images are rotated 90 degree clockwise for standing children
    # and 90 degree anticlock wise for laying children to make children
    # head at top and toe at bottom
    if scan_type in standing_scan_type:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif scan_type in laying_scan_type:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reorient_back(image, scan_type):
    if scan_type in standing_scan_type:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif scan_type in laying_scan_type:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image


def blur_face(file_id: str, scan_version, scan_type):
    """Run face blur on given source_path
    Returns:
        bool: True if blurred otherwise False
    """
    response = requests.get(url + f"/api/files/{file_id}", headers=headers)
    rgb_image = np.asarray(Image.open(io.BytesIO(response.content)))

    image = blur_img_transformation_using_scan_version_and_scan_type(rgb_image, scan_version, scan_type)
    image = orient_img(image, scan_type)

    height, width, channels = image.shape
    logging.info(f"{height}, {width}, {channels}")

    resized_height = 500.0
    resize_factor = height / resized_height
    # resized_width = width / resize_factor
    # resized_height, resized_width = int(resized_height), int(resized_width)

    # Scale image down for faster prediction.
    small_image = cv2.resize(
        image, (0, 0), fx=1.0 / resize_factor, fy=1.0 / resize_factor)

    # Find face locations.
    face_locations = face_recognition.face_locations(small_image, model="cnn")

    faces_detected = len(face_locations)
    logging.info("%s %s", faces_detected, "face locations found and blurred for path:")

    # Blur the image.
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was
        # scaled to 1/4 size
        top *= resize_factor
        right *= resize_factor
        bottom *= resize_factor
        left *= resize_factor
        top, right, bottom, left = int(top), int(right), int(bottom), int(left)

        # Extract the region of the image that contains the face.
        # TODO rotate -codinate

        face_image = image[top:bottom, left:right]

        # Blur the face image.
        face_image = cv2.GaussianBlur(
            face_image, ksize=(99, 99), sigmaX=30)

        # Put the blurred face region back into the frame image.
        image[top:bottom, left:right] = face_image

    image = reorient_back(image, scan_type)

    # Write image to hard drive.
    rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

    # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
    logging.info("%s %s", len(face_locations), "face locations found and blurred for path:")
    return rgb_image, True, faces_detected


def post_files(blurred_image):
    _, bin_file = cv2.imencode('.JPEG', blurred_image)
    bin_file = bin_file.tostring()

    files = {
        'file': bin_file,
        'filename': 'test.jpg',
    }

    response = requests.post(url + '/api/files?storage=result', files=files, headers=headers)
    file_id = response.content.decode('utf-8')

    return file_id, response.status_code


def post_blur_files(artifacts):
    for artifact in artifacts:
        blur_id_from_post_request, post_status = post_files(artifact['blurred_image'])
        if post_status == 201:
            artifact['blur_id_from_post_request'] = blur_id_from_post_request
            artifact['generated_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')


def bunch_object_to_json_object(bunch_object):
    """Convert given bunch object to json object"""
    json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
    json_object = json.loads(json_string)
    return json_object


def prepare_result_object(artifacts, scan_id, workflow_id):
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
            start_time=artifact['blur_start_time'],
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res


def prepare_faces_result_object(artifacts, scan_id, workflow_id):
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
            data={'faces_detected': str(artifact['faces_detected'])},
            start_time=artifact['blur_start_time'],
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res


def post_results(result_json_obj):
    """Post the result object produced while Result Generation using POST /results"""
    response = requests.post(url + '/api/results', json=result_json_obj, headers=headers)
    logging.info("%s %s", "Status of post result response:", response.status_code)
    return response.status_code


def post_blur_result_object(artifacts, scan_id, face_recognition_workflow_id, face_detection_workflow_id):
    """Post the result object to the API"""
    res = prepare_result_object(artifacts, scan_id, face_recognition_workflow_id)
    res_object = bunch_object_to_json_object(res)
    if post_results(res_object) == 201:
        logging.info("%s %s", "successfully post blur results:", res_object)

    faces_res = prepare_faces_result_object(artifacts, scan_id, face_detection_workflow_id)
    faces_res_object = bunch_object_to_json_object(faces_res)
    if post_results(faces_res_object) == 201:
        logging.info("%s %s", "successfully post faces detected results:", faces_res_object)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    scan_id = req.params.get('scan_id')
    # face_recognition_workflow_id = req.params.get('face_recognition_workflow_id')
    # face_detection_workflow_id = req.params.get('face_detection_workflow_id')
    if not scan_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_id = req_body.get('scan_id')
            # face_recognition_workflow_id = req_body.get('face_recognition_workflow_id')
            # face_detection_workflow_id = req_body.get('face_detection_workflow_id')

    if scan_id:
        face_recognition_workflow_id = get_workflow_id(getenv("FACE_RECOGNITION_WORKFLOW_NAME"), getenv("FACE_RECOGNITION_WORKFLOW_VERSION"))
        face_detection_workflow_id = get_workflow_id(getenv("FACE_DETECTION_WORKFLOW_NAME"), getenv("FACE_DETECTION_WORKFLOW_VERSION"))
        logging.info(f"starting face blur for scan id {scan_id}, {face_recognition_workflow_id}, {face_detection_workflow_id}")

        response = requests.get(url + f"/api/scans?scan_id={scan_id}", headers=headers)
        scan_metadata = response.json()['scans']
        scan_version = scan_metadata[0]['version']
        scan_type = scan_metadata[0]['type']
        rgb_artifacts = [a for a in scan_metadata[0]['artifacts'] if a['format'] == 'rgb']
        for rgb_artifact in rgb_artifacts:
            rgb_artifact['blur_start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            blur_img_binary, blur_status, faces_detected = blur_face(rgb_artifact['file'], scan_version, scan_type)
            if blur_status:
                rgb_artifact['blurred_image'] = blur_img_binary
                rgb_artifact['faces_detected'] = faces_detected

        post_blur_files(rgb_artifacts)
        post_blur_result_object(rgb_artifacts, scan_id, face_recognition_workflow_id, face_detection_workflow_id)

        # keys_wanted = ['id', 'blur_id_from_post_request']
        # pose_input = {
        #     "blur_artifacts" : [{k: rgb_artifact[k] for k in keys_wanted} for rgb_artifact in rgb_artifacts],
        #     "scan_id" : scan_id,
        #     "scan_version" : scan_version,
        #     "scan_type" : scan_type
        # }

        # return pose_input
        return func.HttpResponse(f"Hello, {scan_id}. This HTTP triggered function executed successfully.")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
