import pickle
import uuid
from datetime import datetime
import cv2

from bunch import Bunch

from utils.preprocessing import blur_input_face_api
from utils.inference import ms_face_api
from utils.result_utils import bunch_object_to_json_object, get_workflow, check_if_results_exists
from utils.constants import BLUR_WORKFLOW_NAME, BLUR_WORKFLOW_VERSION, FACE_DETECTION_WORKFLOW_NAME, FACE_DETECTION_WORKFLOW_VERSION

MAX_BATCH_SIZE = 13

def run_blur_flow(cgm_api, scan_id, artifacts, workflows, scan_type, scan_version, results):
    blur_workflow = get_workflow(workflows, BLUR_WORKFLOW_NAME, BLUR_WORKFLOW_VERSION)
    faces_workflow = get_workflow(workflows, FACE_DETECTION_WORKFLOW_NAME, FACE_DETECTION_WORKFLOW_VERSION)
    if not (check_if_results_exists(results, blur_workflow['id']) and check_if_results_exists(results, faces_workflow['id'])):
        for artifact in artifacts:
            in_image = blur_input_face_api(artifact['raw_file'], scan_type)
            artifact['blurred_image'], artifact['faces_detected'] = ms_face_api(in_image, scan_type)
        post_results(artifacts, cgm_api, scan_id, blur_workflow['id'], faces_workflow['id'])


def post_results(artifacts, cgm_api, scan_id, blur_workflow_id, faces_workflow_id):
    post_blur_files(cgm_api, artifacts)
    blur_files_results = prepare_result_object(artifacts, scan_id, blur_workflow_id)
    blur_files_results_json_object = bunch_object_to_json_object(blur_files_results)
    cgm_api.post_results(blur_files_results_json_object)

    faces_results = prepare_faces_result_object(artifacts, scan_id, faces_workflow_id)
    faces_results_json_object = bunch_object_to_json_object(faces_results)
    cgm_api.post_results(faces_results_json_object)


def post_blur_files(cgm_api, artifacts):
    """Post the blurred file to the API"""
    for artifact in artifacts:
        _, bin_file = cv2.imencode('.JPEG', artifact['blurred_image'])
        bin_file = bin_file.tostring()
        artifact['blur_id_from_post_request'] = cgm_api.post_files(bin_file, 'rgb')


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
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
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
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            data={'faces_detected': str(len(artifact['faces_detected'])), 'face_attributes': artifact['faces_detected']},
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res
