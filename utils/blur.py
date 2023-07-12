import pickle
import uuid
from datetime import datetime
import cv2

from bunch import Bunch

from utils.preprocessing import blur_input
from utils.inference import get_blur_prediction
from utils.result_utils import bunch_object_to_json_object, get_workflow
from utils.constants import BLUR_WORKFLOW_NAME, BLUR_WORKFLOW_VERSION, FACE_DETECTION_WORKFLOW_NAME, FACE_DETECTION_WORKFLOW_VERSION

MAX_BATCH_SIZE = 13

def run_blur_flow(cgm_api, scan_id, artifacts, workflows, scan_type, scan_version):
    blur_workflow = get_workflow(workflows, BLUR_WORKFLOW_NAME, BLUR_WORKFLOW_VERSION)
    faces_workflow = get_workflow(workflows, FACE_DETECTION_WORKFLOW_NAME, FACE_DETECTION_WORKFLOW_VERSION)
    blur_input_data = blur_input(artifacts)
    total_data = len(blur_input_data)
    if total_data <= MAX_BATCH_SIZE:
        pickle_input = pickle.dumps([blur_input_data, scan_type, scan_version])
        predictions = get_blur_prediction(pickle_input)
    else:
        i = 0
        predictions = []
        for i in range(0, total_data, MAX_BATCH_SIZE):
            pickle_input = pickle.dumps([blur_input_data[i:i+MAX_BATCH_SIZE], scan_type, scan_version])
            predictions.extend(get_blur_prediction(pickle_input))
            i += MAX_BATCH_SIZE
    for (artifact, prediction) in zip(artifacts, predictions):
        artifact['blurred_image'] = prediction[0]
        artifact['faces_detected'] = prediction[1]
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
            data={'faces_detected': str(artifact['faces_detected'])},
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res
