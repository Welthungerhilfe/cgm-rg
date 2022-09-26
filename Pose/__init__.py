import imp
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
from utils.rest_api import MlApi
from utils.preprocessing import read_image, orient_image_using_scan_type, preprocess_image_for_pose
from pose_prediction.code.utils.utils import draw_pose
from utils.result_object_utils import bunch_object_to_json_object
from utils.inference import get_pose_boxes_prediction, get_pose_prediction


ml_api = MlApi()


def prepare_no_of_person_result_object(artifacts, scan_id, pose_workflow_id, generated_timestamp):
    """Prepare result object for results generated"""
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        no_of_pose_result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=pose_workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=generated_timestamp,
            data={'no of person using pose': str(artifact['no_of_pose_detected'])},
            start_time=artifact['pose_start_time'],
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(no_of_pose_result)
        for i in range(0, artifact['no_of_pose_detected']):
            pose_score_results = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=scan_id,
                workflow=pose_workflow_id,
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'Pose Scores': str(artifact['pose_score'][i]),
                        'Pose Results': str(artifact['pose_result'][i])},
                start_time=artifact['pose_start_time'],
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(pose_score_results)
    return res


def post_result_object(rgb_artifacts, scan_id, workflow_id, generated_timestamp):
    """Post the result object to the API"""
    res = prepare_no_of_person_result_object(rgb_artifacts, scan_id, workflow_id, generated_timestamp)
    res_object = bunch_object_to_json_object(res)
    # print(res_object)
    ml_api.post_results(res_object)


def main(req: func.HttpRequest,
         context: func.Context) -> str:
    logging.info('Python HTTP trigger function processed a request.')
    response_object = {
        'invocation_id' : context.invocation_id,
        # 'operation_id' : context.trace_context.trace_parent.split('-')[1],
        # 'id' : context.trace_context.trace_parent.split('-')[2]
    }
    scan_id = req.params.get('scan_id')
    # pose_prediction_workflow_id = req.params.get('pose_prediction_workflow_id')
    # pose_visualization_workflow_id = req.params.get('pose_visualization_workflow_id')
    # face_recognition_workflow_id = req.params.get('face_recognition_workflow_id')
    if not scan_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            scan_id = req_body.get('scan_id')
            # pose_prediction_workflow_id = req_body.get('pose_prediction_workflow_id')
            # pose_visualization_workflow_id = req_body.get('pose_visualization_workflow_id')
            # face_recognition_workflow_id = req_body.get('face_recognition_workflow_id')
    try:
        if scan_id:
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            pose_prediction_workflow_id, service_names = ml_api.get_workflow_id_and_service_name(getenv("POSE_PREDICTION_WORKFLOW_NAME"), getenv("POSE_PREDICTION_WORKFLOW_VERSION"), get_service_name=True)
            # pose_visualization_workflow_id = get_workflow_id(getenv("POSE_VISUALIZATION_WORKFLOW_NAME"), getenv("POSE_VISUALIZATION_WORKFLOW_VERSION"))
            # scan_id = results['scan_id']
            # scan_type = results['scan_type']
            # blurred_artifacts = results['blur_artifacts']

            scan_metadata = ml_api.get_scan_metadata(scan_id)
            scan_type = scan_metadata['type']
            logging.info(f"starting pose prediction for scan id {scan_id}, {pose_prediction_workflow_id}")
            rgb_artifacts = [a for a in scan_metadata['artifacts'] if a['format'] == 'rgb']
            pose_result_for_order = {}
            for artifact in rgb_artifacts:
                original_image, shape = read_image(artifact['file'], ml_api)
                rotated_image = orient_image_using_scan_type(original_image, scan_type)
                box_model_input, rotated_image_rgb = preprocess_image_for_pose(rotated_image)
                pose_box_results = get_pose_boxes_prediction(box_model_input, service_names['box_model_service_name'])
                pose_results = get_pose_prediction(rotated_image, pose_box_results, shape, scan_type, service_names['pose_model_service_name'])
                # pose_result_of_artifact = {'no_of_body_pose_detected': len(pose_box_results['pred_boxes']), 'pose_result': pose_results}

                # # rgb_image = np.asarray(Image.open(io.BytesIO(arti['blurred_image'])))
                # # logging.info(f"successfully read image from bytes {rgb_image.shape}")
                # artifact['pose_start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                # no_of_pose_detected, pose_score, pose_result, _ = inference_artifact(
                #     pose_prediction, artifact['file'], scan_type, ml_api)
                # logging.info("%s %s %s %s ", "pose_score", "no_of_pose_detected", pose_score, no_of_pose_detected)
                artifact['no_of_pose_detected'] = len(pose_box_results['pred_boxes'])
                artifact['pose_score'] = [ pose_result['body_pose_score'] for pose_result in pose_results]
                artifact['pose_result'] = pose_results
                pose_result_for_order[artifact['order']] = {"pose_score": artifact['pose_score'], "no_of_pose_detected": artifact['no_of_pose_detected']}

            post_result_object(rgb_artifacts, scan_id, pose_prediction_workflow_id, generated_timestamp)

            response_object['pose_result_for_order'] = pose_result_for_order
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
