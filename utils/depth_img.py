import uuid
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from bunch import Bunch

from utils.preprocessing import load_depth_from_file, prepare_depthmap
from utils.result_utils import bunch_object_to_json_object, get_workflow, check_if_results_exists
from utils.constants import DEPTH_IMG_WORKFLOW_NAME, DEPTH_IMG_WORKFLOW_VERSION


def depth_img_flow(cgm_api, scan_id, artifacts, workflows, results):
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    depth_workflow = get_workflow(workflows, DEPTH_IMG_WORKFLOW_NAME, DEPTH_IMG_WORKFLOW_VERSION)
    if not check_if_results_exists(results, depth_workflow['id']):
        preprocess_and_post_depthmap(cgm_api, artifacts)
        post_result_object(cgm_api, scan_id, artifacts, generated_timestamp, depth_workflow['id'])


def preprocess_and_post_depthmap(cgm_api, artifacts):
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth_from_file(artifact['raw_file'])

        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = depthmap / 2.0
        scaled_depthmap = depthmap * 255.0
        _, bin_file = cv2.imencode('.JPEG', scaled_depthmap)
        bin_file = bin_file.tostring()
        artifact['depth_image_file_id'] = cgm_api.post_files(bin_file, 'rgb')


def prepare_result_object(artifacts, scan_id, generated_timestamp, workflow_id):
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            file=artifact['depth_image_file_id'],
            generated=generated_timestamp,
            start_time=generated_timestamp,
            end_time=generated_timestamp
        ))
        res.results.append(result)
    return res


def post_result_object(cgm_api, scan_id, artifacts, generated_timestamp, workflow_id):
    res = prepare_result_object(artifacts, scan_id, generated_timestamp, workflow_id)
    res_object = bunch_object_to_json_object(res)
    cgm_api.post_results(res_object)
