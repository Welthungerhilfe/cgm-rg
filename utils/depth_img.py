import uuid
from datetime import datetime
from pathlib import Path
import io

import cv2
import matplotlib.pyplot as plt
from bunch import Bunch
import numpy as np

from utils.preprocessing import load_depth_from_file, prepare_depthmap
from utils.result_utils import bunch_object_to_json_object, get_workflow, check_if_results_exists
from utils.constants import DEPTH_IMG_WORKFLOW_NAME, DEPTH_IMG_WORKFLOW_VERSION


def depth_img_flow(cgm_api, scan_id, artifacts, version, workflows, results):
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    depth_workflow = get_workflow(workflows, DEPTH_IMG_WORKFLOW_NAME, DEPTH_IMG_WORKFLOW_VERSION)
    if not check_if_results_exists(results, depth_workflow['id']):
        preprocess_and_post_depthmap(cgm_api, artifacts, version)
        post_result_object(cgm_api, scan_id, artifacts, generated_timestamp, depth_workflow['id'])


def preprocess_and_post_depthmap(cgm_api, artifacts, version):
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth_from_file(artifact['raw_file'])
        if 'ir' in version:
            depthmap = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            depthmap = np.rot90(depthmap, k=-1)
        else:
            depthmap = prepare_depthmap(data, width, height, depth_scale)
        
        bin_file = save_plot_as_binary(depthmap)
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


def save_plot_as_binary(depthmap):
    plt.imshow(depthmap)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Move the buffer cursor to the beginning
    buffer.seek(0)

    # Get the binary data
    binary_data = buffer.read()

    return binary_data
