import pickle
import uuid
from datetime import datetime
import json
from copy import deepcopy

import numpy as np
from bunch import Bunch
import cv2

from utils.constants import APP_POSE_VISUALIZE_WORKFLOW_NAME, APP_POSE_VISUALIZE_WORKFLOW_VERSION, APP_POSE_WORKFLOW_NAME, APP_POSE_WORKFLOW_VERSION
from utils.constants import MLKIT_KEYPOINT_INDEXES, MLKIT_SKELETON, MLKIT_NUM_KPTS, MLKIT_BODY_JOINTS, MlkitColors
from utils.result_utils import bunch_object_to_json_object, get_workflow, check_if_results_exists


def run_app_pose_visualization_flow(cgm_api, scan_id, artifacts, workflows, scan_type, scan_version, results):
    app_pose_visualize_workflow = get_workflow(workflows, APP_POSE_VISUALIZE_WORKFLOW_NAME, APP_POSE_VISUALIZE_WORKFLOW_VERSION)
    app_pose_workflow = get_workflow(workflows, APP_POSE_WORKFLOW_NAME, APP_POSE_WORKFLOW_VERSION)
    if not check_if_results_exists(results, app_pose_visualize_workflow['id']):
        workflow_id = app_pose_visualize_workflow['id']
        app_pose_workflow_id = app_pose_workflow['id']
        # results = cgm_api.get_results_for_workflow(scan_id, app_pose_workflow_id)
        app_pose_results = [r for r in results if r['workflow']==app_pose_workflow_id]
        app_pose_results_dict = {r['source_artifacts'][0]: json.loads(r['data']['poseCoordinates']) for r in app_pose_results}
        for artifact in artifacts:
            if artifact['id'] in app_pose_results_dict:
                app_pose_input = deepcopy(artifact['blurred_image'])
                pose_preds = prepare_draw_kpts(app_pose_results_dict[artifact['id']])
                pose_img = draw_mlkit_pose(np.asarray(pose_preds, dtype=np.float32), app_pose_input)
                _, bin_file = cv2.imencode('.JPEG', pose_img)
                bin_file = bin_file.tostring()
                artifact['app_pose_file_id_from_post_request'] = cgm_api.post_files(bin_file, 'rgb')
        result_bunch_object = prepare_app_pose_visualize_result_object(artifacts, scan_id, workflow_id, app_pose_results_dict)
        result_json_object = bunch_object_to_json_object(result_bunch_object)
        cgm_api.post_results(result_json_object)


def prepare_app_pose_visualize_result_object(artifacts, scan_id, workflow_id, app_pose_results_dict):
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        if artifact['id'] in app_pose_results_dict:
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=scan_id,
                workflow=workflow_id,
                source_artifacts=[artifact['id']],
                source_results=[],
                file=artifact['app_pose_file_id_from_post_request'],
                generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)

    return res


def draw_mlkit_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (MLKIT_NUM_KPTS, 2)
    for i in range(len(MLKIT_SKELETON)):
        kpt_a, kpt_b = MLKIT_SKELETON[i][0], MLKIT_SKELETON[i][1]
        kpt_a, kpt_b = kpt_a - 1, kpt_b - 1

        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        img = cv2.circle(img, (int(x_a), int(y_a)), 6, MlkitColors[i], -1)
        img = cv2.circle(img, (int(x_b), int(y_b)), 6, MlkitColors[i], -1)
        img = cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), MlkitColors[i], 2)
    return img


def prepare_draw_kpts(mlkit_pose_result):
    key_point_result_list = mlkit_pose_result['key_points_coordinate']

    intermediate_key_point_result = {}
    for key_point_result in key_point_result_list:
        key_point, coordinate = list(key_point_result.items())[0]
        x_coordinate, y_coordinate = coordinate['x'], coordinate['y']
        intermediate_key_point_result[key_point] = [x_coordinate, y_coordinate]

    draw_kpts = []
    for index, key_point in MLKIT_KEYPOINT_INDEXES.items():
        draw_kpts.append(
            intermediate_key_point_result[key_point]
        )

    return draw_kpts

