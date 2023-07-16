import pickle
import uuid
from datetime import datetime
from copy import deepcopy

from bunch import Bunch
import cv2

from utils.preprocessing import pose_input
from utils.inference import get_pose_prediction
from utils.result_utils import bunch_object_to_json_object, get_workflow
from utils.constants import POSE_WORKFLOW_NAME, POSE_WORKFLOW_VERSION, POSE_VISUALIZE_WORKFLOW_NAME, POSE_VISUALIZE_WORKFLOW_VERSION, SKELETON, CocoColors, NUM_KPTS


MAX_BATCH_SIZE = 13


def run_pose_flow(cgm_api, scan_id, artifacts, workflows, scan_type, scan_version):
    workflow = get_workflow(workflows, POSE_WORKFLOW_NAME, POSE_WORKFLOW_VERSION)
    visualize_workflow = get_workflow(workflows, POSE_VISUALIZE_WORKFLOW_NAME, POSE_VISUALIZE_WORKFLOW_VERSION)
    pose_data, shape = pose_input(artifacts, scan_type)
    total_data = len(pose_data)
    if total_data <= MAX_BATCH_SIZE:
        pickled_pose_data = pickle.dumps([scan_type, pose_data, shape])
        pose_predictions = get_pose_prediction(pickled_pose_data)
    else:
        i = 0
        pose_predictions = []
        for i in range(0, total_data, MAX_BATCH_SIZE):
            pickled_pose_data = pickle.dumps([scan_type, pose_data[i:i + MAX_BATCH_SIZE], shape])
            pose_predictions.extend(get_pose_prediction(pickled_pose_data))
            i += MAX_BATCH_SIZE
    post_results(artifacts, pose_predictions, cgm_api, scan_id, workflow['id'], visualize_workflow['id'])


def get_pose_results(rgb_artifacts, pose_predictions, scan_id, workflow_id):
    res = Bunch(dict(results=[]))
    for (artifact, pose_prediction) in zip(rgb_artifacts, pose_predictions):
        no_of_pose_detected, pose_score, pose_result = pose_prediction
        no_of_pose_result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            data={'no of person using pose': str(no_of_pose_detected)},
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(no_of_pose_result)
        for i in range(0, no_of_pose_detected):
            pose_score_results = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=scan_id,
                workflow=workflow_id,
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                data={'Pose Scores': str(pose_score[i]),
                      'Pose Results': str(pose_result[i])},
                start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(pose_score_results)
    return res


def post_results(rgb_artifacts, pose_predictions, cgm_api, scan_id, workflow_id, visualize_workflow_id):
    pose_results_bunch_object = get_pose_results(rgb_artifacts, pose_predictions, scan_id, workflow_id)
    pose_results_json_object = bunch_object_to_json_object(pose_results_bunch_object)
    cgm_api.post_results(pose_results_json_object)

    pose_and_blur_visualsation(cgm_api, rgb_artifacts, pose_predictions)
    result_bunch_object = prepare_pose_visualize_object(rgb_artifacts, scan_id, visualize_workflow_id)
    result_json_object = bunch_object_to_json_object(result_bunch_object)
    cgm_api.post_results(result_json_object)


def pose_and_blur_visualsation(cgm_api, artifacts, predictions):
    for (artifact, pose_prediction) in zip(artifacts, predictions):
        img = deepcopy(artifact['blurred_image'])
        no_of_pose_detected, pose_score, pose_result = pose_prediction
        if no_of_pose_detected > 0:
            rotated_pose_preds = pose_result[0]['draw_kpt']
            for kpt in rotated_pose_preds:
                img = draw_pose(kpt, img)
        _, bin_file = cv2.imencode('.JPEG', img)
        bin_file = bin_file.tostring()
        artifact['pose_file_id_from_post_request'] = cgm_api.post_files(bin_file, 'rgb')


def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert (len(keypoints), len(keypoints[0])) == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        img = cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        img = cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        img = cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
    return img


def prepare_pose_visualize_object(artifacts, scan_id, workflow_id):
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            file=artifact['pose_file_id_from_post_request'],
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res
