import pickle
import uuid
from datetime import datetime
import io

from bunch import Bunch
import cv2
import numpy as np
from zipfile import ZipFile
from typing import Tuple, List
import math

# from utils.preprocessing import pose_input
# from utils.inference import get_efficient_pose_prediction
from utils.result_utils import bunch_object_to_json_object, get_workflow, check_if_results_exists
from utils.constants import DEPTH_FEATURE_WORKFLOW_NAME, DEPTH_FEATURE_WORKFLOW_VERSION


IDENTITY_MATRIX_4D = [1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.]

MAX_BATCH_SIZE = 13


def run_depth_features_flow(cgm_api, scan_id, artifacts, workflows, results):
    workflow = get_workflow(workflows, DEPTH_FEATURE_WORKFLOW_NAME, DEPTH_FEATURE_WORKFLOW_VERSION)
    if not check_if_results_exists(results, workflow['id']):
        for artifact in artifacts:
            device_pose = get_device_pose(artifact['raw_file'])
            device_pose_arr = np.array(device_pose).reshape(4, 4).T
            artifact['angle_between_camera_and_floor'] = get_angle_between_camera_and_floor(device_pose_arr)
        post_results(artifacts, cgm_api, scan_id, workflow['id'])


def get_device_pose(content):
    zipfile = ZipFile(io.BytesIO(content))
    with zipfile.open('data') as f:
        # Example for a first_line: '180x135_0.001_7_0.57045287_-0.0057296_0.0022602521_0.82130724_-0.059177425_0.0024800065_0.030834956'
        first_line = f.readline().decode().strip()
        width, height, depth_scale, max_confidence, device_pose = parse_header(first_line)

        return device_pose


def get_results(depth_artifacts, scan_id, workflow_id):
    res = Bunch(dict(results=[]))
    for artifact in depth_artifacts:
        depth_feature_result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            data={
                'angle_between_camera_and_floor': artifact['angle_between_camera_and_floor'],
            },
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(depth_feature_result)
    return res


def post_results(depth_artifacts, cgm_api, scan_id, workflow_id):
    results_bunch_object = get_results(depth_artifacts, scan_id, workflow_id)
    results_json_object = bunch_object_to_json_object(results_bunch_object)
    cgm_api.post_results(results_json_object)


def matrix_calculate(position: List[float], rotation: List[float]) -> List[float]:
    """Calculate a matrix image->world from device position and rotation"""

    output = IDENTITY_MATRIX_4D

    sqw = rotation[3] * rotation[3]
    sqx = rotation[0] * rotation[0]
    sqy = rotation[1] * rotation[1]
    sqz = rotation[2] * rotation[2]

    invs = 1 / (sqx + sqy + sqz + sqw)
    output[0] = (sqx - sqy - sqz + sqw) * invs
    output[5] = (-sqx + sqy - sqz + sqw) * invs
    output[10] = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = rotation[0] * rotation[1]
    tmp2 = rotation[2] * rotation[3]
    output[1] = 2.0 * (tmp1 + tmp2) * invs
    output[4] = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = rotation[0] * rotation[2]
    tmp2 = rotation[1] * rotation[3]
    output[2] = 2.0 * (tmp1 - tmp2) * invs
    output[8] = 2.0 * (tmp1 + tmp2) * invs

    tmp1 = rotation[1] * rotation[2]
    tmp2 = rotation[0] * rotation[3]
    output[6] = 2.0 * (tmp1 + tmp2) * invs
    output[9] = 2.0 * (tmp1 - tmp2) * invs

    output[12] = -position[0]
    output[13] = -position[1]
    output[14] = -position[2]
    return output


def matrix_transform_point(point: np.ndarray, device_pose_arr: np.ndarray) -> np.ndarray:
    """Transformation of point by device pose matrix

    point(np.array of float): 3D point
    device_pose: flattened 4x4 matrix

    Returns:
        3D point(np.array of float)
    """
    point_4d = np.append(point, 1.)
    output = np.matmul(device_pose_arr, point_4d)
    output[0:2] = output[0:2] / abs(output[3])
    return output[0:-1]


def get_angle_between_camera_and_floor(device_pose_arr) -> float:
    """Calculate an angle between camera and floor based on device pose

    The angle is often a negative values because the phone is pointing down.

    Angle examples:
    angle=-90deg: The phone's camera is fully facing the floor
    angle=0deg: The horizon is in the center
    angle=90deg: The phone's camera is facing straight up to the sky.
    """
    forward = matrix_transform_point([0, 0, 1], device_pose_arr)
    camera = matrix_transform_point([0, 0, 0], device_pose_arr)
    return math.degrees(math.asin(camera[1] - forward[1]))


def parse_header(header_line: str) -> Tuple:
    header_parts = header_line.split('_')
    res = header_parts[0].split('x')
    width = int(res[0])
    height = int(res[1])
    depth_scale = float(header_parts[1])
    max_confidence = float(header_parts[2])
    if len(header_parts) >= 10:
        position = (float(header_parts[7]), float(header_parts[8]), float(header_parts[9]))
        rotation = (float(header_parts[3]), float(header_parts[4]),
                    float(header_parts[5]), float(header_parts[6]))
        if position == (0., 0., 0.):
            device_pose = None
        else:
            device_pose = matrix_calculate(position, rotation)
    else:
        device_pose = IDENTITY_MATRIX_4D
    return width, height, depth_scale, max_confidence, device_pose