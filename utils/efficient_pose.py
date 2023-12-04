import pickle
import uuid
from datetime import datetime
from copy import deepcopy
import io

from bunch import Bunch
import cv2
from PIL import Image, ImageDraw
import numpy as np

# from utils.preprocessing import pose_input
from utils.inference import get_efficient_pose_prediction
from utils.result_utils import bunch_object_to_json_object, get_workflow, check_if_results_exists
from utils.constants import EFFICIENT_POSE_WORKFLOW_NAME, EFFICIENT_POSE_WORKFLOW_VERSION, EFFICIENT_POSE_VISUALIZE_WORKFLOW_NAME, EFFICIENT_POSE_VISUALIZE_WORKFLOW_VERSION


MAX_BATCH_SIZE = 13


def run_efficient_pose_flow(cgm_api, scan_id, artifacts, workflows, scan_type, scan_version, results):
    workflow = get_workflow(workflows, EFFICIENT_POSE_WORKFLOW_NAME, EFFICIENT_POSE_WORKFLOW_VERSION)
    visualize_workflow = get_workflow(workflows, EFFICIENT_POSE_VISUALIZE_WORKFLOW_NAME, EFFICIENT_POSE_VISUALIZE_WORKFLOW_VERSION)
    if not (check_if_results_exists(results, workflow['id']) and check_if_results_exists(results, visualize_workflow['id'])):
        eff_pose_input = []
        for artifact in artifacts:
            im = Image.open(io.BytesIO(artifact['raw_file']))
            if scan_type in [100, 101, 102]:
                rotated_image = im.rotate(-90, expand=1)
            elif scan_type in [200, 201, 202]:
                rotated_image = im.rotate(90, expand=1)
            eff_pose_input.append(np.asarray(rotated_image))
        total_data = len(eff_pose_input)
        if total_data <= MAX_BATCH_SIZE:
            pickled_pose_data = pickle.dumps(eff_pose_input)
            pose_predictions = get_efficient_pose_prediction(pickled_pose_data)
        else:
            i = 0
            pose_predictions = []
            for i in range(0, total_data, MAX_BATCH_SIZE):
                pickled_pose_data = pickle.dumps(eff_pose_input[i:i + MAX_BATCH_SIZE])
                pose_predictions.extend(get_efficient_pose_prediction(pickled_pose_data))
        post_results(artifacts, pose_predictions, cgm_api, scan_id, workflow['id'], visualize_workflow['id'], scan_type)


def get_pose_results(rgb_artifacts, pose_predictions, scan_id, workflow_id):
    res = Bunch(dict(results=[]))
    for (artifact, pose_prediction) in zip(rgb_artifacts, pose_predictions):
        for k, v in pose_prediction[1].items():
            pose_prediction[1][k] = v.tolist()
        c_f = {}
        for p in pose_prediction[0][0]:
            c_f[p[0]] = (p[1], p[2])
        no_of_pose_result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            data={
                'coordinates_fraction': c_f,
                'coordinates': pose_prediction[1],
            },
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(no_of_pose_result)
    return res


def post_results(rgb_artifacts, pose_predictions, cgm_api, scan_id, workflow_id, visualize_workflow_id, scan_type):
    pose_results_bunch_object = get_pose_results(rgb_artifacts, pose_predictions, scan_id, workflow_id)
    pose_results_json_object = bunch_object_to_json_object(pose_results_bunch_object)
    cgm_api.post_results(pose_results_json_object)

    pose_and_blur_visualsation(cgm_api, rgb_artifacts, pose_predictions, scan_type)
    result_bunch_object = prepare_pose_visualize_object(rgb_artifacts, scan_id, visualize_workflow_id)
    result_json_object = bunch_object_to_json_object(result_bunch_object)
    cgm_api.post_results(result_json_object)


def pose_and_blur_visualsation(cgm_api, artifacts, predictions, scan_type):
    for (artifact, pose_prediction) in zip(artifacts, predictions):
        img = deepcopy(artifact['blurred_image'])
        image = Image.fromarray(img.astype('uint8'), 'RGB')
        if scan_type in [100, 101, 102]:
            rotated_image = image.rotate(-90, expand=1)
        elif scan_type in [200, 201, 202]:
            rotated_image = image.rotate(90, expand=1)
        pose_vis = annotate_image(rotated_image, pose_prediction[0])
        np_pose_vis = np.asarray(pose_vis)
        if scan_type in [100, 101, 102]:
            image = cv2.rotate(np_pose_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif scan_type in [200, 201, 202]:
            image = cv2.rotate(np_pose_vis, cv2.ROTATE_90_CLOCKWISE)
        _, bin_file = cv2.imencode('.JPEG', image)
        bin_file = bin_file.tostring()
        artifact['eff_pose_file_id_from_post_request'] = cgm_api.post_files(bin_file, 'rgb')


def prepare_pose_visualize_object(artifacts, scan_id, workflow_id):
    res = Bunch(dict(results=[]))
    for artifact in artifacts:
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            file=artifact['eff_pose_file_id_from_post_request'],
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            start_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)

    return res


def annotate_image(image, coordinates):
    """
    Annotates supplied image from predicted coordinates.
    
    Args:
        file_path: path
            System path of image to annotate
        coordinates: list
            Predicted body part coordinates for image
    """

    # Load raw image
    image_width, image_height = image.size
    image_side = image_width if image_width >= image_height else image_height

    # Annotate image
    image_draw = ImageDraw.Draw(image)
    image_coordinates = coordinates[0]
    image = display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
    image = display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side/100))

    # Save annotated image
    # image.save(normpath(file_path.split('.')[0] + '_tracked.png'))
    return image


def display_body_parts(image, image_draw, coordinates, image_height=1024, image_width=1024, marker_radius=5):   
    """
    Draw markers on predicted body part locations.
    
    Args:
        image: PIL Image
            The loaded image the coordinate predictions are inferred for
        image_draw: PIL ImageDraw module
            Module for performing drawing operations
        coordinates: List
            Predicted body part coordinates in image
        image_height: int
            Height of image
        image_width: int
            Width of image
        marker_radius: int
            Radius of marker
           
    Returns:
        Instance of PIL image with annotated body part predictions.
    """
    
    # Define body part colors
    body_part_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw markers
    for i, (body_part, body_part_x, body_part_y) in enumerate(coordinates):
        body_part_x *= image_width
        body_part_y *= image_height
        image_draw.ellipse([(body_part_x - marker_radius, body_part_y - marker_radius), (body_part_x + marker_radius, body_part_y + marker_radius)], fill=body_part_colors[i])
        
    return image


def display_segments(image, image_draw, coordinates, image_height=1024, image_width=1024, segment_width=5):
    """
    Draw segments between body parts according to predicted body part locations.
    
    Args:
        image: PIL Image
            The loaded image the coordinate predictions are inferred for
        image_draw: PIL ImageDraw module
            Module for performing drawing operations
        coordinates: List
            Predicted body part coordinates in image
        image_height: int
            Height of image
        image_width: int
            Width of image
        segment_width: int
            Width of association line between markers
           
    Returns:
        Instance of PIL image with annotated body part segments.
    """
   
    # Define segments and colors
    segments = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)]
    segment_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw segments
    for (body_part_a_index, body_part_b_index) in segments:
        _, body_part_a_x, body_part_a_y = coordinates[body_part_a_index]
        body_part_a_x *= image_width
        body_part_a_y *= image_height
        _, body_part_b_x, body_part_b_y = coordinates[body_part_b_index]
        body_part_b_x *= image_width
        body_part_b_y *= image_height
        image_draw.line([(body_part_a_x, body_part_a_y), (body_part_b_x, body_part_b_y)], fill=segment_colors[body_part_b_index], width=segment_width)
    
    return image
