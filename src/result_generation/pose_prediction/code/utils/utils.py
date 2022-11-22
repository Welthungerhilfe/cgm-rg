import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # noqa: E402
import result_generation.pose_prediction.code.models.pose_hrnet  # noqa
from result_generation.pose_prediction.code.config import cfg
from result_generation.pose_prediction.code.config.constants import (
    COCO_INSTANCE_CATEGORY_NAMES,
    NUM_KPTS,
    SKELETON,
    CocoColors,
    MLKIT_SKELETON,
    MLKIT_NUM_KPTS,
    MLKIT_KEYPOINT_INDEXES,
    MlkitColors
)


from result_generation.pose_prediction.code.utils.post_processing import get_final_preds
from result_generation.pose_prediction.code.utils.transforms import get_affine_transform


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return [], 0

    filtered_index = [pred_score.index(x) for x in pred_score if x > threshold]
    pred_boxes = [pred_boxes[idx] for idx in filtered_index]
    pred_score = [pred_score[idx] for idx in filtered_index]
    pred_classes = [pred_classes[idx] for idx in filtered_index]

    person_boxes, person_scores = [], []
    for box, score, class_ in zip(pred_boxes, pred_score, pred_classes):
        if class_ == 'person':
            person_boxes.append(box)
            person_scores.append(score)

    return person_boxes, person_scores


def rot(keypoints, orientation, height, width):
    """
    Rotate a point counterclockwise,or clockwise.
    """
    rotated_keypoints = list()
    for i in range(0, NUM_KPTS):
        if orientation == 'ROTATE_90_CLOCKWISE':
            rot_x, rot_y = width - keypoints[i][1], keypoints[i][0]
        elif orientation == 'ROTATE_90_COUNTERCLOCKWISE':
            rot_x, rot_y = keypoints[i][1], height - keypoints[i][0]
        rotated_keypoints.append([rot_x, rot_y])
    return rotated_keypoints


def draw_mlkit_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (MLKIT_NUM_KPTS, 2)
    for i in range(len(MLKIT_SKELETON)):
        kpt_a, kpt_b = MLKIT_SKELETON[i][0], MLKIT_SKELETON[i][1]
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
        key_point, coordinate = key_point_result.items()[0]
        x_coordinate, y_coordinate = coordinate['x'], coordinate['y']
        intermediate_key_point_result[key_point] = [x_coordinate, y_coordinate]

    draw_kpts = []
    for index, key_point in MLKIT_KEYPOINT_INDEXES.items():
        draw_kpts.append(
            intermediate_key_point_result[key_point]
        )

    return draw_kpts


def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        img = cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        img = cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        img = cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
    return img


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, score = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))
        return preds, score


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def calculate_pose_score(pose_score):
    return np.mean(pose_score)
