import sys
import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision

sys.path.append(str(Path(__file__).parents[1]))  # noqa: E402
import result_generation.pose_prediction.code.models.pose_hrnet  # noqa
from result_generation.pose_prediction.code.config import cfg, update_config
from result_generation.pose_prediction.code.config.constants import (COCO_KEYPOINT_INDEXES, NUM_KPTS)
from result_generation.pose_prediction.code.models.pose_hrnet import get_pose_net
from result_generation.pose_prediction.code.utils.utils import (box_to_center_scale, calculate_pose_score, draw_pose,
                                                                get_person_detection_boxes, get_pose_estimation_prediction, rot)
import log

logger = log.setup_custom_logger(__name__)

REPO_DIR = Path(os.getenv('APP_DIR', '/app'))


class PosePrediction:
    def __init__(self, ctx):
        self.ctx = ctx

    def load_box_model(self):
        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.box_model.to(self.ctx)
        self.box_model.eval()

    def load_pose_model(self):
        self.pose_model = get_pose_net(cfg)
        model_file_path = str(REPO_DIR / cfg.TEST.MODEL_FILE)
        loaded_model = torch.load(model_file_path, map_location=torch.device('cpu'))
        self.pose_model.load_state_dict(loaded_model, strict=False)
        # self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg.GPUS)
        self.pose_model.to(self.ctx)
        self.pose_model.eval()
        self.pose_model

    def read_image(self, image_path):
        image_bgr = cv2.imread(image_path)
        return image_bgr, image_bgr.shape

    def orient_image_using_scan_type(self, original_image, scan_type):
        if scan_type in [100, 101, 102]:
            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)  # Standing
        elif scan_type in [200, 201, 202]:
            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Laying
        else:
            logger.info("%s %s %s", "Provided scan type", scan_type, "not supported")
            logger.info("Keeping the image in the same orientation as provided")
            rotated_image = original_image
        return rotated_image

    def orient_cordinate_using_scan_type(self, pose_keypoints, scan_type, height, width):
        if scan_type in [100, 101, 102]:
            pose_keypoints = rot(pose_keypoints, 'ROTATE_90_COUNTERCLOCKWISE', height, width)
        elif scan_type in [200, 201, 202]:
            pose_keypoints = rot(pose_keypoints, 'ROTATE_90_CLOCKWISE', height, width)
        else:
            logger.info("%s %s %s", "Provided scan type", scan_type, "not supported")
            logger.info("Keeping the co-ordinate in the same orientation as provided")
        return pose_keypoints

    def preprocess_image(self, rotated_image):
        box_model_input = []
        rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rotated_image_rgb / 255.).permute(2, 0, 1).float().to(self.ctx)
        box_model_input.append(img_tensor)
        return box_model_input, rotated_image_rgb

    def perform_box_on_image(self, box_model_input):
        pred_boxes, pred_score = get_person_detection_boxes(
            self.box_model, box_model_input, threshold=cfg.BOX_MODEL.THRESHOLD)
        return pred_boxes, pred_score

    def perform_pose_on_image(self, pose_bbox, rotated_image_rgb):
        center, scale = box_to_center_scale(pose_bbox, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        pose_preds, pose_score = get_pose_estimation_prediction(self.pose_model, rotated_image_rgb, center, scale)
        return pose_preds, pose_score

    def pose_draw_on_image(self, rotated_pose_preds, original_image):
        if len(rotated_pose_preds) >= 1:
            for kpt in rotated_pose_preds:
                draw_pose(kpt, original_image)  # draw the poses

    def save_final_image(self, final_image_name, original_image):
        cv2.imwrite('outputs/' + final_image_name, original_image)


class ResultGeneration:
    def __init__(self, pose_prediction, save_pose_overlay):
        self.pose_prediction = pose_prediction
        self.save_pose_overlay = save_pose_overlay

    def result_on_artifact_level(self, jpg_path, scan_type):

        original_image, shape = self.pose_prediction.read_image(str(jpg_path))
        rotated_image = self.pose_prediction.orient_image_using_scan_type(original_image, scan_type)
        box_model_input, rotated_image_rgb = self.pose_prediction.preprocess_image(rotated_image)
        logger.info("%s %s ", "shape", shape)

        pred_boxes, pred_score = self.pose_prediction.perform_box_on_image(box_model_input)
        logger.info("%s ", pred_boxes)

        pose_result = []

        # Get Height,Width,color from Image
        (height, width, color) = shape
        body_pose_score = []
        # one box ==> one pose pose[0]

        for idx in range(len(pred_boxes)):
            single_body_pose_result = {}
            key_points_coordinate_list = []
            key_points_prob_list = []

            pose_bbox = pred_boxes[idx]
            pose_preds, pose_score = self.pose_prediction.perform_pose_on_image(pose_bbox, rotated_image_rgb)
            pose_preds[0] = self.pose_prediction.orient_cordinate_using_scan_type(
                pose_preds[0], scan_type, height, width)

            if self.save_pose_overlay:
                self.pose_prediction.pose_draw_on_image(pose_preds, original_image)
                if idx == len(pred_boxes) - 1:
                    self.pose_prediction.save_final_image(jpg_path.split('/')[-1], original_image)

            for i in range(0, NUM_KPTS):
                key_points_coordinate_list.append(
                    {COCO_KEYPOINT_INDEXES[i]: {'x': pose_preds[0][i][0], 'y': pose_preds[0][i][1]}})
                key_points_prob_list.append({COCO_KEYPOINT_INDEXES[i]: {'score': pose_score[0][i][0]}})
            body_pose_score.append(calculate_pose_score(pose_score))
            single_body_pose_result = {
                'bbox_coordinates': pose_bbox,
                'bbox_confidence_score': pred_score,
                'key_points_coordinate': key_points_coordinate_list,
                'key_points_prob': key_points_prob_list,
                'body_pose_score': calculate_pose_score(pose_score),
                'draw_kpt': pose_preds
            }
            pose_result.append(single_body_pose_result)

        return len(pred_boxes), body_pose_score, pose_result

    def result_on_scan_level(self, input_path, scan_type):

        # self.artifact_pose_result = []
        logger.info("Extracting artifacts from scans")

        logger.info("Result Generation Started")

        no_of_body_pose, body_pose_score, pose_result = self.result_on_artifact_level(input_path, scan_type)

        return no_of_body_pose, body_pose_score, pose_result


def init_pose_prediction():
    ctx = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("%s %s", "cuda is available", torch.cuda.is_available())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    args = str(REPO_DIR / 'src/result_generation/pose_prediction/inference-config-hrnet.yaml')
    update_config(cfg, args)

    pose_prediction = PosePrediction(ctx)
    pose_prediction.load_box_model()
    pose_prediction.load_pose_model()
    return pose_prediction


def inference_artifact(pose_prediction, input_path, scan_type):
    result = ResultGeneration(pose_prediction, False)
    no_of_body_pose, body_pose_score, pose_result = result.result_on_artifact_level(input_path, scan_type)
    return no_of_body_pose, body_pose_score, pose_result
