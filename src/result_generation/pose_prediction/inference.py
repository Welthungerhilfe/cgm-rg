import os
import time

import cv2
import glob2 as glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # noqa: E402
import result_generation.pose_prediction.code.models.pose_hrnet
from result_generation.pose_prediction.code.config import cfg, update_config
from result_generation.pose_prediction.code.config.constants import (COCO_KEYPOINT_INDEXES, NUM_KPTS)
from result_generation.pose_prediction.code.models.pose_hrnet import get_pose_net
from result_generation.pose_prediction.code.utils.utils import (box_to_center_scale, calculate_pose_score, draw_pose,
                                                                get_person_detection_boxes, get_pose_estimation_prediction, rot)
import log


logger = log.setup_custom_logger(__name__)


class PosePrediction:
    def __init__(self, ctx):
        self.ctx = ctx

    def load_box_model(self):
        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.box_model.to(self.ctx)
        self.box_model.eval()

    def load_pose_model(self):
        self.pose_model = get_pose_net(cfg)
        self.pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)
        #self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg.GPUS)
        self.pose_model.to(self.ctx)
        self.pose_model.eval()
        self.pose_model

    def read_image(self, image_path):
        self.image_bgr = cv2.imread(image_path)
        return self.image_bgr.shape

    def preprocess_image(self):
        self.input = []
        self.img = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(
            self.img / 255.).permute(2, 0, 1).float().to(self.ctx)
        self.input.append(img_tensor)

    def orient_image_using_scan_type(self, scan_type):
        if scan_type in [100, 101, 102]:
            self.rotated_image = cv2.rotate(self.image_bgr, cv2.ROTATE_90_CLOCKWISE)  # Standing
        elif scan_type in [200, 201, 202]:
            self.rotated_image = cv2.rotate(self.image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Laying
        else:
            logger.info("%s %s %s", "Provided scan type", scan_type, "not supported")
            logger.info("Keeping the image in the same orientation as provided")
            self.rotated_image = self.image_bgr

    def perform_box_on_image(self):
        self.pred_boxes, self.pred_score = get_person_detection_boxes(
            self.box_model, self.input, threshold=cfg.BOX_MODEL.THRESHOLD)
        return self.pred_boxes, self.pred_score

    def perform_pose_on_image(self, idx):
        center, scale = box_to_center_scale(
            self.pred_boxes[idx], cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        self.pose_preds, self.pose_score = get_pose_estimation_prediction(
            self.pose_model, self.img, center, scale)
        return self.pose_preds, self.pose_score

    def orient_cordinate_using_scan_type(self, pose_keypoints, scan_type, height):
        if scan_type in ['100', '101', '102']:
            pose_keypoints = rot(pose_keypoints, 'ROTATE_90_COUNTERCLOCKWISE', height)
        elif scan_type in ['200', '201', '202']:
            pose_keypoints = rot(pose_keypoints, 'ROTATE_90_CLOCKWISE', height)
        else:
            logger.info("%s %s %s", "Provided scan type", scan_type, "not supported")
            logger.info("Keeping the co-ordinate in the same orientation as provided")
        return pose_keypoints

    def pose_draw_on_image(self):
        if len(self.pose_preds) >= 1:
            for kpt in self.pose_preds:
                draw_pose(kpt, self.rotated_image)  # draw the poses

    def save_final_image(self, final_image_name):
        cv2.imwrite('outputs/' + final_image_name, self.rotated_image)


class ResultGeneration:
    def __init__(self, pose_prediction, save_pose_overlay):
        self.pose_prediction = pose_prediction
        self.save_pose_overlay = save_pose_overlay

    def result_on_artifact_level(self, jpg_path, scan_type):

        shape = self.pose_prediction.read_image(jpg_path)
        self.pose_prediction.orient_image_using_scan_type(scan_type)
        self.pose_prediction.preprocess_image()

        pred_boxes, pred_score = self.pose_prediction.perform_box_on_image()

        pose_result = []
        (height, width, color) = shape
        body_pose_score = []

        for idx in range(len(pred_boxes)):
            single_body_pose_result = {}
            key_points_coordinate_list = []
            key_points_prob_list = []

            pose_bbox = pred_boxes[idx]
            pose_preds, pose_score = self.pose_prediction.perform_pose_on_image(idx)

            pose_preds[0] = self.pose_prediction.orient_cordinate_using_scan_type(pose_preds[0], scan_type, height)

            if self.save_pose_overlay:
                self.pose_prediction.pose_draw_on_image()
                # TODO Ensure save image path
                self.pose_prediction.save_final_image(jpg_path.split('/')[-1])
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
                'body_pose_score': calculate_pose_score(pose_score)
            }
            pose_result.append(single_body_pose_result)

        return len(pred_boxes), body_pose_score, pose_result

    def result_on_scan_level(self, artifact_paths, scan_type, result_gen, scan_directory):
        no_of_body_pose_detected = []
        pose_score = []
        pose_results = []

        # self.artifact_pose_result = []
        logger.info("Extracting artifacts from scans")

        logger.info("Result Generation Started")
        for jpg_path in artifact_paths:
            input_path = str(result_gen.get_input_path(scan_directory, jpg_path['file']))

            no_of_body_pose, body_pose_score, pose_result = self.result_on_artifact_level(input_path, scan_type)
            no_of_body_pose_detected.append(no_of_body_pose)
            pose_score.append(body_pose_score)
            pose_results.append(pose_result)
            # self.artifact_pose_result.append(pose_result_of_artifact)
        return no_of_body_pose_detected, pose_score, pose_results


def inference_artifact(artifacts, scan_type, result_gen, scan_directory):
    ctx = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("%s %s", "cuda is available", torch.cuda.is_available())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    args = '/app/src/result_generation/pose_prediction/inference-config-hrnet.yaml'
    update_config(cfg, args)

    pose_prediction = PosePrediction(ctx)
    pose_prediction.load_box_model()
    pose_prediction.load_pose_model()

    result_generation = ResultGeneration(pose_prediction, False)
    return result_generation.result_on_scan_level(artifacts,
                                                  scan_type, result_gen, scan_directory)
