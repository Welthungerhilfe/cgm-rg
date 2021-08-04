import sys
# import logging
from datetime import datetime
from pathlib import Path
import cv2
import tensorflow as tf
import os

import numpy as np

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402
import log


logger = log.setup_custom_logger(__name__)
ORDER_DIFFERENCE_ALLOWED = 3


class HeightFlowRGBD(HeightFlow):
    def run_flow(self):
        rgbd_scans = self.process_rgbd()
        height_predictions = inference.get_height_rgbd_prediction_local(rgbd_scans)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results(height_predictions, generated_timestamp)

    def process_rgbd(self):
        rgbd_scan = []
        scan_image_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            'img')

        image_order_ids = []
        for image_artifact in self.image_artifacts:
            image_order_ids.append(image_artifact['order'])

        for artifact in self.artifacts:
            input_path = self.result_generation.get_input_path(
                self.scan_directory, artifact['file'])
            depth_id = artifact['order']
            closest_order_id = preprocessing.find_corresponding_image(image_order_ids, depth_id)
            if abs(closest_order_id - depth_id) >= ORDER_DIFFERENCE_ALLOWED:
                logger.info("No corresponding image artifact found for the depthmap")
                continue
            result_image_dict = next(
                iter(item for item in self.image_artifacts if item['order'] == closest_order_id), None)
            if result_image_dict:
                image_input_path = self.result_generation.get_input_path(
                    scan_image_directory, result_image_dict['file'])
            else:
                logger.info("%s %s", "No RGB found for order:", depth_id)
                continue
            raw_image = cv2.imread(str(image_input_path))  # cv2.imread gives error when reading from posix path
            rot_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)
            image = preprocessing.preprocess_image(rot_image)
            data, width, height, depth_scale, _max_confidence = preprocessing.load_depth(
                input_path)
            depthmap = preprocessing.prepare_depthmap(
                data, width, height, depth_scale)
            depthmap = preprocessing.preprocess(depthmap)
            rgbd_data = tf.concat([image, depthmap], axis=2)
            rgbd_data = tf.image.resize(
                rgbd_data, (preprocessing.IMAGE_TARGET_HEIGHT,
                            preprocessing.IMAGE_TARGET_WIDTH))
            rgbd_scan.append(rgbd_data)
        rgbd_scan = np.array(rgbd_scan)
        return rgbd_scan
