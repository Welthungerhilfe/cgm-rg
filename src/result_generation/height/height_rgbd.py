import sys
from datetime import datetime
from pathlib import Path
import cv2
import tensorflow as tf

import numpy as np

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowRGBD(HeightFlow):
    def run_rgbd_height_flow(self):
        rgbd_scans = self.process_rgbd()
        height_predictions = inference.get_height_rgbd_prediction_local(
            rgbd_scans)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results(height_predictions, generated_timestamp)

    def process_rgbd(self):
        rgbd_scan = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])
            img_id = artifact['order']
            result_image_dict = next(
                iter(item for item in self.image_artifacts if item['order'] == img_id), None)  # noqa :E501
            if result_image_dict:  # noqa :E501
                image_input_path = self.get_input_path(
                    self.scan_image_directory, result_image_dict['file'])
            else:
                print("No RGB found for order:", img_id)
                continue
            image = cv2.imread(image_input_path)
            image = preprocessing.preprocess_image(image)
            data, width, height, depth_scale, _max_confidence = preprocessing.load_depth(    # noqa :E501
                input_path)  # noqa :E501
            depthmap = preprocessing.prepare_depthmap(
                data, width, height, depth_scale)
            depthmap = preprocessing.preprocess(depthmap)
            rgbd_data = tf.concat([image, depthmap], axis=2)
            rgbd_data = tf.image.resize(
                rgbd_data, (preprocessing.IMAGE_TARGET_HEIGHT,
                            preprocessing.IMAGE_TARGET_WIDTH))
            rgbd_scan.append(rgbd_data)  # noqa :E501
        rgbd_scan = np.array(rgbd_scan)
        return rgbd_scan
