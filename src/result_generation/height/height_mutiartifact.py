import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowMutliArtifact(HeightFlow):

    def process_depthmaps_depthmapmultiartifactlatefusion(self):
        depthmaps_file = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])
            depthmaps_file.append(input_path)
        scans = []
        scans.append(depthmaps_file)
        samples = list(
            map(partial(preprocessing.sample_systematic_from_artifacts, n_artifacts=5), scans))
        return samples

    def create_multiartifact_sample(self, depthmap):
        depthmaps = np.zeros((240, 180, 5))

        for i, depthmap_path in enumerate(depthmap[0]):
            data, width, height, depth_scale, max_confidence = preprocessing.load_depth(
                depthmap_path)
            depthmap, height, width = preprocessing.prepare_depthmap(
                data, width, height, depth_scale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps[:, :, i] = tf.squeeze(depthmap, axis=2)

        depthmaps = tf.stack([depthmaps])
        return depthmaps

    def run_height_flow_depthmapmultiartifactlatefusion(self):
        depthmap = self.process_depthmaps_depthmapmultiartifactlatefusion()
        depthmap = self.create_multiartifact_sample(depthmap)
        height_predictions = inference.get_depthmapmultiartifactlatefusion_height_predictions_local(
            depthmap)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        scan_depthmapmultiartifactlatefusion_level_height_result_bunch = self.scan_level_height_result_object(
            height_predictions, generated_timestamp, self.scan_workflow_obj)
        scan_depthmapmultiartifactlatefusion_level_height_result_json = self.bunch_object_to_json_object(
            scan_depthmapmultiartifactlatefusion_level_height_result_bunch)
        if self.api.post_results(scan_depthmapmultiartifactlatefusion_level_height_result_json) == 201:
            print(
                "successfully posted scan step level M-CNN height results: ",
                scan_depthmapmultiartifactlatefusion_level_height_result_json)
