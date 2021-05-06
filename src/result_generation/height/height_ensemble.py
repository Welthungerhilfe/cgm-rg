import sys
from datetime import datetime
from pathlib import Path
import glob2 as glob

import numpy as np

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowDeepEnsemble(HeightFlow):
    def run_height_flow_deepensemble(self):
        depthmaps = self.process_depthmaps()
        prediction_list_one = []
        model_paths = glob.glob('/app/models/deepensemble/*')
        for model_index, model_path in enumerate(model_paths):
            prediction_list_one += [
                inference.get_ensemble_height_predictions_local(model_path, depthmaps)]
        prediction_list_one = np.array(prediction_list_one)
        std = np.std(prediction_list_one, axis=0)
        prediction_list_one = np.mean(prediction_list_one, axis=0)
        # print(prediction_list_one)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results_deep_ensemble(
            prediction_list_one, generated_timestamp, std)

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])

            data, width, height, depth_scale, _max_confidence = preprocessing.load_depth(
                input_path)
            depthmap = preprocessing.prepare_depthmap(
                data, width, height, depth_scale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)

        return depthmaps
