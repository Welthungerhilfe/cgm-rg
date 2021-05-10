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
    def run_flow(self):
        depthmaps = self.process_depthmaps()
        model_paths = glob.glob('/app/models/deepensemble/*')

        prediction_list = []
        for model_path in model_paths:
            prediction_list += [inference.get_ensemble_height_predictions_local(model_path, depthmaps)]

        prediction_list = np.array(prediction_list)
        std = np.std(prediction_list, axis=0)
        prediction_list = np.mean(prediction_list, axis=0)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results_deep_ensemble(prediction_list, generated_timestamp, std)

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            data, width, height, depth_scale, _max_confidence = preprocessing.load_depth(input_path)
            depthmap = preprocessing.prepare_depthmap(data, width, height, depth_scale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)
        depthmaps = np.array(depthmaps)
        return depthmaps
