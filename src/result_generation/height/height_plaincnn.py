import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowPlainCnn(HeightFlow):
    def run_height_flow(self):
        depthmaps = self.process_depthmaps()
        height_predictions = inference.get_height_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results(height_predictions, generated_timestamp)

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])

            data, width, height, depth_scale, max_confidence = preprocessing.load_depth(
                input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(
                data, width, height, depth_scale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)

        return depthmaps
