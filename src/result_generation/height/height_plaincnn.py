import sys
from datetime import datetime
from pathlib import Path

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowPlainCnn(HeightFlow):
    def run_flow(self):
        depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
        #height_predictions = inference.get_height_predictions_local(depthmaps)
        heatmaps, height_predictions = inference.get_height_prediction_and_heatmap_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        #self.post_height_results(height_predictions, generated_timestamp)
        self.post_height_and_gradcam_results(height_predictions, heatmaps, generated_timestamp)
