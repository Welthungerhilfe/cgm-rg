import sys
from datetime import datetime
from pathlib import Path

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowPlainCnn(HeightFlow):
    def run_flow(self):
        start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
        height_predictions = inference.get_height_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.get_standing_results()
        self.calculate_percentile()
        self.post_height_results(height_predictions, generated_timestamp, start_time)
