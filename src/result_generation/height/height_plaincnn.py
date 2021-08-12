import sys
from datetime import datetime
from pathlib import Path

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlowPlainCnn(HeightFlow):
    def __init__(self, result_generation, artifact_workflow_path, scan_workflow_path, artifacts, person_details, image_artifacts, result_level = None):
        super().__init__(result_generation, artifact_workflow_path, scan_workflow_path, artifacts, person_details, image_artifacts=image_artifacts)
        self.result_level = result_level

    def run_flow(self):
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        if self.result_level == "artifact":
            depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
            height_predictions = inference.get_height_predictions_local(depthmaps)
            self.post_artifact_level_height_results(height_predictions, generated_timestamp)
        elif self.result_level == "scan":
            height_predictions = self.get_height_prediction_for_scan()
            self.post_scan_level_height_results(height_predictions, generated_timestamp)
        else:
            depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
            height_predictions = inference.get_height_predictions_local(depthmaps)
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            self.post_height_results(height_predictions, generated_timestamp)
