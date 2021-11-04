import sys
from datetime import datetime
from pathlib import Path
import glob2 as glob
import os

import numpy as np

from result_generation.height.height import HeightFlow

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402
import log

logger = log.setup_custom_logger(__name__)

REPO_DIR = Path(os.getenv('APP_DIR', '/app'))


class HeightFlowDeepEnsemble(HeightFlow):
    def run_flow(self):
        depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
        model_paths = glob.glob(f'{str(REPO_DIR)}/models/deepensemble/*')

        prediction_list = []
        for model_path in model_paths:
            prediction_list += [inference.get_ensemble_height_predictions_local(model_path, depthmaps)]

        prediction_list = np.array(prediction_list)
        std = np.std(prediction_list, axis=0)
        prediction_list = np.mean(prediction_list, axis=0)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results_deep_ensemble(prediction_list, generated_timestamp, std)

    def post_height_results_deep_ensemble(self, predictions, generated_timestamp, stds):
        """Post the artifact and scan level height results to API"""
        artifact_level_height_result_bunch = self.artifact_level_result_ensemble(
            predictions, generated_timestamp, stds)
        artifact_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_height_result_bunch)
        if self.result_generation.api.post_results(artifact_level_height_result_json) == 201:
            logger.info("successfully post artifact level height results: %s", artifact_level_height_result_json)

        scan_level_height_result_bunch = self.scan_level_result(
            predictions, generated_timestamp, self.scan_workflow_obj, stds)
        scan_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_height_result_bunch)
        if self.result_generation.api.post_results(scan_level_height_result_json) == 201:
            logger.info("successfully post scan level height results: %s", scan_level_height_result_json)
