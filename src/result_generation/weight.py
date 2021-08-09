import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from bunch import Bunch
from cgmzscore import Calculator
from fastcore.basics import store_attr

from result_generation.utils import MAX_AGE, calculate_age

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402
import log


logger = log.setup_custom_logger(__name__)


class WeightFlow:
    """A class to handle weight results generation"""

    def __init__(
            self,
            result_generation,
            artifact_workflow_path,
            scan_workflow_path,
            artifacts,
            person_details):
        store_attr('result_generation,artifact_workflow_path,scan_workflow_path,artifacts,person_details', self)
        self.artifact_workflow_obj = self.result_generation.workflows.load_workflows(self.artifact_workflow_path)
        self.scan_workflow_obj = self.result_generation.workflows.load_workflows(self.scan_workflow_path)
        if self.artifact_workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
        self.scan_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            self.depth_input_format)
        self.artifact_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.artifact_workflow_obj['name'], self.artifact_workflow_obj['version'])
        self.scan_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.scan_workflow_obj['name'], self.scan_workflow_obj['version'])

    def run_flow(self):
        depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
        weight_predictions = inference.get_weight_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_weight_results(weight_predictions, generated_timestamp)

    def artifact_level_result(self, predictions, generated_timestamp):
        """Prepare artifact level weight result object"""
        res = Bunch(dict(results=[]))
        for artifact, prediction in zip(self.artifacts, predictions):
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.artifact_workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'weight': str(prediction[0])},
            ))
            res.results.append(result)
        return res

    def scan_level_result(self, predictions, generated_timestamp):
        """Prepare scan level weight result object"""
        res = Bunch(dict(results=[]))
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=self.result_generation.scan_metadata['id'],
            workflow=self.scan_workflow_obj["id"],
            source_artifacts=[artifact['id'] for artifact in self.artifacts],
            source_results=[],
            generated=generated_timestamp,
        ))
        mean_prediction = self.result_generation.get_mean_scan_results(predictions)
        class_wfa = self.zscore_wfa(mean_prediction)
        result.data = {'mean_weight': mean_prediction, 'Weight Diagnosis': class_wfa}
        res.results.append(result)
        return res

    def zscore_wfa(self, mean_prediction):
        sex = 'M' if self.person_details['sex'] == 'male' else 'F'
        age_in_days = calculate_age(
            self.person_details['date_of_birth'], self.result_generation.scan_metadata['scan_start'])
        class_wfa = 'Not Found'
        if age_in_days <= MAX_AGE:
            zscore_wfa = Calculator().zScore_lhfa(age_in_days=str(age_in_days), sex=sex, height=mean_prediction)
            if zscore_wfa < -3:
                class_wfa = 'Severly Under-weight'
            elif zscore_wfa < -2:
                class_wfa = 'Moderately Under-weight'
            else:
                class_wfa = 'Not underweight'
        return class_wfa

    def post_weight_results(self, predictions, generated_timestamp):
        """Post the artifact and scan level weight results to the API"""
        artifact_level_weight_result_bunch = self.artifact_level_result(predictions, generated_timestamp)
        artifact_level_weight_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_weight_result_bunch)
        if self.result_generation.api.post_results(artifact_level_weight_result_json) == 201:
            logger.info("%s %s", "successfully post artifact level weight results:", artifact_level_weight_result_json)

        scan_level_weight_result_bunch = self.scan_level_result(predictions, generated_timestamp)
        scan_level_weight_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_weight_result_bunch)
        if self.result_generation.api.post_results(scan_level_weight_result_json) == 201:
            logger.info("%s %s", "successfully post scan level weight results:", scan_level_weight_result_json)
