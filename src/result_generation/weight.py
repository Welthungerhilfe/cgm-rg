import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from bunch import Bunch
from cgmzscore import Calculator

from result_generation.utils import MAX_AGE, calculate_age

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class WeightFlow:
    """
    A class to handle weight results generation.

    Attributes
    ----------
    artifact_workflow_path : str
        path of the workflow file for artifact level weight results
    scan_workflow_path : json
        path of the workflow file for scan level weight results
    artifacts : list
        list of artifacts to run weigth flow on

    Methods
    -------
    process_depthmaps():
        Loads the list of depthmaps in scan as numpy array.
    run_weight_flow():
        Driver method for weight flow.
    artifact_level_weight_result_object(predictions, generated_timestamp):
        Prepares artifact level weight result object.
    scan_level_weight_result_object(predictions, generated_timestamp):
        Prepares scan level weight result object.
    post_weight_results(predictions, generated_timestamp):
        Posts the artifact and scan level weight results to api.
    """

    def __init__(
            self,
            result_generation,
            artifact_workflow_path,
            scan_workflow_path,
            artifacts,
            person_details):
        self.result_generation = result_generation
        self.artifacts = artifacts
        self.artifact_workflow_path = artifact_workflow_path
        self.scan_workflow_path = scan_workflow_path
        self.artifact_workflow_obj = self.result_generation.workflows.load_workflows(
            self.artifact_workflow_path)
        self.scan_workflow_obj = self.result_generation.workflows.load_workflows(
            self.scan_workflow_path)
        self.person_details = person_details
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
        depthmaps = self.process_depthmaps()
        weight_predictions = inference.get_weight_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_weight_results(weight_predictions, generated_timestamp)

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            data, width, height, depthScale, _max_confidence = preprocessing.load_depth(input_path)
            depthmap = preprocessing.prepare_depthmap(data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)
        depthmaps = np.array(depthmaps)
        return depthmaps

    def artifact_level_weight_result_object(
            self, predictions, generated_timestamp):
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.artifacts, predictions):
            weight_result = Bunch()
            weight_result.id = f"{uuid.uuid4()}"
            weight_result.scan = self.result_generation.scan_metadata['id']
            weight_result.workflow = self.artifact_workflow_obj["id"]
            weight_result.source_artifacts = [artifact['id']]
            weight_result.source_results = []
            weight_result.generated = generated_timestamp
            result = {'weight': str(prediction[0])}
            weight_result.data = result
            res.results.append(weight_result)

        return res

    def scan_level_weight_result_object(
            self, predictions, generated_timestamp):
        res = Bunch()
        res.results = []
        weight_result = Bunch()
        weight_result.id = f"{uuid.uuid4()}"
        weight_result.scan = self.result_generation.scan_metadata['id']
        weight_result.workflow = self.scan_workflow_obj["id"]
        weight_result.source_artifacts = [artifact['id'] for artifact in self.artifacts]
        weight_result.source_results = []
        weight_result.generated = generated_timestamp
        mean_prediction = self.result_generation.get_mean_scan_results(predictions)
        class_wfa = self.zscore_wfa(mean_prediction)
        result = {'mean_weight': mean_prediction, 'Weight Diagnosis': class_wfa}
        weight_result.data = result

        res.results.append(weight_result)
        return res

    def zscore_wfa(self, mean_prediction):
        sex = 'M' if self.person_details['sex'] == 'male' else 'F'
        age_in_days = calculate_age(
            self.person_details['date_of_birth'], self.result_generation.scan_metadata['scan_start'])
        class_wfa = 'Not Found'
        if age_in_days <= MAX_AGE:
            zscore_wfa = Calculator().zScore_lhfa(
                age_in_days=str(age_in_days), sex=sex, height=mean_prediction)
            if zscore_wfa < -3:
                class_wfa = 'Severly Under-weight'
            elif zscore_wfa < -2:
                class_wfa = 'Moderately Under-weight'
            else:
                class_wfa = 'Not underweight'
        return class_wfa

    def post_weight_results(self, predictions, generated_timestamp):
        artifact_level_weight_result_bunch = self.artifact_level_weight_result_object(
            predictions, generated_timestamp)
        artifact_level_weight_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_weight_result_bunch)
        if self.result_generation.api.post_results(artifact_level_weight_result_json) == 201:
            print(
                "successfully post artifact level weight results: ",
                artifact_level_weight_result_json)

        scan_level_weight_result_bunch = self.scan_level_weight_result_object(
            predictions, generated_timestamp)
        scan_level_weight_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_weight_result_bunch)
        if self.result_generation.api.post_results(scan_level_weight_result_json) == 201:
            print(
                "successfully post scan level weight results: ",
                scan_level_weight_result_json)
