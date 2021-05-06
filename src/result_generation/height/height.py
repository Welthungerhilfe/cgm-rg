import json
import os
import uuid

import numpy as np
from bunch import Bunch
from cgmzscore import Calculator

from result_generation.utils import MAX_AGE, MAX_HEIGHT, MIN_HEIGHT, age


class HeightFlow:
    """Handle height results generation.

    Attributes
    ----------
    api: object
        object of ApiEndpoints class
    workflows: list
        list of registered workflows
    artifact_workflow_path: str
        path of the workflow file for artifact level height results
    scan_workflow_path: json
        path of the workflow file for scan level height results
    artifacts: list
        list of artifacts to run heigth flow on
    scan_parent_dir: str
        directory where scans are stored
    scan_metadata: json
        metadata of the scan to run height flow on
    """

    def __init__(
            self,
            api,
            workflows,
            artifact_workflow_path,
            scan_workflow_path,
            artifacts,
            scan_parent_dir,
            scan_metadata,
            person_details):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.artifact_workflow_path = artifact_workflow_path
        self.scan_workflow_path = scan_workflow_path
        self.artifact_workflow_obj = self.workflows.load_workflows(
            self.artifact_workflow_path)
        self.scan_workflow_obj = self.workflows.load_workflows(
            self.scan_workflow_path)
        self.scan_metadata = scan_metadata
        self.person_details = person_details
        self.scan_parent_dir = scan_parent_dir
        if self.artifact_workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
        self.scan_directory = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'],
            self.depth_input_format)
        self.artifact_workflow_obj['id'] = self.workflows.get_workflow_id(
            self.artifact_workflow_obj['name'], self.artifact_workflow_obj['version'])
        self.scan_workflow_obj['id'] = self.workflows.get_workflow_id(
            self.scan_workflow_obj['name'], self.scan_workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        """Convert given bunch object to json object"""
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)

        return json_object

    def get_input_path(self, directory, file_name):
        """Returns input path for given directory name and file name"""
        return os.path.join(directory, file_name)

    def get_mean_scan_results(self, predictions):
        """Return the average prediction from given list of predictions"""
        return str(np.mean(predictions))

    def artifact_level_height_result_object(self, predictions, generated_timestamp):
        """Prepare artifact level height result object."""
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.artifacts, predictions):
            height_result = Bunch()
            height_result.id = f"{uuid.uuid4()}"
            height_result.scan = self.scan_metadata['id']
            height_result.workflow = self.artifact_workflow_obj["id"]
            height_result.source_artifacts = [artifact['id']]
            height_result.source_results = []
            height_result.generated = generated_timestamp
            result = {'height': str(prediction[0])}
            height_result.data = result
            res.results.append(height_result)

        return res

    def scan_level_height_result_object(self, predictions, generated_timestamp, workflow_obj):
        """Prepare scan level height result object"""
        res = Bunch()
        res.results = []
        height_result = Bunch()
        height_result.id = f"{uuid.uuid4()}"
        height_result.scan = self.scan_metadata['id']
        height_result.workflow = workflow_obj["id"]
        height_result.source_artifacts = [
            artifact['id'] for artifact in self.artifacts]
        height_result.source_results = []
        height_result.generated = generated_timestamp
        mean_prediction = self.get_mean_scan_results(predictions)
        class_lhfa = self.zscore_lhfa(mean_prediction)
        result = {'mean_height': mean_prediction,
                  'Height Diagnosis': class_lhfa}
        height_result.data = result

        res.results.append(height_result)

        return res

    def zscore_lhfa(self, mean_prediction):
        sex = 'M' if self.person_details['sex'] == 'male' else 'F'
        age_in_days = age(
            self.person_details['date_of_birth'], self.scan_metadata['scan_start'])
        class_lhfa = 'Not Found'
        if MIN_HEIGHT < float(mean_prediction) <= MAX_HEIGHT and 0 < age_in_days <= MAX_AGE:
            zscore_lhfa = Calculator().zScore_lhfa(
                age_in_days=str(age_in_days), sex=sex, height=mean_prediction)
            if zscore_lhfa < -3:
                class_lhfa = 'Severly Stunted'
            elif zscore_lhfa < -2:
                class_lhfa = 'Moderately Stunted'
            else:
                class_lhfa = 'Not Stunted'
        return class_lhfa

    def post_height_results(self, predictions, generated_timestamp):
        """Post the artifact and scan level height results to API"""
        artifact_level_height_result_bunch = self.artifact_level_height_result_object(
            predictions, generated_timestamp)
        artifact_level_height_result_json = self.bunch_object_to_json_object(
            artifact_level_height_result_bunch)
        if self.api.post_results(artifact_level_height_result_json) == 201:
            print(
                "successfully post artifact level height results: ",
                artifact_level_height_result_json)

        scan_level_height_result_bunch = self.scan_level_height_result_object(
            predictions, generated_timestamp, self.scan_workflow_obj)
        scan_level_height_result_json = self.bunch_object_to_json_object(
            scan_level_height_result_bunch)
        if self.api.post_results(scan_level_height_result_json) == 201:
            print(
                "successfully post scan level height results: ",
                scan_level_height_result_json)

    def artifact_level_height_result_object_ensemble(self, predictions, generated_timestamp, stds):
        """Prepare artifact level height result object."""
        res = Bunch()
        res.results = []
        for artifact, prediction, std in zip(self.artifacts, predictions, stds):
            height_result = Bunch()
            height_result.id = f"{uuid.uuid4()}"
            height_result.scan = self.scan_metadata['id']
            height_result.workflow = self.artifact_workflow_obj["id"]
            height_result.source_artifacts = [artifact['id']]
            height_result.source_results = []
            height_result.generated = generated_timestamp
            result = {'height': str(prediction[0]), 'uncertainty': str(std[0])}
            height_result.data = result
            res.results.append(height_result)

        return res

    def scan_level_height_result_object_ensemble(self, predictions, generated_timestamp, workflow_obj, stds):
        """Prepare scan level height result object"""
        res = Bunch()
        res.results = []
        height_result = Bunch()
        height_result.id = f"{uuid.uuid4()}"
        height_result.scan = self.scan_metadata['id']
        height_result.workflow = workflow_obj["id"]
        height_result.source_artifacts = [
            artifact['id'] for artifact in self.artifacts]
        height_result.source_results = []
        height_result.generated = generated_timestamp
        mean_prediction = self.get_mean_scan_results(predictions)
        mean_std = self.get_mean_scan_results(stds)
        class_lhfa = self.zscore_lhfa(mean_prediction)
        result = {'mean_height': mean_prediction,
                  'Height Diagnosis': class_lhfa,
                  'uncertainty': mean_std}
        height_result.data = result

        res.results.append(height_result)

        return res

    def post_height_results_deep_ensemble(self, predictions, generated_timestamp, stds):
        """Post the artifact and scan level height results to API"""
        artifact_level_height_result_bunch = self.artifact_level_height_result_object_ensemble(
            predictions, generated_timestamp, stds)
        artifact_level_height_result_json = self.bunch_object_to_json_object(
            artifact_level_height_result_bunch)
        if self.api.post_results(artifact_level_height_result_json) == 201:
            print(
                "successfully post artifact level height results: ",
                artifact_level_height_result_json)

        scan_level_height_result_bunch = self.scan_level_height_result_object_ensemble(
            predictions, generated_timestamp, self.scan_workflow_obj, stds)
        scan_level_height_result_json = self.bunch_object_to_json_object(
            scan_level_height_result_bunch)
        if self.api.post_results(scan_level_height_result_json) == 201:
            print(
                "successfully post scan level height results: ",
                scan_level_height_result_json)
