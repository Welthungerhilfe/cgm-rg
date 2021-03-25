import json
import os
import sys
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from bunch import Bunch
from cgmzscore import Calculator

from result_generation.utils import MAX_AGE, MAX_HEIGHT, MIN_HEIGHT, age

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class HeightFlow:
    """
    A class to handle height results generation.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    workflows : list
        list of registered workflows
    artifact_workflow_path : str
        path of the workflow file for artifact level height results
    scan_workflow_path : json
        path of the workflow file for scan level height results
    artifacts : list
        list of artifacts to run heigth flow on
    scan_parent_dir : str
        directory where scans are stored
    scan_metadata : json
        metadata of the scan to run height flow on

    Methods
    -------
    bunch_object_to_json_object(bunch_object):
        Converts given bunch object to json object.
    get_input_path(directory, file_name):
        Returns input path for given directory name and file name.
    get_mean_scan_results(predictions):
        Returns the average prediction from given list of predictions.
    process_depthmaps():
        Loads the list of depthmaps in scan as numpy array.
    run_height_flow():
        Driver method for height flow.
    artifact_level_height_result_object(predictions, generated_timestamp):
        Prepares artifact level height result object.
    scan_level_height_result_object(predictions, generated_timestamp):
        Prepares scan level height result object.
    post_height_results(predictions, generated_timestamp):
        Posts the artifact and scan level height results to api.
    """

    def __init__(
            self,
            api,
            workflows,
            artifact_workflow_path,
            scan_workflow_path,
            scan_depthmapmultiartifactlatefusion_workflow_path,
            artifacts,
            scan_parent_dir,
            scan_metadata,
            person_details):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.artifact_workflow_path = artifact_workflow_path
        self.scan_workflow_path = scan_workflow_path
        self.scan_depthmapmultiartifactlatefusion_workflow_path = scan_depthmapmultiartifactlatefusion_workflow_path
        self.artifact_workflow_obj = self.workflows.load_workflows(
            self.artifact_workflow_path)
        self.scan_workflow_obj = self.workflows.load_workflows(
            self.scan_workflow_path)
        self.scan_depthmapmultiartifactlatefusion_workflow_obj = self.workflows.load_workflows(
            self.scan_depthmapmultiartifactlatefusion_workflow_path)
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
        self.scan_depthmapmultiartifactlatefusion_workflow_obj['id'] = self.workflows.get_workflow_id(
            self.scan_depthmapmultiartifactlatefusion_workflow_obj['name'], self.scan_depthmapmultiartifactlatefusion_workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)

        return json_object

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def get_mean_scan_results(self, predictions):
        return str(np.mean(predictions))

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])

            data, width, height, depthScale, max_confidence = preprocessing.load_depth(
                input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(
                data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)

        return depthmaps

    def process_depthmaps_depthmapmultiartifactlatefusion(self):
        depthmaps_file = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])
            depthmaps_file.append(input_path)
        scans = []
        scans.append(depthmaps_file)
        samples = list(
            map(partial(preprocessing.sample_systematic_from_artifacts, n_artifacts=5), scans))
        return samples

    def create_multiartifact_sample(self, depthmap):
        depthmaps = np.zeros((240, 180, 5))

        for i, depthmap_path in enumerate(depthmap[0]):
            data, width, height, depthScale, max_confidence = preprocessing.load_depth(
                depthmap_path)
            depthmap, height, width = preprocessing.prepare_depthmap(
                data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps[:, :, i] = tf.squeeze(depthmap, axis=2)

        depthmaps = tf.stack([depthmaps])
        return depthmaps

    def run_height_flow(self):
        depthmaps = self.process_depthmaps()
        height_predictions = inference.get_height_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results(height_predictions, generated_timestamp)

    def run_height_flow_depthmapmultiartifactlatefusion(self):
        depthmap = self.process_depthmaps_depthmapmultiartifactlatefusion()
        depthmap = self.create_multiartifact_sample(depthmap)
        height_predictions = inference.get_depthmapmultiartifactlatefusion_height_predictions_local(
            depthmap)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        scan_depthmapmultiartifactlatefusion_level_height_result_bunch = self.scan_level_height_result_object(
            height_predictions, generated_timestamp, self.scan_depthmapmultiartifactlatefusion_workflow_obj)
        scan_depthmapmultiartifactlatefusion_level_height_result_json = self.bunch_object_to_json_object(
            scan_depthmapmultiartifactlatefusion_level_height_result_bunch)
        if self.api.post_results(scan_depthmapmultiartifactlatefusion_level_height_result_json) == 201:
            print(
                "successfully posted scan step level M-CNN height results: ",
                scan_depthmapmultiartifactlatefusion_level_height_result_json)

    def artifact_level_height_result_object(
            self, predictions, generated_timestamp):
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

    def scan_level_height_result_object(
            self, predictions, generated_timestamp, workflow_obj):
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
        if MIN_HEIGHT < float(mean_prediction) <= MAX_HEIGHT and age_in_days <= MAX_AGE:
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
