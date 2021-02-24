import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from bunch import Bunch

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class Standing_laying:
    """
    A class to handle standing/laying results generation.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    workflows : list
        list of registered workflows
    workflow_path : str
        path of the workflow file for standing_laying
    artifacts : list
        list of artifacts to run standing_laying flow on
    scan_parent_dir : str
        directory where scans are stored
    scan_metadata : json
        metadata of the scan to run standing_laying flow on

    Methods
    -------
    bunch_object_to_json_object(bunch_object):
        Converts given bunch object to json object.
    get_input_path(directory, file_name):
        Returns input path for given directory name and file name.
    run_standing_laying_flow():
        Driver method for Standing laying flow.
    standing_laying_artifacts():
        Give prediction of standing/laying to the list of artifacts.
    prepare_result_object(predictions, generated_timestamp):
        Prepares result object for results generated.
    post_result_object(predictions, generated_timestamp):
        Posts the result object to api.
    """

    def __init__(
            self,
            api,
            workflows,
            workflow_path,
            artifacts,
            scan_parent_dir,
            scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.workflow_path = workflow_path
        self.workflow_obj = self.workflows.load_workflows(self.workflow_path)
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.standing_laying_input_format = 'img'
        self.scan_directory = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'],
            self.standing_laying_input_format)
        self.workflow_obj['id'] = self.workflows.get_workflow_id(
            self.workflow_obj['name'], self.workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)
        return json_object

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def run_standing_laying_flow(self):
        prediction = self.standing_laying_artifacts()
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_result_object(prediction, generated_timestamp)

    def standing_laying_artifacts(self):
        predictions = []
        for i, artifact in enumerate(self.artifacts):

            input_path = self.get_input_path(
                self.scan_directory, artifact['file'])

            print("input_path of image to perform standing laying: ", input_path)

            img = preprocessing.standing_laying_data_preprocessing(input_path)
            prediction = inference.get_standing_laying_prediction_local(img)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

    def prepare_result_object(self, prediction, generated_timestamp):
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.artifacts, prediction):
            standing_laying_result = Bunch()
            standing_laying_result.id = f"{uuid.uuid4()}"
            standing_laying_result.scan = self.scan_metadata['id']
            standing_laying_result.workflow = self.workflow_obj["id"]
            standing_laying_result.source_artifacts = [artifact['id']]
            standing_laying_result.source_results = []
            standing_laying_result.generated = generated_timestamp
            result = {'standing': str(prediction[0])}
            standing_laying_result.data = result
            res.results.append(standing_laying_result)

        return res

    def post_result_object(self, prediction, generated_timestamp):
        standing_laying_result = self.prepare_result_object(
            prediction, generated_timestamp)
        standing_laying_result_object = self.bunch_object_to_json_object(
            standing_laying_result)
        if self.api.post_results(standing_laying_result_object) == 201:
            print("successfully post Standing laying results: ",
                  standing_laying_result_object)
