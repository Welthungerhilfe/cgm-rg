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


class StandingLaying:
    """
    A class to handle standing/laying results generation.

    Attributes
    ----------
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
            result_generation,
            workflow_path,
            artifacts,):
        self.result_generation = result_generation
        self.artifacts = artifacts
        self.workflow_path = workflow_path
        self.workflow_obj = self.result_generation.workflows.load_workflows(self.workflow_path)
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.standing_laying_input_format = 'img'
        self.scan_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            self.standing_laying_input_format)
        self.workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_obj['name'], self.workflow_obj['version'])

    def run_flow(self):
        prediction = self.standing_laying_artifacts()
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_result_object(prediction, generated_timestamp)

    def standing_laying_artifacts(self):
        predictions = []
        for i, artifact in enumerate(self.artifacts):

            input_path = self.result_generation.get_input_path(
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
            standing_laying_result.scan = self.result_generation.scan_metadata['id']
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
        standing_laying_result_object = self.result_generation.bunch_object_to_json_object(
            standing_laying_result)
        if self.result_generation.api.post_results(standing_laying_result_object) == 201:
            print("successfully post Standing laying results: ",
                  standing_laying_result_object)
