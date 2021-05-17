import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from bunch import Bunch
from fastcore.basics import store_attr

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa: E402
import utils.preprocessing as preprocessing  # noqa: E402


class StandingLaying:
    """A class to handle standing/laying results generation"""
    def __init__(
            self,
            result_generation,
            workflow_path,
            artifacts):
        store_attr('result_generation, workflow_path, artifacts', self)
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
        """Give prediction of standing/laying to the list of artifacts"""
        predictions = []
        for i, artifact in enumerate(self.artifacts):
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            print("input_path of image to perform standing laying: ", input_path)
            img = preprocessing.standing_laying_data_preprocessing(input_path)
            prediction = inference.get_standing_laying_prediction_local(img)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

    def prepare_result_object(self, prediction, generated_timestamp):
        """Prepare result object for results generated"""
        res = Bunch(dict(results=[]))
        for artifact, prediction in zip(self.artifacts, prediction):
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'standing': str(prediction[0])},
            ))
            res.results.append(result)
        return res

    def post_result_object(self, prediction, generated_timestamp):
        """Post the result object to the API"""
        res = self.prepare_result_object(prediction, generated_timestamp)
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            print("successfully post Standing laying results: ", res_object)
