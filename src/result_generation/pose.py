import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from bunch import Bunch
from fastcore.basics import store_attr

sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa
import utils.preprocessing as preprocessing  # noqa
from pose_prediction.inference import inference_artifact
import log


logger = log.setup_custom_logger(__name__)


class PosePrediction:
    """A class to handle Pose Prediction results generation"""

    def __init__(
            self,
            result_generation,
            workflow_path,
            artifacts,
            scan_version,
            scan_type):
        store_attr('result_generation,artifacts,workflow_path,artifacts,scan_version,scan_type', self)
        self.workflow_obj = self.result_generation.workflows.load_workflows(self.workflow_path)
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.pose_input_format = 'img'
        self.scan_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            self.pose_input_format)

        self.workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_obj['name'], self.workflow_obj['version'])

    def run_flow(self):
        no_of_person, pose_score, pose_result = self.pose_prediction_artifacts()
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_result_object(no_of_person, pose_score, pose_result, generated_timestamp)

    def pose_prediction_artifacts(self):
        """Give Pose prediction to the list of artifacts"""
        no_of_pose_detected, pose_score, pose_result = inference_artifact(
            self.artifacts, self.scan_type, self.result_generation, self.scan_directory)
        no_of_person = np.array(no_of_pose_detected)
        pose_score = np.array(pose_score)
        pose_result = np.array(pose_result)
        return no_of_person, pose_score, pose_result

    def prepare_result_object(self, no_of_pose_detected, pose_score, pose_result, generated_timestamp):
        """Prepare result object for results generated"""
        res = Bunch(dict(results=[]))
        for artifact, number, score, result in zip(self.artifacts, no_of_pose_detected, pose_score, pose_result):
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'No of person': str(number), 'Pose Scores': str(score), 'Pose Result': str(result)},
            ))
            res.results.append(result)
        return res

    def post_result_object(self, no_of_person, pose_score, pose_result, generated_timestamp):
        """Post the result object to the API"""
        res = self.prepare_result_object(no_of_person, pose_score, pose_result, generated_timestamp)
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            logger.info("%s %s", "successfully post Post prediction results:", res_object)
