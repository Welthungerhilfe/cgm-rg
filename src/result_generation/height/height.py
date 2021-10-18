from datetime import datetime
import uuid
from pathlib import Path
import sys

from bunch import Bunch
from cgmzscore import Calculator
from fastcore.basics import store_attr

sys.path.append(str(Path(__file__).parents[1]))
from result_generation.utils import MAX_AGE, MAX_HEIGHT, MIN_HEIGHT, calculate_age
import log


logger = log.setup_custom_logger(__name__)


class HeightFlow:
    """Handle height results generation"""

    def __init__(
            self,
            result_generation,
            artifact_workflow_path,
            scan_workflow_path,
            artifacts,
            person_details,
            image_artifacts=None):
        store_attr('result_generation,artifact_workflow_path,scan_workflow_path,artifacts,person_details', self)
        self.image_artifacts = [] if image_artifacts is None else image_artifacts
        self.artifact_workflow_obj = self.result_generation.workflows.load_workflows(
            self.artifact_workflow_path)
        self.scan_workflow_obj = self.result_generation.workflows.load_workflows(
            self.scan_workflow_path)
        if self.artifact_workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
            self.scan_directory = Path(self.result_generation.scan_parent_dir) / \
                self.result_generation.scan_metadata['id'] / self.depth_input_format
        self.artifact_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.artifact_workflow_obj['name'], self.artifact_workflow_obj['version'])
        self.scan_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.scan_workflow_obj['name'], self.scan_workflow_obj['version'])

    def artifact_level_result(self, predictions, generated_timestamp):
        """Prepare artifact level height result object"""
        res = Bunch(dict(results=[]))

        for artifact, prediction in zip(self.artifacts, predictions):
            result = Bunch(dict(
                id=str(uuid.uuid4()),
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.artifact_workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'height': str(prediction[0])},
            ))
            res.results.append(result)
        return res

    def scan_level_height_result_object(self, predictions, generated_timestamp, workflow_obj):
        """Prepare scan level height result object"""
        res = Bunch(dict(results=[]))
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=self.result_generation.scan_metadata['id'],
            workflow=workflow_obj["id"],
            source_artifacts=[artifact['id'] for artifact in self.artifacts],
            source_results=[],
            generated=generated_timestamp,
        ))
        mean_prediction = self.result_generation.get_mean_scan_results(predictions)
        class_lhfa = self.zscore_lhfa(mean_prediction)
        result.data = {
            'mean_height': mean_prediction,
            'Height Diagnosis': class_lhfa}
        res.results.append(result)
        return res

    def zscore_lhfa(self, mean_prediction):
        sex = 'M' if self.person_details['sex'] == 'male' else 'F'
        age_in_days = calculate_age(self.person_details['date_of_birth'],
                                    self.result_generation.scan_metadata['scan_start'])
        if MIN_HEIGHT < float(mean_prediction) <= MAX_HEIGHT and 0 < age_in_days <= MAX_AGE:
            zscore_lhfa = Calculator().zScore_lhfa(
                age_in_days=str(age_in_days), sex=sex, height=mean_prediction)
            if zscore_lhfa < -3:
                class_lhfa = 'Severly Stunted'
            elif zscore_lhfa < -2:
                class_lhfa = 'Moderately Stunted'
            else:
                class_lhfa = 'Not Stunted'
        else:
            class_lhfa = 'Not Found'
        return class_lhfa

    def post_height_results(self, predictions, generated_timestamp):
        """Post the artifact and scan level height results to the API"""
        artifact_level_height_result_bunch = self.artifact_level_result(predictions, generated_timestamp)
        artifact_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_height_result_bunch)
        if self.result_generation.api.post_results(artifact_level_height_result_json) == 201:
            logger.info("%s %s", "successfully post artifact level height results:", artifact_level_height_result_json)

        scan_level_height_result_bunch = self.scan_level_height_result_object(
            predictions, generated_timestamp, self.scan_workflow_obj)
        scan_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_height_result_bunch)
        if self.result_generation.api.post_results(scan_level_height_result_json) == 201:
            logger.info("%s %s", "successfully post scan level height results:", scan_level_height_result_json)

    def post_heatmap_image_files(self, heatmaps):
        heatmap_ids = []
        for heatmap in heatmaps:
            heatmap_img_id_from_post_request, post_status = self.result_generation.api.post_files(heatmap)
            if post_status == 201:
                heatmap_ids.append(heatmap_img_id_from_post_request)
            return heatmap_ids

    def prepare_result_object_artifact_level_gradcam(self, generated_timestamp, heatmap_ids):
        """Prepare artifact level gradcam result object"""
        res = Bunch(dict(results=[]))
        for artifact, heatmap_id in zip(self.artifacts, heatmap_ids):
            # DO I NEED ALL OF THIS INFORMATION?
            result = Bunch(dict(
                id=str(uuid.uuid4()),
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.artifact_workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                file=heatmap_id
            ))
            res.results.append(result)

        return res

    def post_heatmap_result_object(self, generated_timestamps, heatmap_ids):
        res = self.prepare_result_object_artifact_level_gradcam(self, generated_timestamps, heatmap_ids)
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            logger.info("%s %s", "Successfully post Heatmap Image results:", res_object)

    def post_gradcam_results(self, heatmaps, generated_timestamps):
        # 1. post heatmap image files itself - returns ID for each heatmap in list
        heatmap_ids = self.post_heatmap_image_files(self, heatmaps)
        # 2. prepare json with IDs, again post json to API
        self.post_heatmap_result_object(self, generated_timestamps, heatmap_ids)

    def post_height_and_gradcam_results(self, predictions, heatmaps, generated_timestamp):
        # send height
        self.post_height_results(self, predictions, generated_timestamp)
        # send gradcam seperately
        self.post_gradcam_results(self, heatmaps, generated_timestamp)

    def artifact_level_result_ensemble(self, predictions, generated_timestamp, stds):
        """Prepare artifact level height result object"""
        res = Bunch(dict(results=[]))
        for artifact, prediction, std in zip(self.artifacts, predictions, stds):
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.artifact_workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'height': str(prediction[0]), 'uncertainty': str(std[0])}
            ))
            res.results.append(result)

        return res

    def scan_level_result(self, predictions, generated_timestamp, workflow_obj, stds):
        """Prepare scan level height result object"""
        res = Bunch(dict(results=[]))
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=self.result_generation.scan_metadata['id'],
            workflow=workflow_obj["id"],
            source_artifacts=[artifact['id'] for artifact in self.artifacts],
            source_results=[],
            generated=generated_timestamp,
        ))
        mean_prediction = self.result_generation.get_mean_scan_results(predictions)
        mean_std = self.result_generation.get_mean_scan_results(stds)
        class_lhfa = self.zscore_lhfa(mean_prediction)
        result = {'mean_height': mean_prediction,
                  'Height Diagnosis': class_lhfa,
                  'uncertainty': mean_std}
        result.data = result
        res.results.append(result)
        return res
