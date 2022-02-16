import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import log
from api_endpoints import ApiEndpoints
from bunch import Bunch
from cgmzscore import Calculator
from error_stats_api_endpoints import ErrorStatsEndpointsManager
from fastcore.basics import store_attr

sys.path.append(str(Path(__file__).parents[1]))
from result_generation.utils import (MAX_AGE, MAX_HEIGHT, MIN_HEIGHT,
                                     calculate_age)

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
            image_artifacts,
            scan_type,
            scan_version,
            scan_meta_data_details,
            standing_laying_workflow_path):
        store_attr('result_generation,artifact_workflow_path,scan_workflow_path,artifacts,person_details,scan_type,scan_version,scan_meta_data_details,standing_laying_workflow_path', self)
        self.image_artifacts = [] if image_artifacts is None else image_artifacts
        self.artifact_workflow_obj = self.result_generation.workflows.load_workflows(
            self.artifact_workflow_path)
        self.standing_laying_workflow_obj = self.result_generation.workflows.load_workflows(
            self.standing_laying_workflow_path)
        self.scan_workflow_obj = self.result_generation.workflows.load_workflows(
            self.scan_workflow_path)
        if self.artifact_workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
            self.rgb_input_format = 'img'
            self.scan_directory = Path(self.result_generation.scan_parent_dir) / \
                self.result_generation.scan_metadata['id'] / self.depth_input_format
            self.scan_rgb_directory = Path(self.result_generation.scan_parent_dir) / \
                self.result_generation.scan_metadata['id'] / self.rgb_input_format
        self.artifact_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.artifact_workflow_obj['name'], self.artifact_workflow_obj['version'])
        self.standing_laying_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.standing_laying_workflow_obj['name'], self.standing_laying_workflow_obj['version'])
        self.scan_workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.scan_workflow_obj['name'], self.scan_workflow_obj['version'])

    def get_standing_results(self):
        url = os.getenv('APP_URL', 'http://localhost:5001')
        cgm_api = ApiEndpoints(url)
        result = cgm_api.get_results(self.standing_laying_workflow_obj['id'],self.result_generation.scan_metadata['id'])
        artifact_id_dict_by_order_id = {}
        sl_data_dict_by_order_id={}
        for image_artifact in self.image_artifacts:
            artifact_id_dict_by_order_id[image_artifact['id']] = image_artifact['order']
        for r in result:
            rgb_image_id = r['source_artifacts'][0]
            sl_data_dict_by_order_id[artifact_id_dict_by_order_id[rgb_image_id]] = float(r['data']['standing_laying'][1:-1])
        for artifact in self.artifacts: 
            if artifact['order'] in sl_data_dict_by_order_id:
                artifact['standing_laying'] = sl_data_dict_by_order_id[artifact['order']]


    def calculate_percentile(self):
        url_error_stats = os.getenv('APP_URL_ERROR_STATS',
                                    'https://cgm-be-ci-dev-errsts-api.azurewebsites.net/')
        logger.info("%s %s", "App URL Error Stats:", url_error_stats)
        cgm_error_stats_api = ErrorStatsEndpointsManager(url_error_stats)
        for artifact in self.artifacts:
            if 'standing_laying' in artifact:
                artifact['percentile'] = cgm_error_stats_api.get_percentile_from_error_stats(
                    self.scan_meta_data_details['age'], self.scan_meta_data_details['scan_type'], self.scan_version, self.artifact_workflow_obj['name'], self.scan_workflow_obj['version'], 99)
            else:
                artifact['percentile'] = cgm_error_stats_api.get_percentile_from_error_stats(
                    self.scan_meta_data_details['age'], self.scan_meta_data_details['scan_type'], self.scan_version, self.artifact_workflow_obj['name'], self.scan_workflow_obj['version'], 99,artifact['standing_laying'])

    def calculate_scan_level_error_stats(self):
        scan_99_percentile_pos_error = None
        scan_99_percentile_neg_error = None
        mae_artifact_result = []
        for artifact in self.artifacts:
            if 'percentile' in artifact and bool(artifact['percentile']):
                if artifact['percentile']['mae'] is not None:
                    mae_artifact_result.append(artifact['percentile']['mae'])
                if artifact['percentile']['99_percentile_pos_error'] is not None:
                    if scan_99_percentile_pos_error is None:
                        scan_99_percentile_pos_error = artifact['percentile']['99_percentile_pos_error']
                    else:
                        scan_99_percentile_pos_error = max(scan_99_percentile_pos_error,
                                                           artifact['percentile']['99_percentile_pos_error'])
                if artifact['percentile']['99_percentile_neg_error'] is not None:
                    if scan_99_percentile_neg_error is None:
                        scan_99_percentile_neg_error = artifact['percentile']['99_percentile_neg_error']
                    else:
                        scan_99_percentile_neg_error = min(scan_99_percentile_neg_error,
                                                           artifact['percentile']['99_percentile_neg_error'])
        if len(mae_artifact_result) > 0:
            mae_artifact_result.sort()
            mid = len(mae_artifact_result) // 2
            mae_scan = (mae_artifact_result[mid] + mae_artifact_result[~mid]) / 2
        else:
            mae_scan = None
        return scan_99_percentile_pos_error, scan_99_percentile_neg_error, mae_scan

    def artifact_level_result(self, predictions, generated_timestamp, start_time):
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
                data={'height': str(prediction[0]), 'pos_pe': artifact['percentile']['99_percentile_neg_error'],
                      'neg_pe': artifact['percentile']['99_percentile_pos_error'], 'mae': artifact['percentile']['mae']}
                if 'percentile' in artifact and bool(artifact['percentile']) else
                {'height': str(prediction[0]), 'pos_pe': None, 'neg_pe': None, 'mae': None},
                start_time=start_time,
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)
        scan_99_percentile_pos_error, scan_99_percentile_neg_error, mae_scan = self.calculate_scan_level_error_stats()
        return res, scan_99_percentile_pos_error, scan_99_percentile_neg_error, mae_scan

    def scan_level_height_result_object(self, predictions, generated_timestamp, workflow_obj, start_time, pos_percentile_error_99, neg_percentile_error_99, mae_scan):
        logger.info("%s", "Scan Level Result started")
        """Prepare scan level height result object"""
        res = Bunch(dict(results=[]))
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=self.result_generation.scan_metadata['id'],
            workflow=workflow_obj["id"],
            source_artifacts=[artifact['id'] for artifact in self.artifacts],
            source_results=[],
            generated=generated_timestamp,
            start_time=start_time,
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        mean_prediction = self.result_generation.get_mean_scan_results(predictions)
        class_lhfa = self.zscore_lhfa(mean_prediction)
        result.data = {
            'mean_height': mean_prediction,
            'Height Diagnosis': class_lhfa,
            'pos_pe': pos_percentile_error_99,
            'neg_pe': neg_percentile_error_99,
            'mae': mae_scan}
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

    def post_height_results(self, predictions, generated_timestamp, start_time):
        """Post the artifact and scan level height results to the API"""
        artifact_level_height_result_bunch, pos_percentile_error_99, neg_percentile_error_99, mae_scan = self.artifact_level_result(
            predictions, generated_timestamp, start_time)
        artifact_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_height_result_bunch)
        if self.result_generation.api.post_results(artifact_level_height_result_json) == 201:
            logger.info("%s %s", "successfully post artifact level height results:", artifact_level_height_result_json)

        scan_level_height_result_bunch = self.scan_level_height_result_object(
            predictions, generated_timestamp, self.scan_workflow_obj, start_time, pos_percentile_error_99, neg_percentile_error_99, mae_scan)
        scan_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_height_result_bunch)
        if self.result_generation.api.post_results(scan_level_height_result_json) == 201:
            logger.info("%s %s", "successfully post scan level height results:", scan_level_height_result_json)

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
                data={'height': str(prediction[0]), 'uncertainty': str(std[0])},
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
