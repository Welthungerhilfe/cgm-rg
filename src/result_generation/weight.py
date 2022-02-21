import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from bunch import Bunch
from cgmzscore import Calculator
from fastcore.basics import store_attr

from error_stats_api_endpoints import ErrorStatsEndpointsManager
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
            person_details,
            scan_type,
            scan_version,
            scan_meta_data_details):
        store_attr(
            'result_generation,artifact_workflow_path,scan_workflow_path,artifacts,person_details,scan_type,scan_version,scan_meta_data_details',
            self)
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

    def calculate_percentile(self):
        url_error_stats = os.getenv('APP_URL_ERROR_STATS',
                                    'https://cgm-be-ci-dev-errsts-api.azurewebsites.net/')
        logger.info("%s %s", "App URL Error Stats:", url_error_stats)
        cgm_error_stats_api = ErrorStatsEndpointsManager(url_error_stats)
        for artifact in self.artifacts:
            artifact['percentile'] = cgm_error_stats_api.get_percentile_from_error_stats(
                self.scan_meta_data_details['age'],
                self.scan_meta_data_details['scan_type'],
                self.scan_version,
                self.artifact_workflow_obj['name'],
                self.scan_workflow_obj['version'],
                99,
                None)

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

    def run_flow(self):
        start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        depthmaps = preprocessing.process_depthmaps(self.artifacts, self.scan_directory, self.result_generation)
        weight_predictions = inference.get_weight_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.calculate_percentile()
        self.post_weight_results(weight_predictions, generated_timestamp, start_time)

    def artifact_level_result(self, predictions, generated_timestamp, start_time):
        """Prepare artifact level weight result object"""
        res = Bunch(dict(results=[]))
        for artifact, prediction in zip(self.artifacts, predictions):

            result = Bunch(
                dict(
                    id=f"{uuid.uuid4()}",
                    scan=self.result_generation.scan_metadata['id'],
                    workflow=self.artifact_workflow_obj["id"],
                    source_artifacts=[
                        artifact['id']],
                    source_results=[],
                    generated=generated_timestamp,
                    data={
                        'weight': str(prediction[0]),
                        'pos_pe': artifact['percentile']['99_percentile_neg_error'],
                        'neg_pe': artifact['percentile']['99_percentile_pos_error'],
                        'mae': artifact['percentile']['mae']} if 'percentile' in artifact and bool(
                        artifact['percentile']) else {
                            'weight': str(
                                prediction[0]), 'pos_pe': None, 'neg_pe': None, 'mae': None},
                    start_time=start_time,
                    end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                ))
            res.results.append(result)
        scan_99_percentile_pos_error, scan_99_percentile_neg_error, mae_scan = self.calculate_scan_level_error_stats()
        return res, scan_99_percentile_pos_error, scan_99_percentile_neg_error, mae_scan

    def scan_level_result(
            self,
            predictions,
            generated_timestamp,
            start_time,
            pos_percentile_error_99,
            neg_percentile_error_99,
            mae_scan):
        """Prepare scan level weight result object"""
        res = Bunch(dict(results=[]))
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=self.result_generation.scan_metadata['id'],
            workflow=self.scan_workflow_obj["id"],
            source_artifacts=[artifact['id'] for artifact in self.artifacts],
            source_results=[],
            generated=generated_timestamp,
            start_time=start_time,
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        mean_prediction = self.result_generation.get_mean_scan_results(predictions)
        median_prediction = self.result_generation.get_median_scan_results(predictions)
        class_wfa = self.zscore_wfa(mean_prediction)

        result.data = {
            'mean_weight': mean_prediction,
            'median_weight': median_prediction,
            'Weight Diagnosis': class_wfa,
            'pos_pe': pos_percentile_error_99,
            'neg_pe': neg_percentile_error_99,
            'mae': mae_scan}

        res.results.append(result)
        return res

    def zscore_wfa(self, mean_prediction):
        sex = 'M' if self.person_details['sex'] == 'male' else 'F'
        age_in_days = calculate_age(
            self.person_details['date_of_birth'], self.result_generation.scan_metadata['scan_start'])
        class_wfa = 'Not Found'
        if age_in_days <= MAX_AGE:
            zscore_wfa = Calculator().zScore_wfa(
                age_in_days=str(age_in_days), sex=sex, weight=mean_prediction)
            if zscore_wfa < -3:
                class_wfa = 'Severly Under-weight'
            elif zscore_wfa < -2:
                class_wfa = 'Moderately Under-weight'
            else:
                class_wfa = 'Not underweight'
        return class_wfa

    def post_weight_results(self, predictions, generated_timestamp, start_time):
        """Post the artifact and scan level weight results to the API"""
        artifact_level_weight_result_bunch, pos_percentile_error_99, neg_percentile_error_99, mae_scan = self.artifact_level_result(
            predictions, generated_timestamp, start_time)
        artifact_level_weight_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_weight_result_bunch)
        if self.result_generation.api.post_results(artifact_level_weight_result_json) == 201:
            logger.info("%s %s", "successfully post artifact level weight results:", artifact_level_weight_result_json)

        scan_level_weight_result_bunch = self.scan_level_result(
            predictions, generated_timestamp, start_time, pos_percentile_error_99, neg_percentile_error_99, mae_scan)
        scan_level_weight_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_weight_result_bunch)
        if self.result_generation.api.post_results(scan_level_weight_result_json) == 201:
            logger.info("%s %s", "successfully post scan level weight results:", scan_level_weight_result_json)
