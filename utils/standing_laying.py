import uuid
from datetime import datetime
import pickle

import cv2
from bunch import Bunch

from utils.preprocessing import standing_laying_data_preprocessing_tf_batch
from utils.result_utils import bunch_object_to_json_object, get_workflow
from utils.constants import STANDING_LAYING_WORKFLOW_NAME, STANDING_LAYING_WORKFLOW_VERSION
from utils.inference import get_json_prediction


def sl_flow(cgm_api, scan_id, artifacts, workflows):
    sl_workflow = get_workflow(workflows, STANDING_LAYING_WORKFLOW_NAME, STANDING_LAYING_WORKFLOW_VERSION)
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    sl_input = standing_laying_data_preprocessing_tf_batch(artifacts)
    sl_pickle = pickle.dumps(sl_input)
    sl_predictions = get_json_prediction(sl_pickle, sl_workflow['data']['service_name'])
    post_result_object(cgm_api, scan_id, artifacts, sl_predictions, generated_timestamp, sl_workflow['id'])


def post_result_object(cgm_api, scan_id, artifacts, predictions, generated_timestamp, workflow_id):
    res = prepare_result_object(artifacts, predictions, generated_timestamp, scan_id, workflow_id)
    res_object = bunch_object_to_json_object(res)
    cgm_api.post_results(res_object)


def prepare_result_object(artifacts, predictions, generated_timestamp, scan_id, workflow_id):
    """Prepare result object for results generated"""
    res = Bunch(dict(results=[]))
    for artifact, prediction in zip(artifacts, predictions):
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=generated_timestamp,
            data={'standing': str(prediction[0])},
            start_time=generated_timestamp,
            end_time=generated_timestamp
        ))
        res.results.append(result)
    return res
