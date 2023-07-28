from bunch import Bunch
import json
import uuid
from datetime import datetime
import numpy as np


def get_mean_scan_results(predictions):
    return str(np.mean(predictions))


def bunch_object_to_json_object(bunch_object):
    """Convert given bunch object to json object"""
    json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
    json_object = json.loads(json_string)
    return json_object


def prepare_result_object(rgb_artifacts, predictions, generated_timestamp, scan_id, workflow_id, result_key):
    """Prepare result object for results generated"""
    res = Bunch(dict(results=[]))
    for artifact, prediction in zip(rgb_artifacts, predictions):
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=scan_id,
            workflow=workflow_id,
            source_artifacts=[artifact['id']],
            source_results=[],
            generated=generated_timestamp,
            data={result_key: str(prediction[0])},
            start_time=artifact['start_time'],
            end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        ))
        res.results.append(result)
    return res


def get_workflow(workflows, workflow_name, workflow_version):
    workflow = [w for w in workflows if w['name'] == workflow_name and w['version'] == workflow_version]
    
    return workflow[0]


def check_if_results_exists(results, workflow_id):
    if [r for r in results if r['workflow'] == workflow_id]:
        return True
    return False
