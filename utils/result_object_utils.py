from bunch import Bunch
import json
import uuid
from datetime import datetime


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


def calculate_age(dob, scan_date):
    date_dob = datetime.strptime(dob, "%Y-%m-%d")
    date_scan = datetime.strptime(scan_date, '%Y-%m-%dT%H:%M:%SZ')
    delta = date_scan - date_dob
    return delta.days
