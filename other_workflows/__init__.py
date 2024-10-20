import logging
import json
from datetime import datetime

import azure.functions as func

from utils.rest_api import CgmApi
from utils.pose import run_pose_flow
from utils.blur import run_blur_flow
from utils.standing_laying import sl_flow
from utils.depth_img import depth_img_flow
from utils.app_pose_visualization import run_app_pose_visualization_flow
from utils.efficient_pose import run_efficient_pose_flow
from utils.depth_features import run_depth_features_flow
from utils.mobilenet_height import run_mobilenet_height_flow


rgb_format = ["rgb", "image/jpeg"]
depth_format = ["depth", "application/zip"]

cgm_api = CgmApi()


def get_scan_by_format(artifacts, file_format):
    return [artifact for artifact in artifacts if artifact['format'] in file_format]


def download_artifacts(artifacts):
    for artifact in artifacts:
        artifact['raw_file'] = cgm_api.get_files(artifact['file'])

def main(msg: func.QueueMessage) -> None:
    logging.info('Python queue trigger function processed a queue item: %s',
                 msg.get_body().decode('utf-8'))
    message_received = json.loads(msg.get_body().decode('utf-8'))
    scan_id = message_received['scan_id']
    scan_metadata = cgm_api.get_scan_metadata(scan_id)
    workflows = cgm_api.get_workflows()
    person_id = scan_metadata['person']
    scan_date = datetime.strptime(scan_metadata['scan_start'], '%Y-%m-%dT%H:%M:%SZ').date()
    manual_measure = cgm_api.get_manual_measures(person_id, scan_date)
    artifacts = scan_metadata['artifacts']
    version = scan_metadata['version']
    scan_type = scan_metadata['type']
    results = scan_metadata['results']
    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    download_artifacts(artifacts)

    depth_artifacts = get_scan_by_format(artifacts, depth_format)
    rgb_artifacts = get_scan_by_format(artifacts, rgb_format)

    try:
        run_blur_flow(cgm_api, scan_id, rgb_artifacts, workflows, scan_type, version, results)
        run_pose_flow(cgm_api, scan_id, rgb_artifacts, workflows, scan_type, version, results)
        # run_efficient_pose_flow(cgm_api, scan_id, rgb_artifacts, workflows, scan_type, version, results)
        run_app_pose_visualization_flow(cgm_api, scan_id, rgb_artifacts, workflows, scan_type, version, results)
    except Exception as e:
        logging.info("rgb workflows failed")
        logging.error(e)

    try:
        depth_img_flow(cgm_api, scan_id, depth_artifacts, version, workflows, results)
    except Exception as e:
        logging.info("depth image workflow failed")
        logging.error(e)

    try:
        run_depth_features_flow(cgm_api, scan_id, depth_artifacts, workflows, results)
    except Exception as e:
        logging.info("depth feature workflow failed")
        logging.error(e)
    try:
        run_mobilenet_height_flow(cgm_api, scan_id, depth_artifacts, workflows, results, manual_measure)
    except Exception as e:
        logging.info("mobilenet height workflow failed")
        logging.error(e)
