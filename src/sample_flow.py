import cv2
import numpy as np
import os
import sys
import json
from pprint import pprint
from pathlib import Path
# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

from api_endpoints import ApiEndpoints
from process_workflows import ProcessWorkflows
from get_scan_metadata import GetScanMetadata
from prepare_artifacts import PrepareArtifacts

sys.path.append(str(Path(__file__).parents[1]))
from result_generation.pose_prediction.code.utils.utils import (
    draw_mlkit_pose,
    prepare_draw_kpts
)



scan_parent_dir = 'data/scans/'
scan_id = '01d77020-6f01-11ed-ad18-ff1a7240801c'
workflow_id = '78c17760-6599-11ed-be62-07e04204a375'
url = os.getenv('APP_URL', 'http://localhost:5001')
# logger.info("%s %s", "App URL:", url)
print("APP URL ", url)
rgb_format = 'img'

scan_dir= Path(scan_parent_dir) / scan_id

cgm_api = ApiEndpoints(url)
workflow = ProcessWorkflows(cgm_api)
#get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)

scan_metadata = cgm_api.get_scan_metadata(scan_id)

print("scan_metadata")
pprint(scan_metadata)


data_processing = PrepareArtifacts(
    cgm_api, 
    scan_metadata, 
    scan_parent_dir
    )

data_processing.process_scan_metadata()
data_processing.create_scan_dir()
data_processing.create_artifact_dir()
rgb_artifacts = data_processing.download_artifacts('img')
depth_artifacts = data_processing.download_artifacts('depth')


result_metadata =  cgm_api.get_results(
    scan_id = scan_id, 
    workflow_id = workflow_id)



print("result_metadata")
pprint(result_metadata)


pose_result_by_artifact_id = {}

# TODO  checks if source artifact is not present
# check if pose coordinates is not present

for result in result_metadata:
    if 'poseCoordinates' in result['data']:
        if len(result['source_artifacts']) > 0:
            pose_result_by_artifact_id[result['source_artifacts'][0]] = json.loads(
                result['data']['poseCoordinates']
            )
        else:
            print("More than one source artifact for result ")
            print(result['source_artifacts'])


for artifact in rgb_artifacts:
    print("=================================================")
    print("artifact id ")
    print(artifact['id'])
    print('artifact')
    print(artifact)
    if artifact['id'] in pose_result_by_artifact_id:
        artifact['app_pose_result'] = pose_result_by_artifact_id[artifact['id']]
        print('Pose result ')
        print(artifact['app_pose_result'])
        artifact['mlkit_draw_kpt'] = prepare_draw_kpts(artifact['app_pose_result'])



for artifact in rgb_artifacts:
    print("=================================================")
    input_path = scan_dir / rgb_format / artifact['file']
    output_path = scan_dir / rgb_format / (artifact['file'] + '_pose.jpeg')
    print("artifact id ")
    print(artifact['id'])

    print("input_path of image to perform blur:", input_path)
    print("output_path of image to perform blur:", output_path)
    assert os.path.exists(input_path), f"{input_path} does not exist"
    rgb_image = cv2.imread(str(input_path))
    # img = artifact['blurred_image']
    # Here we are considering that mlkit is generating one pos
    # So this expect one pose. For multi pose visualisation
    # we will need to modify the code accordingly
    pose_preds = artifact['mlkit_draw_kpt']
    print("pose_preds")
    print(pose_preds)
    pose_img = draw_mlkit_pose(np.asarray(pose_preds, dtype=np.float32), rgb_image)
    artifact['mlkit_pose_blurred_image'] = pose_img

    cv2.imwrite(str(output_path), pose_img)










# scan_id ==> scan_metadata ==> download artifacts


# scan_id, workflow_id ==> results


# parse ==> for each artifact ==> map with app pose results 


# visualise on each artifact ==> generate result images

# send result image
# send result object


