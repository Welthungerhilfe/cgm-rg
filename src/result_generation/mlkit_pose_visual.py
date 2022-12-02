import os
import sys
import log
import cv2
import uuid
import json
import numpy as np
from bunch import Bunch
from pathlib import Path
from datetime import datetime
from fastcore.basics import store_attr
from api_endpoints import ApiEndpoints

logger = log.setup_custom_logger(__name__)

sys.path.append(str(Path(__file__).parents[1]))
from result_generation.pose_prediction.code.utils.utils import (
    draw_mlkit_pose,
    prepare_draw_kpts
)


class MLkitPoseVisualise:
    """Face blur results generation"""

    def __init__(
            self,
            result_generation,
            workflow_app_pose_path,
            workflow_mlkit_pose_visualize_pose_path,
            artifacts,
            scan_version,
            scan_type,
    ):

        store_attr(
            'result_generation,workflow_app_pose_path,workflow_mlkit_pose_visualize_pose_path,artifacts,scan_version,scan_type',
            self)
        # self.workflow_blur_obj = self.result_generation.workflows.load_workflows(
        #     self.workflow_blur_path
        #     )
        self.workflow_app_pose_obj = self.result_generation.workflows.load_workflows(
            self.workflow_app_pose_path
        )
        self.workflow_mlkit_pose_visualize_obj = self.result_generation.workflows.load_workflows(
            self.workflow_mlkit_pose_visualize_pose_path
        )
        # if self.workflow_blur_obj["data"]["input_format"] == 'image/jpeg':
        #     self.blur_input_format = 'img'
        self.scan_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            'img'
        )

        # self.workflow_blur_obj['id'] = self.result_generation.workflows.get_workflow_id(
        #     self.workflow_blur_obj['name'], self.workflow_blur_obj['version'])

        self.workflow_app_pose_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_app_pose_obj['name'], self.workflow_app_pose_obj['version'])

        self.workflow_mlkit_pose_visualize_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_mlkit_pose_visualize_obj['name'], self.workflow_mlkit_pose_visualize_obj['version'])

    def run_flow(self):
        """Driver method for Mlkit Pose Visualise flow"""
        # self.blur_set_resize_factor()
        logger.info('%s', 'Pose Prediction Started')

        self.get_app_pose_result()
        self.mlkit_pose_visualsation()
        self.post_mlkit_pose_visualization_files()
        self.post_mlkit_pose_visualization_object()

    def get_app_pose_result(self):
        url = os.getenv('APP_URL', 'http://localhost:5001')
        cgm_api = ApiEndpoints(url)

        scan_level_app_pose_results = cgm_api.get_results(
            scan_id=self.result_generation.scan_metadata['id'],
            workflow_id=self.workflow_app_pose_obj['id']
        )

        pose_result_by_artifact_id = {}
        for result in scan_level_app_pose_results:
            if 'poseCoordinates' in result['data']:
                if len(result['source_artifacts']) > 0:
                    pose_result_by_artifact_id[result['source_artifacts'][0]] = json.loads(
                        result['data']['poseCoordinates']
                    )
                else:
                    print("More than one source artifact for result ")
                    print(result['source_artifacts'])

        # from pprint import pprint
        # print("--------------Scan Level Results-------------------------------------")
        # pprint(scan_level_app_pose_results)
        # print("---------------------------------------------------------------------")

        # print("self.artifacts ")
        # print(self.artifacts)

        logger.info("--------------Scan Level Results-------------------------------------")
        logger.info(scan_level_app_pose_results)
        logger.info("---------------------------------------------------------------------")

        logger.info("self.artifacts ")
        logger.info("%s %s", "abc ", self.artifacts)

        for artifact in self.artifacts:
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

    def mlkit_pose_visualsation(self):
        print("============================================================")
        print(self.scan_directory)
        # print(artifact['file'])
        print("============================================================")

        logger.info("============================================================")
        logger.info(self.scan_directory)
        # logger.info(artifact['file'])
        logger.info("============================================================")

        for artifact in self.artifacts:
            artifact['pose_start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            input_path = self.result_generation.get_input_path(
                self.scan_directory, artifact['file']
            )
            logger.info("%s %s", "input_path of image to perform blur:", input_path)
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
            # output_path = str(input_path) +'_pose.jpeg'
            # print(output_path)
            # cv2.imwrite(str(output_path), pose_img)

    def post_mlkit_pose_visualization_files(self):
        """Post the blurred file to the API"""
        for artifact in self.artifacts:
            pose_id_from_post_request, post_status = self.result_generation.api.post_files(
                artifact['mlkit_pose_blurred_image'])
            if post_status == 201:
                artifact['pose_id_from_post_request'] = pose_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    def prepare_mlkit_pose_visualize_object(self):
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_mlkit_pose_visualize_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                file=artifact['pose_id_from_post_request'],
                generated=artifact['generated_timestamp'],
                start_time=artifact['pose_start_time'],
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)

        return res

    def post_mlkit_pose_visualization_object(self):
        res = self.prepare_mlkit_pose_visualize_object()
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            logger.info("%s %s", "successfully post pose results:", res_object)
