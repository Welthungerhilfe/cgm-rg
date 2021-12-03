import uuid
from datetime import datetime

import cv2
import numpy as np
from bunch import Bunch
import log
from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.models.HRNET.hrnet3d import (convert_2dskeleton_to_3d,
                                        write_skeleton_into_obj)
from joblib import load
from result_generation.height.height import HeightFlow
from result_generation.pose_prediction.inference import (inference_artifact,
                                                         init_pose_prediction)
from utils.config_train import CONFIG_TRAIN
from utils.pose_utils import get_features_from_fpath

logger = log.setup_custom_logger(__name__)


class HeightFlowPose3D(HeightFlow):
    def run_flow(self):
        if self.scan_version in ["v0.9", "v1.1.0", "v1.0.2"]:
            start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.info("%s", "Result genration Pose 3d start")
            if self.scan_type in [200, 201, 202]:
                mean_prediction = self.pose_prediction_artifacts()
            else:
                mean_prediction = 0
            generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            self.post_height_results(mean_prediction, generated_timestamp, start_time)

    def pose_prediction_artifacts(self):
        """Blur the list of artifacts"""
        pose_prediction = init_pose_prediction()
        logger.info("%s", "Pose 3d Model loading")
        MODEL = load('/app/models/pose-3d/2021q4-points3d-rf-height-28k-200and201.joblib')
        mean_prediction = 0
        processed_artifacts = 0
        for image_artifact, artifact in zip(self.image_artifacts, self.artifacts):
            logger.info("%s %s", "Order id ", artifact['order'])
            if(image_artifact['order'] == artifact['order']):
                logger.info("%s ", "Order id Matched ")
                input_rgb_path = self.result_generation.get_input_path(self.scan_rgb_directory, image_artifact['file'])
                logger.info("%s %s", "input_path of image to perform Pose prediction for Pose-3d:", input_rgb_path)
                no_of_body_pose, _, _, self.persons_coordinates = inference_artifact(
                    pose_prediction, input_rgb_path, self.scan_type)
                if no_of_body_pose == 1:
                    logger.info("%s %s", "No of body_pose ", no_of_body_pose)
                    input_depth_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
                    self.dmap = Depthmap.create_from_zip_absolute(
                        input_depth_path, 0, '/app/src/result_generation/height/camera_calibration_p30pro_EU.txt')

                    self.floor = self.dmap.get_floor_level()
                    logger.info("%s %s", "Floor Value ", self.floor)
                    rgb = cv2.imread(str(input_rgb_path))
                    dim = (640, int(rgb.shape[0] / rgb.shape[1] * 640.0))
                    self.rgb = cv2.resize(rgb, dim, cv2.INTER_AREA)
                    self.dmap.resize(rgb.shape[1], rgb.shape[0])
                    self.export_object('/app/output_skeleton.obj')
                    obj_file_path = '/app/output_skeleton.obj'
                    child_features = get_features_from_fpath(obj_file_path, config_train=CONFIG_TRAIN)
                    feats = np.array(list(child_features.values()))
                    logger.info("%s %s", "Pose 3d Prediction ", self.floor)
                    prediction = MODEL.predict([feats])[0]
                    artifact['prediction'] = prediction
                    mean_prediction += prediction
                    processed_artifacts += 1
                else:
                    logger.info("%s ", "No Body Pose Detected ")
                    artifact['prediction'] = 0
            else:
                logger.info("%s ", "Order id Not Matched ")

        if processed_artifacts != 0:
            mean_prediction = mean_prediction / processed_artifacts
        return mean_prediction

    def get_person_joints(self) -> list:
        assert self.get_person_count() == 1

        joints = []
        confidences = []
        pose = self.persons_coordinates['pose_result'][0]
        for confidence, joint in zip(pose['key_points_prob'], pose['key_points_coordinate']):
            confidence = float(list(confidence.values())[0]['score'])
            confidences.append(confidence)

            x = int(list(joint.values())[0]['x'])
            y = self.rgb.shape[0] - int(list(joint.values())[0]['y']) - 1
            joints.append([x, y])

        return convert_2dskeleton_to_3d(self.dmap, self.floor, joints, confidences)

    def export_object(self, filepath: str, use_skeleton=True):

        # export 3d skeleton
        joints = self.get_person_joints()
        write_skeleton_into_obj(filepath, joints)

    def get_person_count(self) -> int:
        return self.persons_coordinates['no_of_body_pose_detected']

    def post_height_results(self, mean_prediction, generated_timestamp, start_time):
        """Post the artifact and scan level height results to the API"""
        artifact_level_height_result_bunch = self.artifact_level_result(generated_timestamp, start_time)
        artifact_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            artifact_level_height_result_bunch)
        if self.result_generation.api.post_results(artifact_level_height_result_json) == 201:
            logger.info("%s %s", "successfully post artifact level height results:", artifact_level_height_result_json)

        scan_level_height_result_bunch = self.scan_level_height_result_object(mean_prediction,
                                                                              generated_timestamp, self.scan_workflow_obj, start_time)
        scan_level_height_result_json = self.result_generation.bunch_object_to_json_object(
            scan_level_height_result_bunch)
        if self.result_generation.api.post_results(scan_level_height_result_json) == 201:
            logger.info("%s %s", "successfully post scan level height results:", scan_level_height_result_json)

    def artifact_level_result(self, generated_timestamp, start_time):
        """Prepare artifact level height result object"""
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            result = Bunch(dict(
                id=str(uuid.uuid4()),
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.artifact_workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=generated_timestamp,
                data={'height': str(artifact['prediction'])},
                start_time=start_time,
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)
        return res

    def scan_level_height_result_object(self, mean_prediction, generated_timestamp, workflow_obj, start_time):
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
        class_lhfa = self.zscore_lhfa(str(mean_prediction))
        result.data = {
            'mean_height': str(mean_prediction),
            'Height Diagnosis': class_lhfa}
        res.results.append(result)
        return res
