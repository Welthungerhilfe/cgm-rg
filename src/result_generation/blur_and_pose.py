import os
import cv2
import uuid
import face_recognition
from bunch import Bunch
from datetime import datetime
from fastcore.basics import store_attr
import sys
from pathlib import Path


import log
sys.path.append(str(Path(__file__).parents[1]))
import utils.inference as inference  # noqa
import utils.preprocessing as preprocessing  # noqa
from result_generation.pose_prediction.inference import inference_artifact, init_pose_prediction
from result_generation.pose_prediction.code.utils.utils import draw_pose


logger = log.setup_custom_logger(__name__)


standing_scan_type = ["101", "102", "103"]
laying_scan_type = ["201", "202", "203"]


class PoseAndBlurFlow:
    """Face blur results generation"""

    def __init__(
            self,
            result_generation,
            workflow_blur_path,
            workflow_faces_path,
            workflow_pose_path,
            workflow_pose_visualize_pose_path,
            artifacts,
            scan_version,
            scan_type,
            workflow_type):

        store_attr(
            'result_generation,artifacts,workflow_blur_path,workflow_faces_path,workflow_pose_path,workflow_pose_visualize_pose_path,artifacts,scan_version,scan_type,workflow_type', self)
        self.workflow_blur_obj = self.result_generation.workflows.load_workflows(self.workflow_blur_path)

        self.workflow_faces_obj = self.result_generation.workflows.load_workflows(self.workflow_faces_path)
        self.workflow_pose_obj = self.result_generation.workflows.load_workflows(self.workflow_pose_path)
        self.workflow_pose_visualize_obj = self.result_generation.workflows.load_workflows(
            self.workflow_pose_visualize_pose_path)
        if self.workflow_blur_obj["data"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        self.scan_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            self.blur_input_format)

        self.workflow_blur_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_blur_obj['name'], self.workflow_blur_obj['version'])
        self.workflow_faces_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_faces_obj['name'], self.workflow_faces_obj['version'])
        self.workflow_pose_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_pose_obj['name'], self.workflow_pose_obj['version'])
        self.workflow_pose_visualize_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_pose_visualize_obj['name'], self.workflow_pose_visualize_obj['version'])

    def run_flow(self):
        """Driver method for blur flow"""
        # self.blur_set_resize_factor()
        self.blur_artifacts()
        if 'BLUR' in self.workflow_type:
            self.post_blur_files()
            self.post_blur_result_object()
        if 'POSE' in self.workflow_type:
            logger.info('%s', 'Pose Prediction Started')
            self.pose_prediction_artifacts()
            self.pose_and_blur_visualsation()
            self.post_pose_with_blur_visualization_files()
            self.post_pose_result_object()
            self.post_pose_with_blur_visualization_object()

    def pose_prediction_artifacts(self):
        """Blur the list of artifacts"""
        pose_prediction = init_pose_prediction()
        for artifact in self.artifacts:
            artifact['pose_start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            logger.info("input_path of image to perform Pose prediction: %s", input_path)
            no_of_pose_detected, pose_score, pose_result = inference_artifact(
                pose_prediction, input_path, self.scan_type)
            logger.info("pose_score no_of_pose_detected %s %s", pose_score, no_of_pose_detected)
            artifact['no_of_pose_detected'] = no_of_pose_detected
            artifact['pose_score'] = pose_score
            artifact['pose_result'] = pose_result

    def blur_artifacts(self):
        """Blur the list of artifacts"""
        for artifact in self.artifacts:
            artifact['blur_start_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            logger.info("input_path of image to perform blur: %s", input_path)
            blur_img_binary, blur_status, faces_detected = self.blur_face(input_path)
            logger.info("Blur Done")

            if blur_status:
                artifact['blurred_image'] = blur_img_binary
                artifact['faces_detected'] = faces_detected

    def pose_and_blur_visualsation(self):
        for artifact in self.artifacts:
            img = artifact['blurred_image']
            if artifact['no_of_pose_detected'] > 0:
                rotated_pose_preds = artifact['pose_result'][0]['draw_kpt']
                for kpt in rotated_pose_preds:
                    img = draw_pose(kpt, img)
            artifact['pose_blurred_image'] = img

    # def blur_set_resize_factor(self):
    #     if self.scan_version in resize_factor_for_scan_version:
    #         self.resize_factor = resize_factor_for_scan_version[self.scan_version]
    #     else:
    #         # Default Resize factor to 1
    #         logger.info("New Scan Version Type")
    #         self.resize_factor = 1
    def orient_img(self, image):
        # The images are rotated 90 degree clockwise for standing children
        # and 90 degree anticlock wise for laying children to make children
        # head at top and toe at bottom
        if self.scan_type in standing_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.scan_type in laying_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image

    def reorient_back(self, image):
        if self.scan_type in standing_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.scan_type in laying_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image

    def blur_img_transformation_using_scan_version_and_scan_type(self, rgb_image):
        if self.scan_version in ["v0.7"]:
            # Make the image smaller, The limit of cgm-api to post an image is 500 KB.
            # Some of the images of v0.7 is greater than 500 KB
            rgb_image = cv2.resize(
                rgb_image, (0, 0), fx=1.0 / 1.3, fy=1.0 / 1.3)

        # print("scan_version is ", self.scan_version)
        image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

        # if self.scan_version in ["v0.1", "v0.2", "v0.4", "v0.5", "v0.6", "v0.7", "v0.8", "v0.9", "v1.0"]:
        # The images are provided in 90degrees turned. Here we rotate 90 degress to
        # the right.
        if self.scan_type in standing_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.scan_type in laying_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        logger.info("scan_version is %s", self.scan_version)
        logger.info("swapped image axis")
        return image

    def blur_face(self, source_path: str) -> bool:
        """Run face blur on given source_path

        Returns:
            bool: True if blurred otherwise False
        """
        # Read the image.
        assert os.path.exists(source_path), f"{source_path} does not exist"
        rgb_image = cv2.imread(str(source_path))

        image = self.blur_img_transformation_using_scan_version_and_scan_type(rgb_image)
        image = self.orient_img(image)

        height, width, channels = image.shape

        resized_height = 500.0
        resize_factor = height / resized_height
        # resized_width = width / resize_factor
        # resized_height, resized_width = int(resized_height), int(resized_width)

        # Scale image down for faster prediction.
        small_image = cv2.resize(
            image, (0, 0), fx=1.0 / resize_factor, fy=1.0 / resize_factor)

        # Find face locations.
        face_locations = face_recognition.face_locations(small_image, model="cnn")

        faces_detected = len(face_locations)
        logger.info("%s face locations found and blurred for path: %s", faces_detected, source_path)

        # Blur the image.
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was
            # scaled to 1/4 size
            top *= resize_factor
            right *= resize_factor
            bottom *= resize_factor
            left *= resize_factor
            top, right, bottom, left = int(top), int(right), int(bottom), int(left)

            # Extract the region of the image that contains the face.
            # TODO rotate -codinate

            face_image = image[top:bottom, left:right]

            # Blur the face image.
            face_image = cv2.GaussianBlur(
                face_image, ksize=(99, 99), sigmaX=30)

            # Put the blurred face region back into the frame image.
            image[top:bottom, left:right] = face_image

        image = self.reorient_back(image)

        # Write image to hard drive.
        rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

        logger.info("%s face locations found and blurred for path: %s", len(face_locations), source_path)
        return rgb_image, True, faces_detected

    def post_blur_files(self):
        """Post the blurred file to the API"""
        for artifact in self.artifacts:
            blur_id_from_post_request, post_status = self.result_generation.api.post_files(
                artifact['blurred_image'])
            if post_status == 201:
                artifact['blur_id_from_post_request'] = blur_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    def post_pose_with_blur_visualization_files(self):
        """Post the blurred file to the API"""
        for artifact in self.artifacts:
            pose_id_from_post_request, post_status = self.result_generation.api.post_files(
                artifact['pose_blurred_image'])
            if post_status == 201:
                artifact['pose_id_from_post_request'] = pose_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    def prepare_no_of_person_result_object(self):
        """Prepare result object for results generated"""
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            no_of_pose_result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_pose_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=artifact['generated_timestamp'],
                data={'no of person using pose': str(artifact['no_of_pose_detected'])},
                start_time=artifact['pose_start_time'],
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(no_of_pose_result)
            for i in range(0, artifact['no_of_pose_detected']):
                pose_score_results = Bunch(dict(
                    id=f"{uuid.uuid4()}",
                    scan=self.result_generation.scan_metadata['id'],
                    workflow=self.workflow_pose_obj["id"],
                    source_artifacts=[artifact['id']],
                    source_results=[],
                    generated=artifact['generated_timestamp'],
                    data={'Pose Scores': str(artifact['pose_score'][i]),
                          'Pose Results': str(artifact['pose_result'][i])},
                    start_time=artifact['pose_start_time'],
                    end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                ))
                res.results.append(pose_score_results)
        return res

    def prepare_pose_visualize_object(self):
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_pose_visualize_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                file=artifact['pose_id_from_post_request'],
                generated=artifact['generated_timestamp'],
                start_time=artifact['pose_start_time'],
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)

        return res

    # def prepare_blur_result_object(self):

    def prepare_result_object(self):
        """Prepare result object for results generated"""
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_blur_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                file=artifact['blur_id_from_post_request'],
                generated=artifact['generated_timestamp'],
                start_time=artifact['blur_start_time'],
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)

        return res

    def prepare_faces_result_object(self):
        """Prepare result object for results generated"""
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_faces_obj['id'],
                source_artifacts=[artifact['id']],
                source_results=[],
                generated=artifact['generated_timestamp'],
                data={'faces_detected': str(artifact['faces_detected'])},
                start_time=artifact['blur_start_time'],
                end_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            res.results.append(result)

        return res

    def post_pose_result_object(self):
        """Post the result object to the API"""
        pose_res = self.prepare_no_of_person_result_object()
        pose_res_object = self.result_generation.bunch_object_to_json_object(pose_res)
        if self.result_generation.api.post_results(pose_res_object) == 201:
            logger.info("successfully post pose detected results: %s", pose_res_object)

    def post_pose_with_blur_visualization_object(self):
        res = self.prepare_pose_visualize_object()
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            logger.info("successfully post pose results: %s", res_object)

    def post_blur_result_object(self):
        """Post the result object to the API"""
        res = self.prepare_result_object()
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            logger.info("successfully post blur results: %s", res_object)

        faces_res = self.prepare_faces_result_object()
        faces_res_object = self.result_generation.bunch_object_to_json_object(faces_res)
        if self.result_generation.api.post_results(faces_res_object) == 201:
            logger.info("successfully post faces detected results: %s", faces_res_object)
