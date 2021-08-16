import os
import cv2
import uuid
import face_recognition
from bunch import Bunch
from datetime import datetime
from fastcore.basics import store_attr

import log


logger = log.setup_custom_logger(__name__)

# resize_factor_for_scan_version = {
#     "v0.1": 3,
#     "v0.2": 3,
#     "v0.4": 3,
#     "v0.5": 3,
#     "v0.6": 3,
#     "v0.7": 4,
#     "v0.8": 1,
#     "v0.9": 1,
#     "v1.0": 1,
# }

standing_scan_type = ["101", "102", "103"]
laying_scan_type = ["201", "202", "203"]


class BlurFlow:
    """Face blur results generation"""

    def __init__(
            self,
            result_generation,
            workflow_path,
            workflow_faces_path,
            artifacts,
            scan_version,
            scan_type):
        store_attr('result_generation,artifacts,workflow_path,workflow_faces_path,artifacts,scan_version,scan_type', self)
        self.workflow_obj = self.result_generation.workflows.load_workflows(self.workflow_path)
        self.workflow_faces_obj = self.result_generation.workflows.load_workflows(self.workflow_faces_path)
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        self.scan_directory = os.path.join(
            self.result_generation.scan_parent_dir,
            self.result_generation.scan_metadata['id'],
            self.blur_input_format)

        self.workflow_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_obj['name'], self.workflow_obj['version'])
        self.workflow_faces_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.workflow_faces_obj['name'], self.workflow_faces_obj['version'])

        self.scan_version = scan_version

    def run_flow(self):
        """Driver method for blur flow"""
        # self.blur_set_resize_factor()
        self.blur_artifacts()
        self.post_blur_files()
        self.post_result_object()

    def blur_artifacts(self):
        """Blur the list of artifacts"""
        for artifact in self.artifacts:
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            logger.info("%s %s", "input_path of image to perform blur:", input_path)
            blur_img_binary, blur_status, faces_detected = self.blur_face(input_path)
            if blur_status:
                artifact['blurred_image'] = blur_img_binary
                artifact['faces_detected'] = faces_detected

    # def blur_set_resize_factor(self):
    #     if self.scan_version in resize_factor_for_scan_version:
    #         self.resize_factor = resize_factor_for_scan_version[self.scan_version]
    #     else:
    #         # Default Resize factor to 1
    #         logger.info("New Scan Version Type")
    #         self.resize_factor = 1

    #     logger.info("%s %s", "resize_factor is", self.resize_factor)
    #     logger.info("%s %s", "scan_version is", self.scan_version)

    def blur_img_transformation_using_scan_version_and_scan_type(self, rgb_image):
        if self.scan_version in ["v0.7"]:
            # Make the image smaller, The limit of cgm-api to post an image is 500 KB.
            # Some of the images of v0.7 is greater than 500 KB
            rgb_image = cv2.resize(
                rgb_image, (0, 0), fx=1.0 / 1.3, fy=1.0 / 1.3)

        # print("scan_version is ", self.scan_version)
        image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

        # if self.scan_version in ["v0.1", "v0.2", "v0.4", "v0.5", "v0.6", "v0.7", "v0.8", "v0.9", "v1.0"]:
        # The images are provided in 90degrees turned. Here we rotate 90degress to
        # the right.
        if self.scan_type in standing_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.scan_type in laying_scan_type:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        logger.info("%s %s", "scan_version is", self.scan_version)
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
        height, width, channels = image.shape

        resized_height = 500.0
        resize_factor = height / resized_height
        # resized_width = width / resize_factor
        # resized_height, resized_width = int(resized_height), int(resized_width)

        # Scale image down for faster prediction.
        # small_image = cv2.resize(
        #     image, (0, 0), fx=1.0 / self.resize_factor, fy=1.0 / self.resize_factor)

        small_image = cv2.resize(
            image, (0, 0), fx=1.0 / resize_factor, fy=1.0 / resize_factor)

        # Find face locations.
        face_locations = face_recognition.face_locations(small_image, model="cnn")

        faces_detected = len(face_locations)

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
            face_image = image[top:bottom, left:right]

            # Blur the face image.
            face_image = cv2.GaussianBlur(
                face_image, ksize=(99, 99), sigmaX=30)

            # Put the blurred face region back into the frame image.
            image[top:bottom, left:right] = face_image

        # if self.scan_version in ["v0.2", "v0.4", "v0.6", "v0.7", "v0.8"]:
        #    # Rotate image back.
        #    image = np.swapaxes(image, 0, 1)

        # Write image to hard drive.
        rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

        # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
        logger.info("%s %s %s", len(face_locations), "face locations found and blurred for path:", source_path)
        return rgb_image, True, faces_detected

    def post_blur_files(self):
        """Post the blurred file to the API"""
        for artifact in self.artifacts:
            blur_id_from_post_request, post_status = self.result_generation.api.post_files(
                artifact['blurred_image'])
            if post_status == 201:
                artifact['blur_id_from_post_request'] = blur_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime(
                    '%Y-%m-%dT%H:%M:%SZ')

    def prepare_result_object(self):
        """Prepare result object for results generated"""
        res = Bunch(dict(results=[]))
        for artifact in self.artifacts:
            result = Bunch(dict(
                id=f"{uuid.uuid4()}",
                scan=self.result_generation.scan_metadata['id'],
                workflow=self.workflow_obj["id"],
                source_artifacts=[artifact['id']],
                source_results=[],
                file=artifact['blur_id_from_post_request'],
                generated=artifact['generated_timestamp'],
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
            ))
            res.results.append(result)

        return res

    def post_result_object(self):
        """Post the result object to the API"""
        res = self.prepare_result_object()
        res_object = self.result_generation.bunch_object_to_json_object(res)
        if self.result_generation.api.post_results(res_object) == 201:
            logger.info("%s %s", "successfully post blur results:", res_object)

        faces_res = self.prepare_faces_result_object()
        faces_res_object = self.result_generation.bunch_object_to_json_object(faces_res)
        if self.result_generation.api.post_results(faces_res_object) == 201:
            logger.info("%s %s", "successfully post faces detected results:", faces_res_object)
