import os
import cv2
import uuid
import logging
import face_recognition
import numpy as np
from bunch import Bunch
from datetime import datetime
from fastcore.basics import store_attr


resize_factor_for_scan_version = {
    "v0.1": 3,
    "v0.2": 3,
    "v0.4": 3,
    "v0.5": 3,
    "v0.6": 3,
    "v0.7": 4,
    "v0.8": 1,
    "v0.9": 1,
    "v1.0": 1,
}

FRONT_SCANS = [100, 200]
BACK_SCANS = [102, 202]
ROTATED_SCANS = [101, 201]


def front_scan_inference(faces_detected):
    """ provide the model assumpition for front scan"""
    num_of_faces = np.array(faces_detected)
    if np.any((num_of_faces == 0)):
        prediction = 'inaccurate'
    else:
        prediction = 'accurate'
    return f"{prediction}"


def back_scan_inference(faces_detected):
    """ provide the model assumpition for back scan"""
    num_of_faces = np.array(faces_detected)
    if np.all(num_of_faces == 0):
        prediction = 'accurate'
    else:
        prediction = 'inaccurate'
    return f"{prediction}"


def rotated_scan_inference():
    """ provide the model assumpition for 360 scan"""
    prediction = 'unsure'
    return f"{prediction}"


class BlurFlow:
    """Face blur results generation"""

    def __init__(
            self,
            result_generation,
            workflow_path,
            workflow_faces_path,
            blur_worklfow_scan_path,
            artifacts,
            scan_step,
            scan_version):
        store_attr('result_generation,artifacts,workflow_path,workflow_faces_path,blur_worklfow_scan_path,artifacts,scan_step,scan_version', self)
        self.workflow_obj = self.result_generation.workflows.load_workflows(self.workflow_path)
        self.workflow_faces_obj = self.result_generation.workflows.load_workflows(self.workflow_faces_path)
        self.blur_workflow_scan_obj = self.result_generation.workflows.load_workflows(self.blur_worklfow_scan_path)
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
        self.blur_workflow_scan_obj['id'] = self.result_generation.workflows.get_workflow_id(
            self.blur_workflow_scan_obj['name'], self.blur_workflow_scan_obj['version'])

        self.scan_step = scan_step
        self.scan_version = scan_version

    def run_flow(self):
        """Driver method for blur flow"""
        self.blur_set_resize_factor()
        self.blur_artifacts()
        self.post_blur_files()
        self.post_result_object()

    def blur_artifacts(self):
        """Blur the list of artifacts"""
        self.scan_level_faces = []
        for artifact in self.artifacts:
            input_path = self.result_generation.get_input_path(self.scan_directory, artifact['file'])
            print(f"input_path of image to perform blur: {input_path}\n")
            blur_img_binary, blur_status, faces_detected = self.blur_face(input_path)
            if blur_status:
                artifact['blurred_image'] = blur_img_binary
                artifact['faces_detected'] = faces_detected
                self.scan_level_faces.append(faces_detected)

    def blur_set_resize_factor(self):
        if self.scan_version in resize_factor_for_scan_version:
            self.resize_factor = resize_factor_for_scan_version[self.scan_version]
        else:
            # Default Resize factor to 1
            print("New Scan Version Type")
            self.resize_factor = 1

        print("resize_factor is ", self.resize_factor)
        print("scan_version is ", self.scan_version)

    def blur_img_transformation_using_scan_version(self, rgb_image):
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
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        print("scan_version is ", self.scan_version)
        print("swapped image axis")
        return image

    def blur_face(self, source_path: str) -> bool:
        """Run face blur on given source_path

        Returns:
            bool: True if blurred otherwise False
        """
        # Read the image.
        assert os.path.exists(source_path), f"{source_path} does not exist"
        rgb_image = cv2.imread(str(source_path))
        print("image_size:", rgb_image.shape)

        image = self.blur_img_transformation_using_scan_version(rgb_image)

        # Scale image down for faster prediction.
        small_image = cv2.resize(
            image, (0, 0), fx=1.0 / self.resize_factor, fy=1.0 / self.resize_factor)

        # Find face locations.
        face_locations = face_recognition.face_locations(small_image, model="cnn")

        faces_detected = len(face_locations)

        # Blur the image.
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was
            # scaled to 1/4 size
            top *= self.resize_factor
            right *= self.resize_factor
            bottom *= self.resize_factor
            left *= self.resize_factor

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
        print(f"{len(face_locations)} face locations found and blurred for path: {source_path}\n")
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
            print("successfully post blur results: ", res_object)

        faces_res = self.prepare_faces_result_object()
        faces_res_object = self.result_generation.bunch_object_to_json_object(faces_res)
        if self.result_generation.api.post_results(faces_res_object) == 201:
            print("successfully post faces detected results: ", faces_res_object)

        scan_level_assumption = self.blur_scan_level_result()
        scan_level_assumption_object = self.result_generation.bunch_object_to_json_object(scan_level_assumption)
        if self.result_generation.api.post_results(scan_level_assumption_object) == 201:
            print("successfully post scan level results: ", scan_level_assumption_object)

    def get_scan_results(self, scan_step, scan_level_faces):
        if scan_step in FRONT_SCANS:
            assumption = front_scan_inference(scan_level_faces)
        elif scan_step in BACK_SCANS:
            assumption = back_scan_inference(scan_level_faces)
        elif scan_step in ROTATED_SCANS:
            assumption = rotated_scan_inference()
        else:
            logging.info("unknown scan_step %", scan_level_faces)
        return assumption

    def blur_scan_level_result(self):
        """Prepare scan level blur result object"""
        res = Bunch(dict(results=[]))
        result = Bunch(dict(
            id=f"{uuid.uuid4()}",
            scan=self.result_generation.scan_metadata['id'],
            workflow=self.blur_workflow_scan_obj["id"],
            source_artifacts=[artifact['id'] for artifact in self.artifacts],
            source_results=[],
            generated=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        ))
        assumption = self.get_scan_results(self.scan_step, self.scan_level_faces)
        result.data = {'model_prediction': assumption}
        res.results.append(result)
        return res
