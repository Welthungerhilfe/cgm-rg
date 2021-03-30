import json
import os
import uuid
from datetime import datetime

# import cv2
# import face_recognition
import numpy as np
from bunch import Bunch

RESIZE_FACTOR = 4


class BlurFlow:
    """
    A class to handle face blur results generation.
    
    Args:
        api (object): object of ApiEndpoints class
        workflows (list): list of registered workflows
        workflow_path (str): path of the workflow file for face blurring
        artifacts (list): list of artifacts to run blur flow on
        scan_parent_dir (str): directory where scans are stored
        scan_metadata (json): metadata of the scan to run blur flow on
    """

    def __init__(
            self,
            api,
            workflows,
            workflow_path,
            artifacts,
            scan_parent_dir,
            scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.workflow_path = workflow_path
        self.workflow_obj = self.workflows.load_workflows(self.workflow_path)
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        self.scan_directory = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'],
            self.blur_input_format)
        self.workflow_obj['id'] = self.workflows.get_workflow_id(
            self.workflow_obj['name'], self.workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        """
        Convert a bunch object to json object.
        
        Args:
            bunch_object (Bunch): The data to be converted as bunch_object.

        Returns:
            Returns the data as a json object.

        """
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)

        return json_object

    def get_input_path(self, directory, file_name):
        """
        Returns the input path for given directory and filename
        
        Args:
            directory (str): directory of the file.
            file_name (str): name of the file.

        Returns:
            Returns the input_path as a string.

        """
        return os.path.join(directory, file_name)

    def run_blur_flow(self):
        """
        Driver method for blur flow.
        """
        self.blur_artifacts()
        self.post_blur_files()
        self.post_result_object()

    def blur_artifacts(self):
        """
        Runs face blurring on all the artifacts.
        """
        for i, artifact in enumerate(self.artifacts):

            input_path = self.get_input_path(
                self.scan_directory,
                artifact['file'])
            # target_path = input_path + '_blur.jpg'

            print("input_path of image to perform blur: ", input_path, '\n')

            # blur_status = blur_faces_in_file(input_path, target_path)
            blur_img_binary, blur_status = self.blur_face(input_path)

            if blur_status:
                artifact['blurred_image'] = blur_img_binary

    def blur_face(self, source_path: str):
        """
        Blur image.
        
        Returns:
            bool: True if blurred otherwise False
        """
        # Read the image.
        assert os.path.exists(source_path), f"{source_path} does not exist"
        rgb_image = cv2.imread(source_path)
        image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

        # The images are provided in 90degrees turned. Here we rotate 90degress to
        # the right.
        image = np.swapaxes(image, 0, 1)

        # Scale image down for faster prediction.
        small_image = cv2.resize(
            image, (0, 0), fx=1.0 / RESIZE_FACTOR, fy=1.0 / RESIZE_FACTOR)

        # Find face locations.
        face_locations = face_recognition.face_locations(
            small_image, model="cnn")

        # Blur the image.
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was
            # scaled to 1/4 size
            top *= RESIZE_FACTOR
            right *= RESIZE_FACTOR
            bottom *= RESIZE_FACTOR
            left *= RESIZE_FACTOR

            # Extract the region of the image that contains the face.
            face_image = image[top:bottom, left:right]

            # Blur the face image.
            face_image = cv2.GaussianBlur(
                face_image, ksize=(99, 99), sigmaX=30)

            # Put the blurred face region back into the frame image.
            image[top:bottom, left:right] = face_image

        # Rotate image back.
        image = np.swapaxes(image, 0, 1)

        # Write image to hard drive.
        rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

        # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
        print(
            f"{len(face_locations)} face locations found and blurred for path: {source_path}\n")
        return rgb_image, True

    def post_blur_files(self):
        """
        Posts all the blur files using API and stores the file_id returned by API.
        """
        for artifact in self.artifacts:
            blur_id_from_post_request, post_status = self.api.post_files(
                artifact['blurred_image'])
            if post_status == 201:
                artifact['blur_id_from_post_request'] = blur_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime(
                    '%Y-%m-%dT%H:%M:%SZ')

    def prepare_result_object(self):
        """
        Prepares a blur result object according to specifications of the API.

        Returns:
            result object as a Bunch object.
        """
        res = Bunch()
        res.results = []
        for artifact in self.artifacts:
            blur_result = Bunch()
            blur_result.id = f"{uuid.uuid4()}"
            blur_result.scan = self.scan_metadata['id']
            blur_result.workflow = self.workflow_obj["id"]
            blur_result.source_artifacts = [artifact['id']]
            blur_result.source_results = []
            blur_result.file = artifact['blur_id_from_post_request']
            blur_result.generated = artifact['generated_timestamp']
            res.results.append(blur_result)

        return res

    def post_result_object(self):
        """
        Posts the prepared blur results to the API
        """
        blur_result = self.prepare_result_object()
        blur_result_object = self.bunch_object_to_json_object(blur_result)
        if self.api.post_results(blur_result_object) == 201:
            print("successfully post blur results: ", blur_result_object)
