import os
import cv2
import json
import uuid
import copy
import pprint
import argparse
from datetime import datetime
import numpy as np
import face_recognition
from bunch import Bunch
from api_endpoints import ApiEndpoints
import utils.inference as inference
import utils.preprocessing as preprocessing

RESIZE_FACTOR = 4


class BlurFlow:
    """
    A class to handle face blur results generation.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    workflows : list
        list of registered workflows
    workflow_path : str
        path of the workflow file for face blurring
    artifacts : list
        list of artifacts to run blur flow on
    scan_parent_dir : str
        directory where scans are stored
    scan_metadata : json
        metadata of the scan to run blur flow on

    Methods
    -------
    bunch_object_to_json_object(bunch_object):
        Converts given bunch object to json object.
    get_input_path(directory, file_name):
        Returns input path for given directory name and file name.
    run_blur_flow():
        Driver method for blur flow.
    blur_artifacts():
        Blurs the list of artifacts.
    blur_face(source_path):
        Runs face blur on given source_path.
    post_blur_files():
        Posts the blurred file to api.
    prepare_result_object():
        Prepares result object for results generated.
    post_result_object():
        Posts the result object to api.
    """
    def __init__(self, api, workflows, workflow_path, artifacts, scan_parent_dir, scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.workflow_path = workflow_path
        self.workflow_obj = self.workflows.load_workflows(self.workflow_path)
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        self.scan_directory = os.path.join(self.scan_parent_dir, self.scan_metadata['id'], self.blur_input_format)
        self.workflow_obj['id'] = self.workflows.get_workflow_id(self.workflow_obj['name'], self.workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)

        return json_object

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def run_blur_flow(self):
        self.blur_artifacts()
        self.post_blur_files()
        self.post_result_object()

    def blur_artifacts(self):
        for i, artifact in enumerate(self.artifacts):

            input_path = self.get_input_path(self.scan_directory, artifact['file'])
            # target_path = input_path + '_blur.jpg'

            print("input_path of image to perform blur: ", input_path)

            # blur_status = blur_faces_in_file(input_path, target_path)
            blur_img_binary, blur_status = self.blur_face(input_path)

            if blur_status:
                artifact['blurred_image'] = blur_img_binary

    def blur_face(self, source_path: str):
        """Blur image
        Returns:
            tuple: (blurred_rgb_image, boolean: True if blurred otherwise False)
        """
        # Read the image.
        assert os.path.exists(source_path), f"{source_path} does not exist"
        rgb_image = cv2.imread(source_path)
        image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

        # The images are provided in 90degrees turned. Here we rotate 90degress to
        # the right.
        image = np.swapaxes(image, 0, 1)

        # Scale image down for faster prediction.
        small_image = cv2.resize(image, (0, 0), fx=1.0 / RESIZE_FACTOR, fy=1.0 / RESIZE_FACTOR)

        # Find face locations.
        face_locations = face_recognition.face_locations(small_image, model="cnn")

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
            face_image = cv2.GaussianBlur(face_image, ksize=(99, 99), sigmaX=30)

            # Put the blurred face region back into the frame image.
            image[top:bottom, left:right] = face_image

        # Rotate image back.
        image = np.swapaxes(image, 0, 1)

        # Write image to hard drive.
        rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

        # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
        print(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
        return rgb_image, True

    def post_blur_files(self):
        for artifact in self.artifacts:
            blur_id_from_post_request, post_status = self.api.post_files(artifact['blurred_image'])
            if post_status == 201:
                artifact['blur_id_from_post_request'] = blur_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    def prepare_result_object(self):
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
        blur_result = self.prepare_result_object()
        blur_result_object = self.bunch_object_to_json_object(blur_result)
        if self.api.post_results(blur_result_object) == 201:
            print("successfully post blur results: ", blur_result_object)


class HeightFlow:
    """
    A class to handle height results generation.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    workflows : list
        list of registered workflows
    artifact_workflow_path : str
        path of the workflow file for artifact level height results
    scan_workflow_path : json
        path of the workflow file for scan level height results
    artifacts : list
        list of artifacts to run heigth flow on
    scan_parent_dir : str
        directory where scans are stored
    scan_metadata : json
        metadata of the scan to run height flow on

    Methods
    -------
    bunch_object_to_json_object(bunch_object):
        Converts given bunch object to json object.
    get_input_path(directory, file_name):
        Returns input path for given directory name and file name.
    get_mean_scan_results(predictions):
        Returns the average prediction from given list of predictions.
    process_depthmaps():
        Loads the list of depthmaps in scan as numpy array.
    run_height_flow():
        Driver method for height flow.
    artifact_level_height_result_object(predictions, generated_timestamp):
        Prepares artifact level height result object.
    scan_level_height_result_object(predictions, generated_timestamp):
        Prepares scan level height result object.
    post_height_results(predictions, generated_timestamp):
        Posts the artifact and scan level height results to api.
    """
    def __init__(self, api, workflows, artifact_workflow_path, scan_workflow_path, artifacts, scan_parent_dir, scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.artifact_workflow_path = artifact_workflow_path
        self.scan_workflow_path = scan_workflow_path
        self.artifact_workflow_obj = self.workflows.load_workflows(self.artifact_workflow_path)
        self.scan_workflow_obj = self.workflows.load_workflows(self.scan_workflow_path)
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.artifact_workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
        self.scan_directory = os.path.join(self.scan_parent_dir, self.scan_metadata['id'], self.depth_input_format)
        self.artifact_workflow_obj['id'] = self.workflows.get_workflow_id(self.artifact_workflow_obj['name'], self.artifact_workflow_obj['version'])
        self.scan_workflow_obj['id'] = self.workflows.get_workflow_id(self.scan_workflow_obj['name'], self.scan_workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)

        return json_object

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def get_mean_scan_results(self, predictions):
        return str(np.mean(predictions))

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(self.scan_directory, artifact['file'])

            data, width, height, depthScale, max_confidence = preprocessing.load_depth(input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)

        return depthmaps

    def run_height_flow(self):
        depthmaps = self.process_depthmaps()
        height_predictions = inference.get_height_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_height_results(height_predictions, generated_timestamp)

    def artifact_level_height_result_object(self, predictions, generated_timestamp):
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.artifacts, predictions):
            height_result = Bunch()
            height_result.id = f"{uuid.uuid4()}"
            height_result.scan = self.scan_metadata['id']
            height_result.workflow = self.artifact_workflow_obj["id"]
            height_result.source_artifacts = [artifact['id']]
            height_result.source_results = []
            height_result.generated = generated_timestamp
            result = {'height': str(prediction[0])}
            height_result.data = result
            res.results.append(height_result)

        return res

    def scan_level_height_result_object(self, predictions, generated_timestamp):
        res = Bunch()
        res.results = []
        height_result = Bunch()
        height_result.id = f"{uuid.uuid4()}"
        height_result.scan = self.scan_metadata['id']
        height_result.workflow = self.scan_workflow_obj["id"]
        height_result.source_artifacts = [artifact['id'] for artifact in self.artifacts]
        height_result.source_results = []
        height_result.generated = generated_timestamp
        mean_prediction = self.get_mean_scan_results(predictions)
        result = {'mean_height': mean_prediction}
        height_result.data = result

        res.results.append(height_result)

        return res

    def post_height_results(self, predictions, generated_timestamp):
        artifact_level_height_result_bunch = self.artifact_level_height_result_object(predictions, generated_timestamp)
        artifact_level_height_result_json = self.bunch_object_to_json_object(artifact_level_height_result_bunch)
        if self.api.post_results(artifact_level_height_result_json) == 201:
            print("successfully post artifact level height results: ", artifact_level_height_result_json)

        scan_level_height_result_bunch = self.artifact_level_height_result_object(predictions, generated_timestamp)
        scan_level_height_result_json = self.bunch_object_to_json_object(scan_level_height_result_bunch)
        if self.api.post_results(scan_level_height_result_json) == 201:
            print("successfully post scan level height results: ", scan_level_height_result_json)


class WeightFlow:
    """
    A class to handle weight results generation.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    workflows : list
        list of registered workflows
    artifact_workflow_path : str
        path of the workflow file for artifact level weight results
    scan_workflow_path : json
        path of the workflow file for scan level weight results
    artifacts : list
        list of artifacts to run weigth flow on
    scan_parent_dir : str
        directory where scans are stored
    scan_metadata : json
        metadata of the scan to run weight flow on

    Methods
    -------
    bunch_object_to_json_object(bunch_object):
        Converts given bunch object to json object.
    get_input_path(directory, file_name):
        Returns input path for given directory name and file name.
    get_mean_scan_results(predictions):
        Returns the average prediction from given list of predictions.
    process_depthmaps():
        Loads the list of depthmaps in scan as numpy array.
    run_weight_flow():
        Driver method for weight flow.
    artifact_level_weight_result_object(predictions, generated_timestamp):
        Prepares artifact level weight result object.
    scan_level_weight_result_object(predictions, generated_timestamp):
        Prepares scan level weight result object.
    post_weight_results(predictions, generated_timestamp):
        Posts the artifact and scan level weight results to api.
    """
    def __init__(self, api, workflows, artifact_workflow_path, scan_workflow_path, artifacts, scan_parent_dir, scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.artifact_workflow_path = artifact_workflow_path
        self.scan_workflow_path = scan_workflow_path
        self.artifact_workflow_obj = self.workflows.load_workflows(self.artifact_workflow_path)
        self.scan_workflow_obj = self.workflows.load_workflows(self.scan_workflow_path)
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.artifact_workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
        self.scan_directory = os.path.join(self.scan_parent_dir, self.scan_metadata['id'], self.depth_input_format)
        self.artifact_workflow_obj['id'] = self.workflows.get_workflow_id(self.artifact_workflow_obj['name'], self.artifact_workflow_obj['version'])
        self.scan_workflow_obj['id'] = self.workflows.get_workflow_id(self.scan_workflow_obj['name'], self.scan_workflow_obj['version'])

    def bunch_object_to_json_object(self, bunch_object):
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)

        return json_object

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def get_mean_scan_results(self, predictions):
        return str(np.mean(predictions))

    def process_depthmaps(self):
        depthmaps = []
        for artifact in self.artifacts:
            input_path = self.get_input_path(self.scan_directory, artifact['file'])

            data, width, height, depthScale, max_confidence = preprocessing.load_depth(input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)

        return depthmaps

    def run_weight_flow(self):
        depthmaps = self.process_depthmaps()
        weight_predictions = inference.get_weight_predictions_local(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.post_weight_results(weight_predictions, generated_timestamp)

    def artifact_level_weight_result_object(self, predictions, generated_timestamp):
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.artifacts, predictions):
            weight_result = Bunch()
            weight_result.id = f"{uuid.uuid4()}"
            weight_result.scan = self.scan_metadata['id']
            weight_result.workflow = self.artifact_workflow_obj["id"]
            weight_result.source_artifacts = [artifact['id']]
            weight_result.source_results = []
            weight_result.generated = generated_timestamp
            result = {'weight': str(prediction[0])}
            weight_result.data = result
            res.results.append(weight_result)

        return res

    def scan_level_weight_result_object(self, predictions, generated_timestamp):
        res = Bunch()
        res.results = []
        weight_result = Bunch()
        weight_result.id = f"{uuid.uuid4()}"
        weight_result.scan = self.scan_metadata['id']
        weight_result.workflow = self.scan_workflow_obj["id"]
        weight_result.source_artifacts = [artifact['id'] for artifact in self.artifacts]
        weight_result.source_results = []
        weight_result.generated = generated_timestamp
        mean_prediction = self.get_mean_scan_results(predictions)
        result = {'mean_weight': mean_prediction}
        weight_result.data = result

        res.results.append(weight_result)

        return res

    def post_weight_results(self, predictions, generated_timestamp):
        artifact_level_weight_result_bunch = self.artifact_level_weight_result_object(predictions, generated_timestamp)
        artifact_level_weight_result_json = self.bunch_object_to_json_object(artifact_level_weight_result_bunch)
        if self.api.post_results(artifact_level_weight_result_json) == 201:
            print("successfully post artifact level weight results: ", artifact_level_weight_result_json)

        scan_level_weight_result_bunch = self.artifact_level_weight_result_object(predictions, generated_timestamp)
        scan_level_weight_result_json = self.bunch_object_to_json_object(scan_level_weight_result_bunch)
        if self.api.post_results(scan_level_weight_result_json) == 201:
            print("successfully post scan level weight results: ", scan_level_weight_result_json)


class ProcessWorkflows:
    """
    A class to process all the workflows.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class

    Methods
    -------
    get_list_of_worflows():
        Gets the list of workflows from api

    get_workflow_id(workflow_name, workflow_version):
        Gets the id of the workflow for given workflow name and version

    load_workflows(workflow_path):
        Loads the workflow from given path
    """

    def __init__(self, api):
        # self.workflow_object = workflow_object
        self.api = api

    def get_list_of_worflows(self):
        self.workflows = self.api.get_workflows()

    def get_workflow_id(self, workflow_name, workflow_version):
        workflow_obj_with_id = list(filter(lambda workflow: (workflow['name'] == workflow_name and workflow['version'] == workflow_version), self.workflows['workflows']))[0]

        return workflow_obj_with_id['id']

    def load_workflows(self, workflow_path):
        with open(workflow_path, 'r') as f:
            workflow_obj = json.load(f)

        return workflow_obj


class GetScanMetadata:
    """
    A class to get and process scan metadata.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    scan_metadata_path : str
        path to store scan metadata

    Methods
    -------
    get_unprocessed_scans():
        Returns the no of scans in scan metadata.

    get_scan_metadata():
        Returns the scan metadata
    """
    def __init__(self, api, scan_metadata_path):
        """
        Constructs all the necessary attributes for the GetScanMetadata object.

        Parameters
        ----------
            api : objects
                object of api_endpoints class
            scan_metadata_path : str
                path to store scan metadata
        """
        self.api = api
        self.scan_metadata_path = scan_metadata_path

    def get_unprocessed_scans(self):
        """
        Gets unprocessed_scans from api and returns the no of scans

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        return self.api.get_scan(self.scan_metadata_path)

    def get_scan_metadata(self):
        with open(self.scan_metadata_path, 'r') as f:
            scan_metadata_obj = json.load(f)
        scan_metadata = scan_metadata_obj['scans'][0]

        return scan_metadata


class PrepareArtifacts:
    """
    A class to prepare artifacts for result generation.

    Attributes
    ----------
    api : object
        object of ApiEndpoints class
    scan_metadata : json
        metadata of the scan to run weight flow on
    scan_parent_dir : str
        directory where scans are stored

    Methods
    -------
    download_artifacts(input_format):
        Download artifacts for the scan
    check_artifact_format():
        Checks the format of the artifact
    add_artifacts_to_format_dictionary():
        Sort artifacts according to input format
    process_scan_metadata():
        Process artifacts in a scan.
    create_scan_dir():
        Create directory to store artifacts in scan.
    create_artifact_dir():
        Create directory to store downloaded artifacts.
    """
    def __init__(self, api, scan_metadata, scan_parent_dir):
        self.api = api
        self.scan_metadata = scan_metadata
        self.format_wise_artifact = {}
        self.scan_parent_dir = scan_parent_dir
        self.scan_dir = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'])

    def download_artifacts(self, input_format):
        print(f"\nDownloading Artifacts for { input_format } format")
        self.artifacts = []

        for i, artifact in enumerate(self.format_wise_artifact[input_format]):
            mod_artifact = copy.deepcopy(artifact)

            print("\nDownloading Artifact Name: ", mod_artifact["file"])
            status_code = self.api.get_files(mod_artifact["file"], os.path.join(self.scan_dir, input_format))
            # status_code = get_files_mockup(mod_artifact["file"], format_dir)
            if status_code == 200:
                mod_artifact['download_status'] = True
                self.artifacts.append(mod_artifact)

        print(f"\nBelow Artifacts for { input_format } workflow")
        print(self.artifacts)
        print("\nDownload Artifact for completed")

        return self.artifacts

    def check_artifact_format(self, format):
        if format in ['image/jpeg', 'rgb']:
            return 'img'
        elif format in ['application/zip', 'depth']:
            return 'depth'

    def add_artifacts_to_format_dictionary(self, format, artifact):
        if format in self.format_wise_artifact:
            self.format_wise_artifact[format].append(artifact)
        else:
            self.format_wise_artifact[format] = [artifact]

    def process_scan_metadata(self):
        '''
        Process the scan object to get the list of jpeg id
        and artifact id return a dict of format as key and
        list of file id as values
        '''
        artifact_list = self.scan_metadata['artifacts']

        for artifact in artifact_list:
            mod_artifact = copy.deepcopy(artifact)
            mod_artifact['download_status'] = False

            mod_artifact['format'] = self.check_artifact_format(artifact['format'])

            self.add_artifacts_to_format_dictionary(mod_artifact['format'], mod_artifact)

        print("\nPrepared format wise Artifact:")
        pprint.pprint(self.format_wise_artifact)

    def create_scan_dir(self):
        '''
        Create a scan dir and format wise dir inside scan dir
        in which all the artifacts will be downloaded
        .
        └── scans
            ├── 3fa85f64-5717-4562-b3fc-2c963f66afa6
            │   └── img
            │       ├── 3fa85f64-5717-4562-b3fc-2c963f6shradul
            │       ├── 3fa85f64-5717-4562-b3fc-2c963fmayank
            │       ├── 69869078-33e1-11eb-af63-cf4006664c92
            │       └── 699b71dc-33e1-11eb-af63-e32a5809de47
            └── 59560ba2-33e1-11eb-af63-4b01606d9610
                └── img
                    ├── 5850e04c-33e1-11eb-af63-4f5622046249
                    └── 5850e04c-33e1-11eb-af63-4f5622046249_blur.jpg
        '''
        os.makedirs(self.scan_dir, exist_ok=False)

    def create_artifact_dir(self):
        for artifact_format in self.format_wise_artifact:
            os.makedirs(os.path.join(self.scan_dir, artifact_format), exist_ok=False)


def main():
    parser = argparse.ArgumentParser(
        description='Please provide model_id and workflow paths.')

    '''
    parser.add_argument('--url',
                        default="http://localhost:5001",
                        type=str,
                        help='API endpoint URL')
    '''

    parser.add_argument('--scan_parent_dir',
                        default="data/scans/",
                        type=str,
                        help='Parent directory in which scans will be stored')

    parser.add_argument('--blur_workflow_path',
                        default="src/workflows/blur-worflow-post.json",
                        type=str,
                        help='Blur Workflow path')

    parser.add_argument('--height_workflow_artifact_path',
                        default="src/workflows/height-worflow-artifact.json",
                        type=str,
                        help='Height Workflow Artifact path')
    parser.add_argument('--height_workflow_scan_path',
                        default="src/workflows/height-worflow-scan.json",
                        type=str,
                        help='Height Workflow Scan path')

    parser.add_argument('--weight_workflow_artifact_path',
                        default="/app/src/workflows/weight-worflow-artifact.json",
                        type=str,
                        help='Weight Workflow Artifact path')
    parser.add_argument('--weight_workflow_scan_path',
                        default="/app/src/workflows/weight-worflow-scan.json",
                        type=str,
                        help='Weight Workflow Scan path')

    args = parser.parse_args()

    preprocessing.set_width(int(240 * 0.75))
    preprocessing.set_height(int(180 * 0.75))

    print("\nApp Environment : ", os.environ['APP_ENV'])

    if os.environ['APP_ENV'] == 'LOCAL':
        url = "http://localhost:5001"
    elif os.environ['APP_ENV'] == 'SANDBOX':
        url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
    elif os.environ['APP_ENV'] == 'DEMO':
        url = "https://cgm-be-ci-qa-scanner-api.azurewebsites.net"

    scan_endpoint = '/api/scans/unprocessed?limit=1'
    get_file_endpoint = '/api/files/'
    post_file_endpoint = '/api/files'
    result_endpoint = '/api/results'
    workflow_endpoint = '/api/workflows'

    scan_parent_dir = args.scan_parent_dir
    blur_workflow_path = args.blur_workflow_path
    height_workflow_artifact_path = args.height_workflow_artifact_path
    height_workflow_scan_path = args.height_workflow_scan_path
    weight_workflow_artifact_path = args.weight_workflow_artifact_path
    weight_workflow_scan_path = args.weight_workflow_scan_path

    scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

    cgm_api = ApiEndpoints(
        url,
        scan_endpoint,
        get_file_endpoint,
        post_file_endpoint,
        result_endpoint,
        workflow_endpoint)

    workflow = ProcessWorkflows(cgm_api)

    get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)

    if get_scan_metadata.get_unprocessed_scans() > 0:
        scan_metadata = get_scan_metadata.get_scan_metadata()
        workflow.get_list_of_worflows()
        data_processing = PrepareArtifacts(cgm_api, scan_metadata, scan_parent_dir)
        data_processing.process_scan_metadata()
        data_processing.create_scan_dir()
        data_processing.create_artifact_dir()
        rgb_artifacts = data_processing.download_artifacts('img')
        depth_artifacts = data_processing.download_artifacts('depth')

        blurflow = BlurFlow(cgm_api, workflow, blur_workflow_path, rgb_artifacts, scan_parent_dir, scan_metadata)
        heightflow = HeightFlow(cgm_api, workflow, height_workflow_artifact_path, height_workflow_scan_path, depth_artifacts, scan_parent_dir, scan_metadata)
        weightflow = WeightFlow(cgm_api, workflow, weight_workflow_artifact_path, weight_workflow_scan_path, depth_artifacts, scan_parent_dir, scan_metadata)

        blurflow.run_blur_flow()
        heightflow.run_height_flow()
        weightflow.run_weight_flow()


if __name__ == "__main__":
    main()
