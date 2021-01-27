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
    def __init__(self, api, workflows, workflow_obj, artifacts, scan_parent_dir, scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.workflow_obj = workflow_obj
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.workflow_obj["data"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        self.scan_directory = os.path.join(self.scan_parent_dir, self.scan_metadata['id'], self.blur_input_format)
        self.workflow_obj['id'] = workflows.get_workflow_id(workflow_obj['name'], workflow_obj['version'])

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def run_blur_flow(self):
        '''
        Run the blur Workflow on the downloaded artifacts
        '''
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
        """Blur image
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
        small_image = cv2.resize(image, (0, 0), fx=1.0 /
                                RESIZE_FACTOR, fy=1.0 / RESIZE_FACTOR)

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
        print(f"{len(face_locations)} face locations found and blurred for path: {source_path}\n")
        return rgb_image, True

    def post_blur_files(self):
        for artifact in self.artifacts:
            blur_id_from_post_request, post_status = self.api.post_files(
                    artifact['blurred_image'])
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
            blur_result.source_artifacts = [artifact['file']]
            blur_result.source_results = []
            blur_result.file = artifact['blur_id_from_post_request']
            blur_result.generated = artifact['generated_timestamp']
            res.results.append(blur_result)
        
        return res

    def post_result_object(self):
        blur_result = self.prepare_result_object()
        blur_result_string = json.dumps(blur_result, indent=2, separators=(',', ':'))
        blur_result_object = json.loads(blur_result_string)
        if self.api.post_results(blur_result_object) == 201:
            print("successfully post blur results: ", blur_result_object)


class HeightFlow:
    def __init__(self, api, workflows, artifact_workflow_obj, scan_workflow_obj, artifacts, scan_parent_dir, scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.artifact_workflow_obj = artifact_workflow_obj
        self.scan_workflow_obj = scan_workflow_obj
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.workflow_obj["meta"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        self.scan_directory = os.path.join(self.scan_parent_dir, self.scan_metadata['id'], self.blur_input_format)
        self.artifact_workflow_obj['id'] = workflows.get_workflow_id(artifact_workflow_obj['name'], artifact_workflow_obj['version'])
        self.scan_workflow_obj['id'] = workflows.get_workflow_id(scan_workflow_obj['name'], scan_workflow_obj['version'])
    

class WeightFlow:
    pass


class ProcessWorkflows:
    def __init__(self, api):
        # self.workflow_object = workflow_object
        self.api = api

    def get_list_of_worflows(self):
        self.workflows = self.api.get_workflows()

    def get_workflow_id(self, workflow_name, workflow_version):
        blur_workflow_obj_with_id = list(filter(lambda workflow: (workflow['name'] == workflow_name and workflow['version'] == workflow_version), self.workflows['workflows']))[0]

        return blur_workflow_obj_with_id['id']


class GetScanMetadata:
    def __init__(self, api, scan_metadata_path):
        self.api = api
        self.scan_metadata_path = scan_metadata_path
    
    def get_unprocessed_scans(self):
        
        return self.api.get_scan(self.scan_metadata_path)

    def get_scan_metadata(self):
        with open(self.scan_metadata_path, 'r') as f:
            scan_metadata_obj  = json.load(f)
        self.scan_metadata = scan_metadata_obj['scans'][0]

        return self.scan_metadata


class DataProcessing:
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

        for i, artifact in enumerate(
                self.format_wise_artifact[input_format]):
            mod_artifact = copy.deepcopy(artifact)

            print("\nDownloading Artifact Name: ", mod_artifact["file"], '\n')
            status_code = self.api.get_files(
                mod_artifact["file"], os.path.join(self.scan_dir, input_format))
            # status_code = get_files_mockup(mod_artifact["file"], format_dir)
            if status_code == 200:
                mod_artifact['download_status'] = True
                self.artifacts.append(mod_artifact)

        print(f"\nBelow Artifacts for { input_format } workflow\n")
        print(self.artifacts)
        print("\nDownload Artifact for completed\n")

        return self.artifacts

    def check_artifact_format(self, format):
        if format == 'image/jpeg' or format == 'rgb':
            return 'img'
        elif format == 'application/zip':
            return 'depth'

    def add_artifacts_to_format_dictionary(self, format, artifact):
        if format in self.format_wise_artifact:
            self.format_wise_artifact[format].append(
                artifact)
        else:
            self.format_wise_artifact[format] = [
                artifact]

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

        print("\nPrepared format wise Artifact:\n")
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
        if not os.path.isdir(self.scan_dir):
            os.makedirs(self.scan_dir)

    def create_artifact_dir(self):
        for artifact_format in self.format_wise_artifact:
            if not os.path.exists(
                os.path.join(
                    self.scan_dir,
                    artifact_format)):
                os.makedirs(os.path.join(self.scan_dir, artifact_format))

    def get_processed_depthmaps(self):
        self.depthmaps = []
        for i, artifact in enumerate(self.depth_artifact):
            input_path = self.get_input_path(
                self.depth_artifact_dir,
                artifact['file'])

            data, width, height, depthScale, maxConfidence = preprocessing.load_depth(input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        self.depthmaps = np.array(depthmaps)


class GenerateResults:
    pass


class MakeResultObjects:
    pass


class ScanResults:
    def __init__(self, scan_metadata, blur_workflow, height_workflow, weight_workflow, scan_parent_dir, api):
        self.scan_metadata = scan_metadata
        self.blur_workflow = blur_workflow
        self.height_workflow = height_workflow
        self.weight_workflow = weight_workflow
        self.format_wise_artifact = {}
        if self.blur_workflow["meta"]["input_format"] == 'image/jpeg':
            self.blur_input_format = 'img'
        if self.height_workflow["meta"]["input_format"] == 'application/zip':
            self.height_input_format = 'depth'
        if self.weight_workflow["meta"]["input_format"] == 'application/zip':
            self.weight_input_format = 'depth'
        self.depth_input_format = 'depth'
        self.scan_parent_dir = scan_parent_dir
        self.scan_dir = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'])
        self.api = api
        self.blur_workflow_artifact_dir = os.path.join(
            self.scan_dir, self.blur_input_format)
        self.depth_artifact_dir = os.path.join(
            self.scan_dir, 'depth')
        # self.height_service_name = height_workflow["service_name"]

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
            # Change the format from image/jpeg to img
            if artifact['format'] == 'image/jpeg' or artifact['format'] == 'rgb':
                mod_artifact['format'] = 'img'
            elif artifact['format'] == 'application/zip':
                mod_artifact['format'] = 'depth'

            if mod_artifact['format'] in self.format_wise_artifact:
                self.format_wise_artifact[mod_artifact['format']].append(
                    mod_artifact)
            else:
                self.format_wise_artifact[mod_artifact['format']] = [
                    mod_artifact]

        print("\nPrepared format wise Artifact:\n")
        pprint.pprint(self.format_wise_artifact)

    def create_scan_and_artifact_dir(self):
        '''
        Create a scan dir and format wise dir inside scan dir
        in which all the artifacts will be downloaded
        .
        └── scans
            ├── 3fa85f64-5717-4562-b3fc-2c963f66afa6
            │   └── img
            │       ├── 3fa85f64-5717-4562-b3fc-2c963f6shradul
            │       ├── 3fa85f64-5717-4562-b3fc-2c963fmayank
            │       ├── 69869078-33e1-11eb-af63-cf4006664c92
            │       └── 699b71dc-33e1-11eb-af63-e32a5809de47
            └── 59560ba2-33e1-11eb-af63-4b01606d9610
                └── img
                    ├── 5850e04c-33e1-11eb-af63-4f5622046249
                    └── 5850e04c-33e1-11eb-af63-4f5622046249_blur.jpg
        '''
        if not os.path.isdir(self.scan_dir):
            os.makedirs(self.scan_dir)

        for artifact_format in self.format_wise_artifact:
            if not os.path.exists(
                os.path.join(
                    self.scan_dir,
                    artifact_format)):
                os.makedirs(os.path.join(self.scan_dir, artifact_format))

    def download_blur_flow_artifact(self):
        print("\nDownload Artifacts for Blur Worflow Started")

        self.blur_format_wise_artifact = []

        for i, artifact in enumerate(
                self.format_wise_artifact[self.blur_input_format]):
            mod_artifact = copy.deepcopy(artifact)

            print("\nDownloading Artifact Name: ", mod_artifact["file"], '\n')
            status_code = self.api.get_files(
                mod_artifact["file"], self.blur_workflow_artifact_dir)
            # status_code = get_files_mockup(mod_artifact["file"], format_dir)
            if status_code == 200:
                mod_artifact['download_status'] = True
                self.blur_format_wise_artifact.append(mod_artifact)

        print("\nBelow Artifacts for blur workflow\n")
        print(self.blur_format_wise_artifact)
        print("\nDownload Artifact for completed\n")

    def prepare_blur_result_object(
            self,
            source_artifacts_list,
            blur_id_from_post_request,
            generated_timestamp):
        '''
        Prepare the result object in the results format
        '''
        blur_result = Bunch()
        # Need to clarify
        # blur_result.id = str(uuid.uuid4())
        blur_result.id = f"{uuid.uuid4()}"
        blur_result.scan = self.scan_metadata['id']
        blur_result.workflow = self.blur_workflow["id"]
        blur_result.source_artifacts = source_artifacts_list
        blur_result.source_results = []
        # blur_result.data = Bunch()
        blur_result.file = blur_id_from_post_request
        blur_result.generated = generated_timestamp
        blur_result.meta = []

        res = Bunch()
        res.results = []
        res.results.append(blur_result)
        return res

    def run_blur_flow(self):
        '''
        Run the blur Workflow on the downloaded artifacts
        '''
        for i, artifact in enumerate(self.blur_format_wise_artifact):

            input_path = os.path.join(
                self.blur_workflow_artifact_dir,
                artifact['file'])
            # target_path = input_path + '_blur.jpg'

            print("input_path of image to perform blur: ", input_path, '\n')

            # blur_status = blur_faces_in_file(input_path, target_path)
            blur_img_binary, blur_status = blur_face(input_path)

            if blur_status:
                # Post the blur files
                blur_id_from_post_request, post_status = self.api.post_files(
                    blur_img_binary)
                # blur_id_from_post_request, post_status = self.api.post_files_using_path(target_path)
                # blur_id_from_post_request, post_status = '5850e04c-33e1-11eb-af63-4f5622046249' ,True

                print("Post status of uploading the blur file: ", post_status)

                if post_status == 201:
                    print("\nStarting to post the blur result Json\n")

                    # prepare results
                    source_artifacts_list = [artifact['id']]
                    generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

                    blur_result = self.prepare_blur_result_object(
                        source_artifacts_list, blur_id_from_post_request, generated_timestamp)

                    print("----------------------------------------------")
                    print("Blur Result Json :")

                    blur_result_string = json.dumps(
                        blur_result, indent=2, separators=(',', ':'))
                    blur_result_object = json.loads(blur_result_string)
                    pprint.pprint(blur_result_object)

                    print("----------------------------------------------")

                    # post the blur results using /results
                    if self.api.post_results(blur_result_object) == 201:
                        print(
                            "\nResult posted successfully for Blur Result No. : ", i)

    def get_mean_scan_results(self, predictions):
        return np.mean(predictions)

    def download_depth_artifact(self):
        print("\nDownload Artifacts for height Worflow Started")

        self.depth_artifact = []

        for i, artifact in enumerate(
                self.format_wise_artifact[self.depth_input_format]):
            mod_artifact = copy.deepcopy(artifact)

            print("\nDownloading Artifact Name: ", mod_artifact["file"], '\n')
            status_code = self.api.get_files(
                mod_artifact["file"], self.depth_artifact_dir)
            # status_code = get_files_mockup(mod_artifact["file"], format_dir)
            if status_code == 200:
                mod_artifact['download_status'] = True
                self.depth_artifact.append(mod_artifact)

        print("\nBelow are depth Artifacts\n")
        print(self.depth_artifact)
        print("\nDownload depth Artifact for completed\n")

    def prepare_height_result_object(
            self,
            predictions,
            generated_timestamp):
        '''
        Prepare the result object in the results format
        '''
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.depth_artifact, predictions):
            height_result = Bunch()
            height_result.id = f"{uuid.uuid4()}"
            height_result.scan = self.scan_metadata['id']
            height_result.workflow = self.height_workflow["id"]
            height_result.source_artifacts = [artifact['id']]
            height_result.source_results = []
            height_result.generated = generated_timestamp
            result = {'height': prediction[0]}
            height_result.data = result
            res.results.append(height_result)

        return res

    def prepare_scan_height_result_object(
            self,
            predictions,
            generated_timestamp):
        '''
        Prepare the scan result object in the results format
        '''
        res = Bunch()
        res.results = []
        height_result = Bunch()
        height_result.id = f"{uuid.uuid4()}"
        height_result.scan = self.scan_metadata['id']
        height_result.workflow = self.height_workflow["id"]
        height_result.source_artifacts = [artifact['id'] for artifact in self.depth_artifact]
        height_result.source_results = []
        height_result.generated = generated_timestamp
        mean_prediction = self.get_mean_scan_results(predictions)
        result = {'mean_height': mean_prediction}
        height_result.data = result

        res.results.append(height_result)

        return res

    def run_height_flow(self):
        '''
        Run the height Workflow on the downloaded artifacts
        '''
        depthmaps = []
        for i, artifact in enumerate(self.depth_artifact):
            input_path = os.path.join(
                self.depth_artifact_dir,
                artifact['file'])

            data, width, height, depthScale, maxConfidence = preprocessing.load_depth(input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        height_predictions = inference.get_height_predictions_local(depthmaps)
        print("height predictions are: ", height_predictions)

        height_result = self.prepare_height_result_object(height_predictions.tolist(), generated_timestamp)
        height_result_string = json.dumps(height_result, indent=2, separators=(',', ':'))
        height_result_object = json.loads(height_result_string)
        if self.api.post_results(height_result_object) == 201:
            print("successfully post artifact level height results: ", height_result_object)

        scan_height_result = self.prepare_scan_height_result_object(height_predictions.tolist(), generated_timestamp)
        scan_height_result_string = json.dumps(scan_height_result, indent=2, separators=(',', ':'))
        scan_height_result_object = json.loads(scan_height_result_string)
        if self.api.post_results(scan_height_result_object) == 201:
            print("successfully post scan level height results: ", scan_height_result_object)

    def prepare_weight_result_object(
            self,
            predictions,
            generated_timestamp):
        '''
        Prepare the result object in the results format
        '''
        res = Bunch()
        res.results = []
        for artifact, prediction in zip(self.depth_artifact, predictions):
            weight_result = Bunch()
            weight_result.id = f"{uuid.uuid4()}"
            weight_result.scan = self.scan_metadata['id']
            weight_result.workflow = self.weight_workflow["id"]
            weight_result.source_artifacts = [artifact['id']]
            weight_result.source_results = []
            weight_result.generated = generated_timestamp
            result = {'weight': prediction[0]}
            weight_result.data = result
            res.results.append(weight_result)

        return res

    def prepare_scan_weight_result_object(
            self,
            predictions,
            generated_timestamp):
        '''
        Prepare the scan result object in the results format
        '''
        res = Bunch()
        res.results = []
        weight_result = Bunch()
        weight_result.id = f"{uuid.uuid4()}"
        weight_result.scan = self.scan_metadata['id']
        weight_result.workflow = self.weight_workflow["id"]
        weight_result.source_artifacts = [artifact['id'] for artifact in self.depth_artifact]
        weight_result.source_results = []
        weight_result.generated = generated_timestamp
        mean_prediction = self.get_mean_scan_results(predictions)
        result = {'mean_weight': mean_prediction}
        weight_result.data = result

        res.results.append(weight_result)

        return res

    def run_weight_flow(self):
        '''
        Run the weight Workflow on the downloaded artifacts
        '''
        depthmaps = []
        for i, artifact in enumerate(self.depth_artifact):
            input_path = os.path.join(
                self.depth_artifact_dir,
                artifact['file'])

            data, width, height, depthScale, maxConfidence = preprocessing.load_depth(input_path)
            depthmap, height, width = preprocessing.prepare_depthmap(data, width, height, depthScale)
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)

        depthmaps = np.array(depthmaps)
        generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        weight_predictions = inference.get_weight_predictions_local(depthmaps)
        print("weight predictions are: ", weight_predictions)

        weight_result = self.prepare_weight_result_object(weight_predictions.tolist(), generated_timestamp)
        weight_result_string = json.dumps(weight_result, indent=2, separators=(',', ':'))
        weight_result_object = json.loads(weight_result_string)
        if self.api.post_results(weight_result_object) == 201:
            print("successfully post artifact level weight results: ", weight_result_object)

        scan_weight_result = self.prepare_scan_weight_result_object(weight_predictions.tolist(), generated_timestamp)
        scan_weight_result_string = json.dumps(scan_weight_result, indent=2, separators=(',', ':'))
        scan_weight_result_object = json.loads(scan_weight_result_string)
        if self.api.post_results(scan_weight_result_object) == 201:
            print("successfully post scan level weight results: ", scan_weight_result_object)


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

    parser.add_argument('--height_workflow_path',
                        default="src/workflows/height-worflow.json",
                        type=str,
                        help='Height Workflow path')

    parser.add_argument('--weight_workflow_path',
                        default="/app/src/workflows/weight-worflow.json",
                        type=str,
                        help='Weight Workflow path')

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
    height_workflow_path = args.height_workflow_path
    weight_workflow_path = args.weight_workflow_path

    scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

    cgm_api = ApiEndpoints(
        url,
        scan_endpoint,
        get_file_endpoint,
        post_file_endpoint,
        result_endpoint,
        workflow_endpoint)

    workflows = cgm_api.get_workflows()
    cgm_api.get_scan(scan_metadata_path)

    # scan_metadata_path = './schema/scan_with_blur_artifact.json'
    with open(scan_metadata_path, 'r') as f:
        scan_metadata = f.read()
    scan_metadata_obj = json.loads(scan_metadata)

    if len(scan_metadata_obj['scans']) > 0:

        print(" Starting Result Generation Workflow on a scan")
        # Taking a single scan at a time
        scan_metadata_obj = scan_metadata_obj['scans'][0]

        with open(blur_workflow_path, 'r') as f:
            blur_workflow_obj = json.load(f)

        with open(height_workflow_path, 'r') as f:
            height_workflow_obj = json.load(f)

        with open(weight_workflow_path, 'r') as f:
            weight_workflow_obj = json.load(f)

        blur_workflow_obj['id'] = get_workflow_id(blur_workflow_obj['name'], blur_workflow_obj['version'], workflows)
        height_workflow_obj['id'] = get_workflow_id(height_workflow_obj['name'], height_workflow_obj['version'], workflows)
        weight_workflow_obj['id'] = get_workflow_id(weight_workflow_obj['name'], weight_workflow_obj['version'], workflows)

        scan_results = ScanResults(
            scan_metadata_obj,
            blur_workflow_obj,
            height_workflow_obj,
            weight_workflow_obj,
            scan_parent_dir,
            cgm_api)

        scan_results.process_scan_metadata()
        scan_results.create_scan_and_artifact_dir()
        scan_results.download_blur_flow_artifact()
        scan_results.run_blur_flow()
        scan_results.download_depth_artifact()
        scan_results.run_height_flow()
        scan_results.run_weight_flow()

    else:
        print("No Scan found without Results")


if __name__ == "__main__":
    main()
