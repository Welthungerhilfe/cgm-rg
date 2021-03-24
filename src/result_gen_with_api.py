import argparse
import copy
import json
import os
import pprint
import uuid

import utils.preprocessing as preprocessing
from api_endpoints import ApiEndpoints
from result_generation.blur import BlurFlow
from result_generation.depthmap_image import DepthMapImgFlow
from result_generation.height import HeightFlow
from result_generation.standing import Standing_laying
from result_generation.weight import WeightFlow


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
        workflow_obj_with_id = list(
            filter(
                lambda workflow: (
                    workflow['name'] == workflow_name and workflow['version'] == workflow_version),
                self.workflows['workflows']))[0]

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
            status_code = self.api.get_files(
                mod_artifact["file"], os.path.join(
                    self.scan_dir, input_format))
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

            mod_artifact['format'] = self.check_artifact_format(
                artifact['format'])

            self.add_artifacts_to_format_dictionary(
                mod_artifact['format'], mod_artifact)

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
        os.makedirs(self.scan_dir, exist_ok=True)

    def create_artifact_dir(self):
        for artifact_format in self.format_wise_artifact:
            os.makedirs(
                os.path.join(
                    self.scan_dir,
                    artifact_format),
                exist_ok=True)


def person(api, person_id):
    return api.get_person_details(person_id)


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
                        default="src/workflows/blur-workflow.json",
                        type=str,
                        help='Blur Workflow path')
    parser.add_argument('--standing_laying_workflow_path',
                        default="src/workflows/standing_laying-workflow.json",
                        type=str,
                        help='Standing laying Workflow path')

    parser.add_argument('--depthmap_img_workflow_path',
                        default="src/workflows/depthmap-img-workflow.json",
                        type=str,
                        help='Depthmap Image Workflow path')

    parser.add_argument('--height_workflow_artifact_path',
                        default="src/workflows/height-workflow-artifact.json",
                        type=str,
                        help='Height Workflow Artifact path')

    parser.add_argument('--height_MCNN_workflow_path',
                        default="src/workflows/height-MCNN-workflow.json",
                        type=str,
                        help='Height Workflow MCNN Artifact path')

    parser.add_argument('--height_workflow_scan_path',
                        default="src/workflows/height-workflow-scan.json",
                        type=str,
                        help='Height Workflow Scan path')

    parser.add_argument('--weight_workflow_artifact_path',
                        default="src/workflows/weight-workflow-artifact.json",
                        type=str,
                        help='Weight Workflow Artifact path')

    parser.add_argument('--weight_workflow_scan_path',
                        default="src/workflows/weight-workflow-scan.json",
                        type=str,
                        help='Weight Workflow Scan path')

    args = parser.parse_args()

    # preprocessing.set_width(int(240 * 0.75))
    # preprocessing.set_height(int(180 * 0.75))

    preprocessing.set_width(int(240))
    preprocessing.set_height(int(180))

    print("\nApp Environment : ", os.environ['APP_ENV'])

    if os.environ['APP_ENV'] == 'LOCAL':
        url = "http://localhost:5001"
    elif os.environ['APP_ENV'] == 'SANDBOX':
        url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
    elif os.environ['APP_ENV'] == 'DEMO':
        url = "https://cgm-be-ci-qa-scanner-api.azurewebsites.net"
    elif os.environ['APP_ENV'] == 'INBMZ':
        url = "https://cgm-be-ci-inbmz-scanner-api.azurewebsites.net"

    scan_endpoint = '/api/scans/unprocessed?limit=1'
    get_file_endpoint = '/api/files/'
    post_file_endpoint = '/api/files'
    result_endpoint = '/api/results'
    workflow_endpoint = '/api/workflows'
    person_detail_endpoint = '/api/persons/'

    scan_parent_dir = args.scan_parent_dir
    blur_workflow_path = args.blur_workflow_path
    standing_laying_workflow_path = args.standing_laying_workflow_path
    depthmap_img_workflow_path = args.depthmap_img_workflow_path
    height_workflow_artifact_path = args.height_workflow_artifact_path
    height_workflow_scan_path = args.height_workflow_scan_path
    height_MCNN_workflow_path = args.height_MCNN_workflow_path
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
        workflow_endpoint,
        person_detail_endpoint)

    workflow = ProcessWorkflows(cgm_api)

    get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)

    if get_scan_metadata.get_unprocessed_scans() > 0:
        scan_metadata = get_scan_metadata.get_scan_metadata()
        workflow.get_list_of_worflows()
        data_processing = PrepareArtifacts(
            cgm_api, scan_metadata, scan_parent_dir)
        data_processing.process_scan_metadata()
        data_processing.create_scan_dir()
        data_processing.create_artifact_dir()
        rgb_artifacts = data_processing.download_artifacts('img')
        depth_artifacts = data_processing.download_artifacts('depth')
        person_details = person(cgm_api, scan_metadata['person'])

        blurflow = BlurFlow(
            cgm_api,
            workflow,
            blur_workflow_path,
            rgb_artifacts,
            scan_parent_dir,
            scan_metadata)
        standing_laying = Standing_laying(
            cgm_api,
            workflow,
            standing_laying_workflow_path,
            rgb_artifacts,
            scan_parent_dir,
            scan_metadata)
        depthmap_img_flow = DepthMapImgFlow(
            cgm_api,
            workflow,
            depthmap_img_workflow_path,
            depth_artifacts,
            scan_parent_dir,
            scan_metadata)
        heightflow = HeightFlow(
            cgm_api,
            workflow,
            height_workflow_artifact_path,
            height_workflow_scan_path,
            height_MCNN_workflow_path,
            depth_artifacts,
            scan_parent_dir,
            scan_metadata,
            person_details)
        weightflow = WeightFlow(
            cgm_api,
            workflow,
            weight_workflow_artifact_path,
            weight_workflow_scan_path,
            depth_artifacts,
            scan_parent_dir,
            scan_metadata,
            person_details)

        try:
            blurflow.run_blur_flow()
        except Exception as e:
            print(e)

        try:
            depthmap_img_flow.run_depthmap_img_flow()
        except Exception as e:
            print(e)

        try:
            standing_laying.run_standing_laying_flow()
        except Exception as e:
            print(e)

        try:
            heightflow.run_height_flow()
        except Exception as e:
            print(e)

        try:
            heightflow.run_height_flow_MCNN()
        except Exception as e:
            print(e)

        try:
            weightflow.run_weight_flow()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
