import argparse
import copy
import json
import os
import pprint
import uuid

from api_endpoints import ApiEndpoints
from result_generation.blur import BlurFlow
from result_generation.depthmap_image import DepthMapImgFlow
from result_generation.height.height_plaincnn import HeightFlowPlainCnn
from result_generation.height.height_mutiartifact import HeightFlowMultiArtifact
from result_generation.standing import StandingLaying
from result_generation.weight import WeightFlow


class ProcessWorkflows:
    """Process all the workflows"""

    def __init__(self, api: "ApiEndpoints"):
        self.api = api

    def get_list_of_worflows(self):
        """Get the list of workflows from api"""
        self.workflows = self.api.get_workflows()

    def get_workflow_id(self, workflow_name, workflow_version):
        """Get the id of the workflow for given workflow name and version"""
        workflow_obj_with_id = list(
            filter(
                lambda workflow: (
                    workflow['name'] == workflow_name and workflow['version'] == workflow_version),
                self.workflows['workflows']))[0]
        return workflow_obj_with_id['id']

    def load_workflows(self, workflow_path):
        """Load the workflow from given path"""
        with open(workflow_path, 'r') as f:
            workflow_obj = json.load(f)

        return workflow_obj


class GetScanMetadata:
    """Get and process scan metadata."""

    def __init__(self, api: "ApiEndpoints", scan_metadata_path: str):
        """Construct all the necessary attributes for the GetScanMetadata object.

        Parameters:
            api
            scan_metadata_path: path to store scan metadata
        """
        self.api = api
        self.scan_metadata_path = scan_metadata_path

    def get_unprocessed_scans(self):
        """Get unprocessed_scans from api and returns the no of scans

        Returns
            the no of scans in scan metadata.
        """

        return self.api.get_scan(self.scan_metadata_path)

    def get_scan_metadata(self):
        with open(self.scan_metadata_path, 'r') as f:
            scan_metadata_obj = json.load(f)
        scan_metadata = scan_metadata_obj['scans'][0]

        return scan_metadata


class PrepareArtifacts:
    """Prepare artifacts for result generation.

    Attributes
        scan_metadata : json
            metadata of the scan to run weight flow on
        scan_parent_dir : str
            directory where scans are stored
    """

    def __init__(self, api: "ApiEndpoints", scan_metadata, scan_parent_dir):
        self.api = api
        self.scan_metadata = scan_metadata
        self.format_wise_artifact = {}
        self.scan_parent_dir = scan_parent_dir
        self.scan_dir = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'])

    def download_artifacts(self, input_format):
        """Download artifacts for the scan"""
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
        """Check the format of the artifact"""
        if format in ['image/jpeg', 'rgb']:
            return 'img'
        elif format in ['application/zip', 'depth']:
            return 'depth'

    def add_artifacts_to_format_dictionary(self, format, artifact):
        """Sort artifacts according to input format"""
        if format in self.format_wise_artifact:
            self.format_wise_artifact[format].append(artifact)
        else:
            self.format_wise_artifact[format] = [artifact]

    def process_scan_metadata(self):
        """Process artifacts in a scan.

        Process the scan object to get the list of jpeg id
        and artifact id return a dict of format as key and
        list of file id as values
        """
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
        '''Create directory to store artifacts in scan.

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
        """Create directory to store downloaded artifacts"""
        for artifact_format in self.format_wise_artifact:
            os.makedirs(
                os.path.join(self.scan_dir,artifact_format),
                exist_ok=True)


def person(api, person_id):
    return api.get_person_details(person_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_parent_dir', default="data/scans/", help='Parent directory in which scans will be stored')  # noqa: E501
    parser.add_argument('--blur_workflow_path', default="src/workflows/blur-workflow.json")  # noqa: E501
    parser.add_argument('--standing_laying_workflow_path', default="src/workflows/standing_laying-workflow.json")  # noqa: E501
    parser.add_argument('--depthmap_img_workflow_path', default="src/workflows/depthmap-img-workflow.json")  # noqa: E501
    parser.add_argument('--height_workflow_artifact_path', default="src/workflows/height-plaincnn-workflow-artifact.json", help='Height Workflow Artifact path')  # noqa: E501
    parser.add_argument('--height_depthmapmultiartifactlatefusion_workflow_path', default="src/workflows/height-depthmapmultiartifactlatefusion-workflow.json")  # noqa: E501
    parser.add_argument('--height_workflow_scan_path', default="src/workflows/height-plaincnn-workflow-scan.json")  # noqa: E501
    parser.add_argument('--weight_workflow_artifact_path', default="src/workflows/weight-workflow-artifact.json")  # noqa: E501
    parser.add_argument('--weight_workflow_scan_path', default="src/workflows/weight-workflow-scan.json")  # noqa: E501
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    scan_parent_dir = args.scan_parent_dir
    blur_workflow_path = args.blur_workflow_path
    standing_laying_workflow_path = args.standing_laying_workflow_path
    depthmap_img_workflow_path = args.depthmap_img_workflow_path
    height_workflow_artifact_path = args.height_workflow_artifact_path
    height_workflow_scan_path = args.height_workflow_scan_path
    height_depthmapmultiartifactlatefusion_workflow_path = args.height_depthmapmultiartifactlatefusion_workflow_path
    weight_workflow_artifact_path = args.weight_workflow_artifact_path
    weight_workflow_scan_path = args.weight_workflow_scan_path

    scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

    # URL
    url = os.getenv('APP_URL', 'http://localhost:5001')
    print(f"App URL: {url}")
    cgm_api = ApiEndpoints(url)

    workflow = ProcessWorkflows(cgm_api)

    get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)

    if get_scan_metadata.get_unprocessed_scans() <= 0:
        return
    scan_metadata = get_scan_metadata.get_scan_metadata()
    scan_version = scan_metadata['version']
    print("Scan Type Version: ", scan_version)
    workflow.get_list_of_worflows()
    data_processing = PrepareArtifacts(cgm_api, scan_metadata, scan_parent_dir)
    data_processing.process_scan_metadata()
    data_processing.create_scan_dir()
    data_processing.create_artifact_dir()
    rgb_artifacts = data_processing.download_artifacts('img')
    depth_artifacts = data_processing.download_artifacts('depth')
    person_details = person(cgm_api, scan_metadata['person'])

    flows = []

    flow = BlurFlow(
        cgm_api,
        workflow,
        blur_workflow_path,
        rgb_artifacts,
        scan_parent_dir,
        scan_metadata,
        scan_version)
    flows.append(flow)

    flow = StandingLaying(
        cgm_api,
        workflow,
        standing_laying_workflow_path,
        rgb_artifacts,
        scan_parent_dir,
        scan_metadata)
    flows.append(flow)

    flow = DepthMapImgFlow(
        cgm_api,
        workflow,
        depthmap_img_workflow_path,
        depth_artifacts,
        scan_parent_dir,
        scan_metadata)
    flows.append(flow)

    flow = HeightFlowPlainCnn(
        cgm_api,
        workflow,
        height_workflow_artifact_path,
        height_workflow_scan_path,
        depth_artifacts,
        scan_parent_dir,
        scan_metadata,
        person_details)
    flows.append(flow)

    flow = HeightFlowMultiArtifact(
        cgm_api,
        workflow,
        height_workflow_artifact_path,
        height_depthmapmultiartifactlatefusion_workflow_path,
        depth_artifacts,
        scan_parent_dir,
        scan_metadata,
        person_details)
    flows.append(flow)

    flow = WeightFlow(
        cgm_api,
        workflow,
        weight_workflow_artifact_path,
        weight_workflow_scan_path,
        depth_artifacts,
        scan_parent_dir,
        scan_metadata,
        person_details)
    flows.append(flow)

    for flow in flows:
        try:
            flow.run_flow()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
