import argparse
import copy
import json
import os
import pprint
import uuid

from api_endpoints import ApiEndpoints
from result_generation.blur import BlurFlow
from result_generation.result_generation import ResultGeneration
from result_generation.depthmap_image import DepthMapImgFlow
from result_generation.height.height_plaincnn import HeightFlowPlainCnn
from result_generation.height.height_mutiartifact import HeightFlowMultiArtifact
from result_generation.height.height_ensemble import HeightFlowDeepEnsemble
from result_generation.standing import StandingLaying
from result_generation.weight import WeightFlow
from result_generation.height.height_rgbd import HeightFlowRGBD


class ProcessWorkflows:
    """Process all the workflows"""

    def __init__(self, api: ApiEndpoints):
        self.api = api

    def get_list_of_worflows(self):
        """Get the list of workflows from api"""
        self.workflows = self.api.get_workflows()

    def get_workflow_id(self, workflow_name, workflow_version):
        """Get the id of the workflow for given workflow name and version"""
        workflow_obj_with_id = list(
            filter(lambda workflow: (
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

    def __init__(self, api: ApiEndpoints, scan_metadata_path: str):
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

    def get_unprocessed_scans_for_scan_version_workflow_id(self, scan_version, workflow_id, scan_metadata_path):
        """
        Gets unprocessed_scans from api filtered by scan verion type and workflow id and returns the no of scans

        Parameters
        ----------
        scan_version : Scan Version of unprocessed scan
        workflow_id : Workflow id of unprocessed scan
        scan_metadata_path : Path to store the scan metadata

        Returns
        -------
        Length of the unprocessed scan filtered by scan verion type and workflow id
        """

        return self.api.get_scan_for_scan_version_workflow_id(
            scan_version, workflow_id, scan_metadata_path)

    def get_scan_metadata(self):
        with open(self.scan_metadata_path, 'r') as f:
            scan_metadata_obj = json.load(f)
        scan_metadata = scan_metadata_obj['scans'][0]

        return scan_metadata

    def get_scan_metadata_by_path(self, scan_metadata_path):
        with open(scan_metadata_path, 'r') as f:
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

    def __init__(self, api: ApiEndpoints, scan_metadata, scan_parent_dir):
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
            os.makedirs(os.path.join(self.scan_dir, artifact_format), exist_ok=True)


def person(api, person_id):
    return api.get_person_details(person_id)


def parse_args():
    parser = argparse.ArgumentParser()
    workflow_dir = 'src/workflows'
    parser.add_argument('--scan_parent_dir', default="data/scans/", help='Parent directory in which scans will be stored')  # noqa: E501
    parser.add_argument('--blur_workflow_path', default=f"{workflow_dir}/blur-workflow.json")  # noqa: E501
    parser.add_argument('--standing_laying_workflow_path', default=f"{workflow_dir}/standing_laying-workflow.json")  # noqa: E501
    parser.add_argument('--depthmap_img_workflow_path', default=f"{workflow_dir}/depthmap-img-workflow.json")  # noqa: E501
    parser.add_argument('--height_workflow_artifact_path', default=f"{workflow_dir}/height-plaincnn-workflow-artifact.json", help='Height Workflow Artifact path')  # noqa: E501
    parser.add_argument('--height_depthmapmultiartifactlatefusion_workflow_path', default=f"{workflow_dir}/height-depthmapmultiartifactlatefusion-workflow.json")  # noqa: E501
    parser.add_argument('--height_workflow_scan_path', default=f"{workflow_dir}/height-plaincnn-workflow-scan.json")  # noqa: E501
    parser.add_argument('--height_ensemble_workflow_artifact_path', default="/app/src/workflows/height-ensemble-workflow-artifact.json")  # noqa: E501
    parser.add_argument('--height_ensemble_workflow_scan_path', default="/app/src/workflows/height-ensemble-workflow-scan.json")  # noqa: E501
    parser.add_argument('--weight_workflow_artifact_path', default=f"{workflow_dir}/weight-workflow-artifact.json")  # noqa: E501
    parser.add_argument('--weight_workflow_scan_path', default=f"{workflow_dir}/weight-workflow-scan.json")  # noqa: E501
    parser.add_argument('--height_rgbd_workflow_artifact_path', default=f"{workflow_dir}/height-rgbd-workflow-artifact.json")  # noqa: E501
    parser.add_argument('--height_rgbd_workflow_scan_path', default=f"{workflow_dir}/height-rgbd-workflow-scan.json")  # noqa: E501
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
    height_ensemble_workflow_artifact_path = args.height_ensemble_workflow_artifact_path
    height_ensemble_workflow_scan_path = args.height_ensemble_workflow_scan_path
    height_depthmapmultiartifactlatefusion_workflow_path = args.height_depthmapmultiartifactlatefusion_workflow_path
    weight_workflow_artifact_path = args.weight_workflow_artifact_path
    weight_workflow_scan_path = args.weight_workflow_scan_path
    height_rgbd_workflow_artifact_path = args.height_rgbd_workflow_artifact_path
    height_rgbd_workflow_scan_path = args.height_rgbd_workflow_scan_path

    scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

    # URL
    url = os.getenv('APP_URL', 'http://localhost:5001')
    print(f"App URL: {url}")
    cgm_api = ApiEndpoints(url)

    workflow = ProcessWorkflows(cgm_api)
    get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)

    # logic to initiate rgbd workflow for v0.9 starts here

    workflow.get_list_of_worflows()
    filterby_workflow_metadata = workflow.load_workflows(
        height_rgbd_workflow_scan_path)
    filterby_scan_version_val = 'v0.9'

    filterby_workflow_name = filterby_workflow_metadata['name']
    filterby_workflow_version = filterby_workflow_metadata['version']
    print("Filter by workflow Name: ", filterby_workflow_name)
    print("Filter by workflow Version: ", filterby_workflow_version)
    filterby_workflow_id_val = workflow.get_workflow_id(
        filterby_workflow_name, filterby_workflow_version)
    filterby_scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    filterby_scan_metadata_path = os.path.join(
        scan_parent_dir, filterby_scan_metadata_name)
    # Start cgm-rg for scan filtered by scan version and workflow id

    if get_scan_metadata.get_unprocessed_scans_for_scan_version_workflow_id(
            filterby_scan_version_val,
            filterby_workflow_id_val,
            filterby_scan_metadata_path) > 0:

        print('-------------------------------------------------------------------------------------------')
        print(
            "Started cgm-rg for scan filtered by ",
            filterby_scan_version_val,
            " and ",
            filterby_workflow_id_val)
        scan_metadata = get_scan_metadata.get_scan_metadata_by_path(
            filterby_scan_metadata_path)
        scan_version = scan_metadata['version']
        print("Scan Version: ", scan_version)
        print("Filterby Scan Version: ", filterby_scan_version_val)
        try:
            assert (scan_version == filterby_scan_version_val)
            data_processing = PrepareArtifacts(
                cgm_api, scan_metadata, scan_parent_dir)
            data_processing.process_scan_metadata()
            data_processing.create_scan_dir()
            data_processing.create_artifact_dir()
            rgb_artifacts = data_processing.download_artifacts('img')
            depth_artifacts = data_processing.download_artifacts('depth')
            person_details = person(cgm_api, scan_metadata['person'])

            result_generation = ResultGeneration(cgm_api, workflow, scan_metadata, scan_parent_dir)

            rgbdflow = HeightFlowRGBD(
                result_generation,
                height_rgbd_workflow_artifact_path,
                height_rgbd_workflow_scan_path,
                depth_artifacts,
                person_details,
                rgb_artifacts)

            try:
                rgbdflow.run_flow()
            except Exception as e:
                print('---------------------------------')
                print(e)
                print("RGBD Flow is not defined")

        except Exception as e:
            print(e)
            print("Scan Version does not match")

    # logic to initiate rgbd workflow for v0.9 ends here

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

    result_generation = ResultGeneration(cgm_api, workflow, scan_metadata, scan_parent_dir)

    flow = BlurFlow(
        result_generation,
        blur_workflow_path,
        rgb_artifacts,
        scan_version)
    flows.append(flow)

    flow = StandingLaying(
        result_generation,
        standing_laying_workflow_path,
        rgb_artifacts)
    flows.append(flow)

    flow = DepthMapImgFlow(
        result_generation,
        depthmap_img_workflow_path,
        depth_artifacts)
    flows.append(flow)

    flow = HeightFlowPlainCnn(
        result_generation,
        height_workflow_artifact_path,
        height_workflow_scan_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    flow = HeightFlowMultiArtifact(
        result_generation,
        height_workflow_artifact_path,
        height_depthmapmultiartifactlatefusion_workflow_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    flow = HeightFlowDeepEnsemble(
        result_generation,
        height_ensemble_workflow_artifact_path,
        height_ensemble_workflow_scan_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    flow = WeightFlow(
        result_generation,
        weight_workflow_artifact_path,
        weight_workflow_scan_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    if scan_version in ['v0.9']:  # TODO update this with better logic
        flow = HeightFlowRGBD(
            result_generation,
            height_rgbd_workflow_artifact_path,
            height_rgbd_workflow_scan_path,
            depth_artifacts,
            person_details,
            rgb_artifacts)
        flows.append(flow)

    for flow in flows:
        try:
            flow.run_flow()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
