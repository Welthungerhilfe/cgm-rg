import copy
import logging
import os
import pprint
from api_endpoints import ApiEndpoints


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        self.scan_dir = os.path.join(self.scan_parent_dir, self.scan_metadata['id'])
        logger.info("%s %s", "Parent Scan Dir:", self.scan_dir)

    def download_artifacts(self, input_format):
        """Download artifacts for the scan"""
        logger.info("%s %s %s", "Downloading Artifacts for", input_format, "format")
        self.artifacts = []

        for i, artifact in enumerate(self.format_wise_artifact[input_format]):
            mod_artifact = copy.deepcopy(artifact)

            logger.info("%s %s", "Downloading Artifact Name:", mod_artifact["file"])
            status_code = self.api.get_files(mod_artifact["file"], os.path.join(self.scan_dir, input_format))
            # status_code = get_files_mockup(mod_artifact["file"], format_dir)
            if status_code == 200:
                mod_artifact['download_status'] = True
                self.artifacts.append(mod_artifact)

        logger.info("%s %s %s", "Below Artifacts for", input_format, "workflow")
        logger.info(self.artifacts)
        logger.info("Download Artifact for completed")

        return self.artifacts

    def check_artifact_format(self, format):
        """Check the format of the artifact"""
        if format in ['image/jpeg', 'rgb']:
            return 'img'
        elif format in ['application/zip', 'depth']:
            return 'depth'
        else:
            raise NameError(f"Unknown format {format}")

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

        logger.info("Prepared format wise Artifact:")
        logger.info(pprint.pformat(self.format_wise_artifact))

    def create_scan_dir(self):
        """Create directory to store artifacts in scan.

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
        """
        os.makedirs(self.scan_dir, exist_ok=True)

    def create_artifact_dir(self):
        """Create directory to store downloaded artifacts"""
        for artifact_format in self.format_wise_artifact:
            os.makedirs(os.path.join(self.scan_dir, artifact_format), exist_ok=True)
