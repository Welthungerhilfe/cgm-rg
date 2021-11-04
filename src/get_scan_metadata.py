import json

from api_endpoints import ApiManager


class MetadataManager:
    """Get and process scan metadata"""

    def __init__(self, scan_metadata_path: str):
        self.scan_metadata_path = scan_metadata_path

    def get_unprocessed_scans(self, api_manager: ApiManager) -> int:
        """Get unprocessed_scans from api and returns the no of scans

        Returns
            the no of scans in scan metadata.
        """

        return api_manager.get_number_of_scans(self.scan_metadata_path)

    def get_unprocessed_scans_for_scan_version_workflow_id(self, api_manager, scan_version, workflow_id, scan_metadata_path):  # TODO unused
        """
        Gets unprocessed_scans from api filtered by scan version type and workflow id and returns the no of scans

        Parameters
        ----------
        scan_version : Scan Version of unprocessed scan
        workflow_id : Workflow id of unprocessed scan
        scan_metadata_path : Path to store the scan metadata

        Returns
        -------
        Length of the unprocessed scan filtered by scan version type and workflow id
        """

        return api_manager.get_scan_for_scan_version_workflow_id(scan_version, workflow_id, scan_metadata_path)

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
