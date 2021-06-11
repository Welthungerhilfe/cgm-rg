import json
from api_endpoints import ApiEndpoints


class GetScanMetadata:
    """Get and process scan metadata"""

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

        return self.api.get_scan_for_scan_version_workflow_id(scan_version, workflow_id, scan_metadata_path)

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
