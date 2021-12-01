import copy
import logging
import os

from bunch import Bunch
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ENDPOINTS = Bunch(dict(
    PERCENTILE_ERROR='api/percentile_errors',
))


class ErrorStatsEndpointsManager:
    def __init__(self, url):
        self.url = url
        self.percentile_error_endpoints = ENDPOINTS.PERCENTILE_ERROR
        self.headers = {}
        self.x_api_key = os.getenv("API_KEY_ERROR_STATS", 'z:6.VQ;j]t>z}JRjSuxdmT-n~j7K2NRZ')

    def prepare_header(self):
        headers = copy.deepcopy(self.headers)
        if self.x_api_key:
            headers['X-API-Key'] = self.x_api_key
        return headers

    def get_percentile_from_error_stats(self, age, scan_type, scan_version, workflow_name, workflow_version, percentile_value):
        """Get the scan metadata filtered by scan_version and workflow_id"""
        headers = self.prepare_header()
        # use scan_version and workflow id to get filtered scans

        response = requests.get(
            self.url + self.percentile_error_endpoints,
            params={
                'age': age,
                'scan_type': scan_type,
                'scan_version': scan_version,
                'workflow_name': workflow_name,
                'workflow_ver': workflow_version,
                'percentile_value': percentile_value,
            },
            headers=headers)

        return response.json()


if __name__ == "__main__":
    url = os.getenv('APP_URL_ERROR_STATS', 'http://localhost:5002')
