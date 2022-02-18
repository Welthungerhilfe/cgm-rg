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
        self.x_api_key = os.getenv("API_KEY_ERROR_STATS", '')

    def prepare_header(self):
        headers = copy.deepcopy(self.headers)
        if self.x_api_key:
            headers['X-API-Key'] = self.x_api_key
        return headers

    def get_percentile_from_error_stats(
            self,
            age,
            scan_type,
            scan_version,
            workflow_name,
            workflow_version,
            percentile_value,
            standing_laying):
        """Get the scan metadata filtered by scan_version and workflow_id"""
        headers = self.prepare_header()
        # use scan_version and workflow id to get filtered scans
        if standing_laying is not None:
            params = {'age': age,
                      'scan_type': scan_type,
                      'scan_version': scan_version,
                      'workflow_name': workflow_name,
                      'workflow_ver': workflow_version,
                      'percentile_value': percentile_value,
                      'standing_laying': standing_laying,
                      }
        else:
            params = {'age': age,
                      'scan_type': scan_type,
                      'scan_version': scan_version,
                      'workflow_name': workflow_name,
                      'workflow_ver': workflow_version,
                      'percentile_value': percentile_value,
                      }

        response = requests.get(
            self.url + self.percentile_error_endpoints,
            params=params,
            headers=headers)

        if response.status_code == 200:
            error_stats = response.json()
        else:
            error_stats = {}
            logger.info("Sending empty error stats due to below enconutered error:")
            logger.error("%s %s", "Status code for response is", response.status_code)
            logger.error("%s %s", "Respone of error stats api", response.json())
            logger.info("%s %s", "Percentile Endpoint", self.url + self.percentile_error_endpoints)
            logger.info("%s %s", "params", params)

        return error_stats


if __name__ == "__main__":
    url = os.getenv('APP_URL_ERROR_STATS', 'http://localhost:5002')
