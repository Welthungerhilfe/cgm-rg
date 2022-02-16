import copy
import json
import logging
import os
import pprint
import uuid

from bunch import Bunch
import cv2
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ENDPOINTS = Bunch(dict(
    SCANS='/api/scans/unprocessed?limit=1',
    GET_FILES='/api/files/',
    POST_FILES='/api/files?storage=result',
    RESULTS='/api/results',
    WORKFLOWS='/api/workflows',
    PERSONS='/api/persons/',
    MOD_SCAN='/api/scans',
    SCAN_META='/api/scan_metadata/',
))


class ApiEndpoints:
    def __init__(self, url):
        self.url = url
        self.scan_endpoint = ENDPOINTS.SCANS
        self.get_file_endpoint = ENDPOINTS.GET_FILES
        self.post_file_endpoint = ENDPOINTS.POST_FILES
        self.result_endpoint = ENDPOINTS.RESULTS
        self.workflow_endpoint = ENDPOINTS.WORKFLOWS
        self.person_detail_endpoint = ENDPOINTS.PERSONS
        self.mod_scan_endpoint = ENDPOINTS.MOD_SCAN
        self.scan_meta_endpoint = ENDPOINTS.SCAN_META
        self.headers = {}
        self.x_api_key = os.getenv("API_KEY", None)

    def prepare_header(self):
        headers = copy.deepcopy(self.headers)
        if self.x_api_key:
            headers['X-API-Key'] = self.x_api_key
        return headers

    def get_files(self, file_id, save_dir):
        """Get the files from api using file id"""
        endpoint = self.url + self.get_file_endpoint

        headers = self.prepare_header()
        response = requests.get(endpoint + file_id, headers=headers)
        logger.info("%s %s", "Status code:", response.status_code)

        file_path = os.path.join(save_dir, file_id)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        return response.status_code

    def post_files_using_path(self, file_path, type_):
        """Post the files using the path of the file"""
        headers = self.prepare_header()
        headers['content_type'] = 'multipart/form-data'  # status_code 201

        endpoint = self.url + self.post_file_endpoint

        files = {
            'file': (open(file_path, 'rb'), type_),
            'filename': file_path.split('/')[-1],
        }

        logger.info("%s %s", "File name to post :", files['filename'])
        response = requests.post(endpoint, files=files, headers=headers)
        file_id = response.content.decode('utf-8')
        logger.info("%s %s", "File Id from post of test.jpg:", file_id)

        return file_id, response.status_code

    def post_files(self, bin_file):
        """
        Post the file result produced such as blur directly
        without saving it to a location to avoid I/O overhead
        """
        headers = self.prepare_header()
        headers['content_type'] = 'multipart/form-data'  # status_code 201

        endpoint = self.url + self.post_file_endpoint

        _, bin_file = cv2.imencode('.JPEG', bin_file)
        bin_file = bin_file.tostring()

        files = {
            'file': bin_file,
            'filename': 'test.jpg',
        }

        response = requests.post(endpoint, files=files, headers=headers)
        file_id = response.content.decode('utf-8')
        logger.info("%s %s", "File Id from post of test.jpg:", file_id)

        return file_id, response.status_code

    def post_results(self, result_json_obj):
        """Post the result object produced while Result Generation using POST /results"""
        endpoint = self.url + self.result_endpoint
        response = requests.post(endpoint, json=result_json_obj, headers=self.prepare_header())
        logger.info("%s %s", "Status of post result response:", response.status_code)
        return response.status_code

    def post_workflow_and_save_response(self, workflow_obj):
        """Post the workflow and saves the response"""
        logger.info("Workflow Post Object:")
        logger.info(pprint.pformat(workflow_obj))

        headers = self.prepare_header()
        endpoint = self.url + self.workflow_endpoint
        response = requests.post(endpoint, json=workflow_obj, headers=headers)
        logger.info("Workflow Post response")
        logger.info("%s %s", "Status code:", response.status_code)

        if response.status_code in [200, 201]:
            content = response.json()
            logger.info(pprint.pformat(content))

        return response

    def post_workflow(self, workflow_path):
        """Mockup of Post the workflows using POST /files"""
        return str(uuid.uuid4()), 200

    def get_scan(self, scan_path):
        """Get the scan metadata"""
        headers = self.prepare_header()
        response = requests.get(self.url + self.scan_endpoint, headers=headers)

        if response.status_code == 200:
            content = response.json()
            logger.info("Scan Details :")
            logger.info(pprint.pformat(content))

            with open(scan_path, 'w') as f:
                json.dump(content, f, indent=4)

            logger.info("%s %s", "Written scan metadata successfully to", scan_path)
            return len(content['scans'])
        else:
            logger.info("%s %s", "Response code :", response.status_code)
            return 0

    def get_scan_for_scan_version_workflow_id(self, scan_version, workflow_id, scan_path):
        """Get the scan metadata filtered by scan_version and workflow_id"""
        headers = self.prepare_header()
        # use scan_version and workflow id to get filtered scans

        response = requests.get(
            self.url + self.mod_scan_endpoint,
            params={
                'scan_version': scan_version,
                'page': 1,
                'limit': 1,
                'processed': 'false',
                'workflow': workflow_id
            },
            headers=headers)

        if response.status_code == 200:
            content = response.json()
            logger.info("Scan Details :")
            logger.info(pprint.pformat(content))

            with open(scan_path, 'w') as f:
                json.dump(content, f, indent=4)

            logger.info("%s %s", "Written scan metadata successfully to", scan_path)
            return len(content['scans'])
        else:
            logger.info("%s %s", "Response code :", response.status_code)
            return 0

    def get_person_details(self, person_id):
        headers = self.prepare_header()
        response = requests.get(
            self.url + self.person_detail_endpoint + person_id + '/basic',
            headers=headers)

        if response.status_code == 200:
            content = response.json()
            logger.info("Person Details :")
            logger.info(pprint.pprint(content))
        return content

    def get_workflows(self):
        """Get all registered workflows"""
        headers = self.prepare_header()
        response = requests.get(self.url + self.workflow_endpoint, headers=headers)
        return response.json()

    def get_scan_meta(self, scan_id):
        """Get scan meta data from scan id"""
        headers = self.prepare_header()
        response = requests.get(self.url + self.scan_meta_endpoint + scan_id, headers=headers)
        return response.json()
    
    def get_results(self,scan_id,workflow_id):
        """Get Result from scan id and workflow id """
        headers = self.prepare_header()

        response = requests.get(
            self.url + self.mod_scan_endpoint,
            params={
                'workflow': workflow_id,
                'show_results': True,
                'scan_id': scan_id
            },
            headers=headers)
        if response.status_code == 200:
            content = response.json()
            logger.info("Result Details :")
            result = content['scans'][0]['results']
            return result
        else:
            logger.info("%s %s", "Response code :", response.status_code)
            return 0
        


if __name__ == "__main__":
    url = os.getenv('APP_URL', 'http://localhost:5001')
    scan_endpoint = '/api/scans/unprocessed?limit=1'
