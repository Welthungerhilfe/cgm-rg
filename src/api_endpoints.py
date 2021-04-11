import copy
import json
import os
import pprint
import uuid

import cv2
import requests


class ApiEndpoints:
    """
    A interface to interact with API endpoints.

    Args:
        url (str): url of the API.
        scan_endpoint (str): endpoint to get unprocessed scan.
        get_file_endpoint (str): endpoint to get a file.
        post_file_endpoint (str): endpoint to post a file.
        result_endpoint (str): endpoint to post the generated results.
        workflow_endpoint (str): endpoint to get list of registered workflows.
        person_detail_endpoint (str): endpoint to get details of the person.
    """
    def __init__(
            self,
            url,
            scan_endpoint,
            get_file_endpoint,
            post_file_endpoint,
            result_endpoint,
            workflow_endpoint,
            person_detail_endpoint):
        self.url = url
        self.scan_endpoint = scan_endpoint
        self.get_file_endpoint = get_file_endpoint
        self.post_file_endpoint = post_file_endpoint
        self.result_endpoint = result_endpoint
        self.workflow_endpoint = workflow_endpoint
        self.person_detail_endpoint = person_detail_endpoint
        self.headers = {}
        self.x_api_key = os.getenv("API_KEY", None)

    def prepare_header(self):
        """
        Prepares header by adding required authentication key.

        Returns:
            Returns the header dict with required authentication key.
        """
        headers = copy.deepcopy(self.headers)

        if self.x_api_key:
            headers['X-API-Key'] = self.x_api_key

        return headers

    def get_files(self, file_id, save_dir):
        '''
        Gets file from api for given file id and stores it in given save directory.

        Args:
            file_id (str): unique id of the file required from the API.
            save_dir (str): the directory where the file needs to be stored after getting it from API.
        
        Returns:
            status code of the request from the API.
        '''
        endpoint = self.url + self.get_file_endpoint

        headers = self.prepare_header()
        response = requests.get(endpoint + file_id, headers=headers)
        print("\nStatus code: ", response.status_code)

        file_path = os.path.join(save_dir, file_id)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        return response.status_code

    def post_files_using_path(self, file_path, type_):
        '''
        Post the file in given file_path.

        Args:
            file_path (str): path of the file to be posted to API.
            type_ (str): type of the file.
        
        Returns:
            unique id of the file returned by the API and the status code of the post request.
        '''
        headers = self.prepare_header()
        headers['content_type'] = 'multipart/form-data'  # status_code 201
        # headers['content-type'] = 'multipart/form-data'  # status_code 400
        # headers['Content-Type'] = 'multipart/form-data'   # status_code 400

        endpoint = self.url + self.post_file_endpoint

        files = {}
        files['file'] = (open(file_path, 'rb'), type_)
        files['filename'] = file_path.split('/')[-1]

        print('\nFile name to post : ', files['filename'])

        response = requests.post(endpoint, files=files, headers=headers)
        file_id = response.content.decode('utf-8')

        print("\nFile Id from post of test.jpg: ", file_id)

        return file_id, response.status_code

    def post_files(self, bin_file):
        '''
        Post the given binary file to API.

        Advantages: No need to save file, reduces I/O overhead.

        Args:
            bin_file (bytes): bytes array of the file to be posted to API.

        Returns:
            unique id of the file returned by the API and the status code of the post request.

        '''
        headers = self.prepare_header()
        headers['content_type'] = 'multipart/form-data'    # status_code 201
        # headers['content-type'] = 'multipart/form-data'     # status_code 400
        # headers['Content-Type'] = 'multipart/form-data'     # status_code 400

        endpoint = self.url + self.post_file_endpoint

        _, bin_file = cv2.imencode('.JPEG', bin_file)
        # _, bin_file = cv2.imencode('.PNG', bin_file)
        bin_file = bin_file.tostring()

        files = {
            'file': bin_file,
            'filename': 'test.jpg'
        }
        '''
        files = {
            'file': bin_file,
            'filename': 'test.PNG'
        }
        '''

        response = requests.post(endpoint, files=files, headers=headers)
        file_id = response.content.decode('utf-8')

        # print("File Id from post of test.jpg: ", file_id)
        print("File Id from post of test.PNG: ", file_id)

        return file_id, response.status_code

    def post_results(self, result_json_obj):
        '''
        Post the given result object produced by Result Generation.

        Args:
            result_json_obj (dict): dictionary with results to be posted.
        
        Returns:
            status code of the request to the API.
        '''
        headers = self.prepare_header()
        endpoint = self.url + self.result_endpoint

        response = requests.post(
            endpoint,
            json=result_json_obj,
            headers=headers)

        print("Status of post result response: ", response.status_code)

        return response.status_code

    def post_workflow_and_save_response(self, workflow_obj):
        '''
        Post the given workflow.

        Args:
            workflow_obj (dict): Workflow object to be posted to the API.

        Returns:
            response from the API to the post request.
        '''
        print("Workflow Post Object: ")
        pprint.pprint(workflow_obj)

        headers = self.prepare_header()
        endpoint = self.url + self.workflow_endpoint
        response = requests.post(endpoint, json=workflow_obj, headers=headers)
        print("Workflow Post response")
        print("Status code: ", response.status_code)

        if response.status_code in [201, 200]:
            content = response.json()
            pprint.pprint(content)

        return response

    def get_scan(self, scan_path):
        '''
        Get the unprocessed scan metadata from the API and store it in given file path.

        Args:
            scan_path (str): file path where the scan metadata will be saved as json file.

        Returns:
            no of scans in received scan metadata.
        '''
        headers = self.prepare_header()
        response = requests.get(self.url + self.scan_endpoint, headers=headers)

        if response.status_code == 200:
            content = response.json()
            print("\nScan Details :")
            pprint.pprint(content)

            with open(scan_path, 'w') as f:
                json.dump(content, f, indent=4)

            print("Written scan metadata successfully to ", scan_path)
            return len(content['scans'])
        else:
            print("Response code : ", response.status_code)
            return 0

    def get_person_details(self, person_id):
        headers = self.prepare_header()
        resposne = requests.get(
            self.url + self.person_detail_endpoint + person_id + '/basic', headers=headers)

        if resposne.status_code == 200:
            content = resposne.json()
            print("\n Person Details :")
            pprint.pprint(content)
        return content

    def get_workflows(self):
        '''
        Get all registerd workflows from API.

        Returns:
            json object with list of all the workflows.
        '''
        headers = self.prepare_header()
        response = requests.get(
            self.url + self.workflow_endpoint, headers=headers)
        return response.json()


if __name__ == "__main__":
    url = os.getenv('APP_URL', 'http://localhost:5001')
    scan_endpoint = '/api/scans/unprocessed?limit=1'
