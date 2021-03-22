import copy
import json
import os
import pprint
import uuid

import cv2
import requests


class ApiEndpoints:
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
        self.x_api_key = os.environ["API_KEY"]

    def prepare_header(self):
        headers = copy.deepcopy(self.headers)

        if self.x_api_key:
            headers['X-API-Key'] = self.x_api_key

        return headers

    def get_files(self, file_id, save_dir):
        '''
        Get the files from api using file id
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
        Post the files using the path of the file
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
        Post the file result produced such as blur directly
        without saving it to a location to avoid I/O overhead
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
        Post the result object produced while Result Generation
        using POST /results
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
        Post the workflow and saves the response
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
            # content['data'] = workflow_obj["data"]
            pprint.pprint(content)

            # with open(response_path, 'w') as f:
            #     json.dump(content, f)

        return response

    def post_workflow(self, workflow_path):
        '''
        Mockup of Post the workflows using POST /files
        '''
        return str(uuid.uuid4()), 200

    def get_scan(self, scan_path):
        '''
        Get the scan metadata
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
        Get all registerd workflows
        '''
        headers = self.prepare_header()
        response = requests.get(
            self.url + self.workflow_endpoint, headers=headers)

        return response.json()


if __name__ == "__main__":
    url = os.getenv('APP_URL','http://localhost:5001')
    scan_endpoint = '/api/scans/unprocessed?limit=1'
