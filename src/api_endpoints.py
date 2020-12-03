import os
import cv2
import json
import pprint
import requests

class ApiEndpoints:
    def __init__(self, url, scan_endpoint, get_file_endpoint, post_file_endpoint, result_endpoint, workflow_endpoint):
        self.url = url
        self.scan_endpoint = scan_endpoint
        self.get_file_endpoint = get_file_endpoint
        self.post_file_endpoint = post_file_endpoint
        self.result_endpoint = result_endpoint
        self.workflow_endpoint = workflow_endpoint
        

    def get_files(self, file_id, save_dir):
        '''
        Get the files from api using file id
        '''
        endpoint = self.url + self.get_file_endpoint
        response = requests.get(endpoint + file_id)
        print("\nStatus code: ", response.status_code)

        file_path = os.path.join(save_dir, file_id)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        return response.status_code


    def post_files_using_path(self, file_path, type_):
        '''
        Post the files using the path of the file
        '''
        endpoint = self.url + self.post_file_endpoint

        files = {}
        files['file'] = (open(file_path, 'rb'), type_)
        files['filename'] = path.split('/')[-1]
        
        print('\nFile name to post : ', files['filename'])

        headers = {'content_type':'multipart/form-data'}
        response = requests.post(endpoint, files=files, headers=headers)
        file_id  = response.content.decode('utf-8')
        
        print("\nFile Id from post of test.jpg: ", file_id, '\n')

        return file_id, response.status_code


    def post_files(self, bin_file):
        '''
        Post the file result produced such as blur directly
        without saving it to a location to avoid I/O overhead
        '''
        endpoint = self.url + self.post_file_endpoint

        _, bin_file = cv2.imencode('.JPEG', bin_file)
        bin_file = bin_file.tostring()

        files = {
                'file': bin_file,
                'filename': 'test.jpg'
                }
        headers = {'content_type':'multipart/form-data'}
        
        response = requests.post(endpoint, files=files,  headers=headers)
        file_id  = response.content.decode('utf-8')
        
        print("File Id from post of test.jpg: ", file_id, '\n')

        return file_id, response.status_code

    def post_results(self, result_json_obj):
        '''
        Post the result object produced while Result Generation
        using POST /results
        '''
        endpoint = self.url + self.result_endpoint
        
        response = requests.post(endpoint, json=result_json_obj)

        print("Status of post result response: ", response.status_code, '\n')

        return response.status_code
    
    def post_workflow(self, workflow_path):
        '''
        Post the workflows using POST /files
        '''
        return str(uuid.uuid4()), 200


    def get_scan(self, scan_path):
        '''
        Get the scan metadata
        '''
        response = requests.get(self.url + self.scan_endpoint)
        if response.status_code == 200:
            content = response.json()
            print("\nScan Details : \n")
            pprint.pprint(content)

            with open(scan_path, 'w') as f:
                json.dump(content, f, indent=4)

            print("\n Written scan metadata successfully to ", scan_path)
        else:
            print("Response code : ", response.status_code)
        

if __name__ == "__main__":
    url = "http://localhost:5001"
    scan_endpoint = '/api/scan/scans/unprocessed?limit=1'

