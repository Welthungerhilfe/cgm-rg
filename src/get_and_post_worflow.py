import json
import pprint
import requests

def sample_get_request(url):
    r = requests.get(url)
    return r

def sample_post_request(url, file_path):
    files = {'file': open(file_path, 'rb')}
    r = requests.post(url, files=files)
    return r

def post_workflow_and_save_response(endpoint, workflow_path, response_path):
    # Read the workflow json
    with open(workflow_path, 'r') as f:
        workflow = f.read()
    
    workflow_obj = json.loads(workflow)
    print("Workflow Post Object: ")
    pprint.pprint(workflow_obj)
    
    response = requests.post(endpoint, json=workflow_obj)
    
    print("Worflow Post response")
    print("Status code: ", response.status_code)
    
    if response.status_code == 201:
        content = response.json()

        content['meta'] = workflow_obj["meta"]
        pprint.pprint(content)

        with open(response_path, 'w') as f:
            json.dump(content, f)

    return response.status_code


if __name__ == "__main__":

    url = "http://localhost:5001"
    workflow_endpoint = '/api/scan/workflows'
    
    #r = sample_get_request(url)
    #print("Content : ", r.content)
    #sample_post_request(url, file_path)

    blur_workflow_path = './schema/blur-workflow.json'
    blur_workflow_response_path = './schema/blur-worflow-post.json'

    status_code = post_workflow_and_save_response(url + workflow_endpoint, 
                    blur_workflow_path, blur_workflow_response_path)


    if status_code == 201:
        print("Blur Worflow Registration sucessfull")
