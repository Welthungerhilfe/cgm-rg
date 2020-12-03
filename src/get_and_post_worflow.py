import json
import uuid
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
    # Creating unique name everytime of blur_workflow since storing in db solution
    # not implemented. Need to implement storage of worflows in DB and remove
    # this part
    workflow_obj['name'] = workflow_obj['name'] + '_' + str(uuid.uuid4())

    print("Workflow Post Object: ")
    pprint.pprint(workflow_obj)

    response = requests.post(endpoint, json=workflow_obj)

    print("Workflow Post response")
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

    # r = sample_get_request(url)
    # print("Content : ", r.content)
    # sample_post_request(url, file_path)

    blur_workflow_path = 'src/schema/blur-workflow.json'
    blur_workflow_response_path = 'src/schema/blur-workflow-post.json'

    status_code = post_workflow_and_save_response(
        url + workflow_endpoint,
        blur_workflow_path,
        blur_workflow_response_path)

    if status_code == 201:
        print("Blur Workflow Registration successfull")
