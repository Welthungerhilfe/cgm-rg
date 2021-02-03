import os
from api_endpoints import ApiEndpoints
import glob
import json


def get_list_of_files(source_folder):
    '''
    Get list of json files in a folder
    '''
    glob_search_path = os.path.join(source_folder, "*.json")
    json_paths = glob.glob(glob_search_path)

    return json_paths


def upsert_workflows(json_paths, workflows, cgm_api):

    workflows = {f"{workflow['name']} {workflow['version']}":workflow for workflow in workflows['workflows']}

    for path in json_paths:
        with open(path, 'r') as f:
            workflow_obj = json.load(f)

        response = cgm_api.post_workflow_and_save_response(workflow_obj)
        status_code = response.status_code
        if status_code == 201:
            print(f"successfully registered workflow for name {workflow_obj['name']} and version {workflow_obj['version']}")
        elif status_code == 200:
            if response.json() != workflows[f"{workflow_obj['name']} {workflow_obj['version']}"]:
                print(f"updated workflow for name {workflow_obj['name']} and version {workflow_obj['version']}")
            else:
                print(f"workflow for name {workflow_obj['name']} and version {workflow_obj['version']} is up to date")
        elif status_code == 403:
            print(f"attempted to update forbidden keys for {workflow_obj['name']} version {workflow_obj['version']}")
        else:
            print(f"unexpected error in registering workflow for name {workflow_obj['name']} and version {workflow_obj['version']}")


if __name__ == "__main__":

    print("\nApp Environment : ", os.environ['APP_ENV'])

    if os.environ['APP_ENV'] == 'LOCAL':
        url = "http://localhost:5001"
    elif os.environ['APP_ENV'] == 'SANDBOX':
        url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
    elif os.environ['APP_ENV'] == 'DEMO':
        url = "https://cgm-be-ci-qa-scanner-api.azurewebsites.net"

    scan_endpoint = '/api/scans/unprocessed?limit=1'
    get_file_endpoint = '/api/files/'
    post_file_endpoint = '/api/files'
    result_endpoint = '/api/results'
    workflow_endpoint = '/api/workflows'

    cgm_api = ApiEndpoints(
        url,
        scan_endpoint,
        get_file_endpoint,
        post_file_endpoint,
        result_endpoint,
        workflow_endpoint)

    # blur_workflow_path = 'src/schema/blur-workflow.json'
    # blur_workflow_response_path = 'src/schema/blur-workflow-post.json'

    workflow_paths = 'src/workflows'
    json_paths = get_list_of_files(workflow_paths)

    workflows = cgm_api.get_workflows()

    upsert_workflows(json_paths, workflows, cgm_api)
