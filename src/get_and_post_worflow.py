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


def check_workflow_exists(json_obj, workflows):
    return any((d.get('name') == json_obj['name'] and d.get('version') == json_obj['version']) for d in workflows)


def check_workflows(json_paths, workflows, cgm_api):

    for path in json_paths:
        with open(path, 'r') as f:
            workflow_obj = json.load(f)

        if not check_workflow_exists(workflow_obj, workflows):
            status_code = cgm_api.post_workflow_and_save_response(workflow_obj)
            if status_code == 201:
                print("successfully registered workflow for name ", workflow_obj['name'], " and version ", workflow_obj['version'])
        else:
            print("workflow for name ", workflow_obj['name'], " and version ", workflow_obj['version'], "already exists")


if __name__ == "__main__":

    print("\nApp Environment : ", os.environ['APP_ENV'])

    if os.environ['APP_ENV'] == 'LOCAL':
        url = "http://localhost:5001"
    elif os.environ['APP_ENV'] == 'SANDBOX':
        url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
    elif os.environ['APP_ENV'] == 'DEMO':
        url = "https://cgm-be-ci-qa-scanner-api.azurewebsites.net"

    scan_endpoint = '/api/scan/scans/unprocessed?limit=1'
    get_file_endpoint = '/api/scan/files/'
    post_file_endpoint = '/api/scan/files'
    result_endpoint = '/api/scan/results'
    workflow_endpoint = '/api/scan/workflows'

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

    check_workflows(json_paths, workflows['workflows'], cgm_api)
