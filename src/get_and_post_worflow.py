import os
from api_endpoints import ApiEndpoints

if __name__ == "__main__":

    print("\nApp Environment : ", os.environ['APP_ENV'])

    if os.environ['APP_ENV'] == 'LOCAL':
        url = "http://localhost:5001"
    elif os.environ['APP_ENV'] == 'SANDBOX':
        url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"

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

    blur_workflow_path = 'src/schema/blur-workflow.json'
    blur_workflow_response_path = 'src/schema/blur-workflow-post.json'

    status_code = cgm_api.post_workflow_and_save_response(
        blur_workflow_path,
        blur_workflow_response_path)

    if status_code == 201:
        print("Blur Workflow Registration Successful")
