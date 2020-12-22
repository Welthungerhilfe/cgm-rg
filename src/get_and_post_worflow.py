import os
import json
import pymongo

from api_endpoints import ApiEndpoints


def check_workflow_exists(workflow_path, collection):
    with open(workflow_path) as f:
        workflow_json = json.load(f)

    results = collection.count_documents({"name": workflow_json["name"], "version": workflow_json["version"]})
    print("no of existing document ", results)
    return results


def insert_workflow_document(workflow_path, collection):

    with open(workflow_path) as f:
        workflow_json = json.load(f)

    collection.insert_one(workflow_json)


if __name__ == "__main__":

    os.environ["APP_ENV"] = "LOCAL"
    print("\nApp Environment : ", os.environ['APP_ENV'])

    if os.environ['APP_ENV'] == 'LOCAL':
        url = "http://localhost:5001"
        collection_name = "local"
    elif os.environ['APP_ENV'] == 'SANDBOX':
        url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
        collection_name = "sandbox"
    elif os.environ['APP_ENV'] == 'DEMO':
        url = "https://cgm-be-ci-qa-scanner-api.azurewebsites.net"
        collection_name = "demo"

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

    cluster = client = pymongo.MongoClient("mongodb+srv://nikhil:bxQnvPpBDKI5VAnu@cluster0.y7zec.mongodb.net/<dbname>?retryWrites=true&w=majority")
    db = cluster['cgm-rg']
    collection = db['test']

    current_dir = os.path.dirname(os.path.realpath(__file__))
    print(current_dir)
    blur_workflow_path = os.path.join(current_dir, 'schema/blur-workflow.json')
    blur_workflow_response_path = os.path.join(current_dir, 'schema/blur-workflow-post.json')

    no_of_documents = check_workflow_exists(blur_workflow_path, collection)

    if no_of_documents == 0:
        status_code = cgm_api.post_workflow_and_save_response(
            blur_workflow_path,
            blur_workflow_response_path)

        if status_code == 201:
            print("Blur Workflow Registration Successful, adding to db")
            insert_workflow_document(blur_workflow_response_path, collection)
