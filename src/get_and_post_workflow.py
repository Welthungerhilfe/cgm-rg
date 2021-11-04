import glob
import json
import os

import log
from api_endpoints import ApiManager


logger = log.setup_custom_logger(__name__)


def get_list_of_files(source_folder):
    """Get list of json files in a folder"""
    glob_search_path = os.path.join(source_folder, "*.json")
    json_paths = glob.glob(glob_search_path)
    return json_paths


def log_status(workflows, workflow_obj, response):
    if response.status_code == 201:
        logger.info("successfully registered workflow for name %s and version %s",
                    workflow_obj['name'], workflow_obj['version'])

    elif response.status_code == 200:
        if response.json() != workflows[f"{workflow_obj['name']} {workflow_obj['version']}"]:
            logger.info("updated workflow for name %s and version %s",
                        workflow_obj['name'], workflow_obj['version'])
        else:
            logger.info("workflow for name %s and version %s is up to date",
                        workflow_obj['name'], workflow_obj['version'])
    elif response.status_code == 403:
        logger.info("attempted to update forbidden keys for %s version %s",
                    workflow_obj['name'], workflow_obj['version'])
    else:
        logger.info("unexpected error in registering workflow for name %s and version %s",
                    workflow_obj['name'], workflow_obj['version'])


def upsert_workflows(json_paths, workflows, cgm_api):
    workflows = {f"{workflow['name']} {workflow['version']}": workflow for workflow in workflows['workflows']}

    for path in json_paths:
        with open(path, 'r') as f:
            workflow_obj = json.load(f)

        response = cgm_api.post_workflow_and_save_response(workflow_obj)
        log_status(workflows, workflow_obj, response)


if __name__ == "__main__":
    url = os.getenv('APP_URL', 'http://localhost:5001')
    logger.info("%s %s", "App URL:", url)
    cgm_api = ApiManager(url)
    workflow_paths = 'src/workflows'
    json_paths = get_list_of_files(workflow_paths)
    workflows = cgm_api.get_workflows()
    upsert_workflows(json_paths, workflows, cgm_api)
