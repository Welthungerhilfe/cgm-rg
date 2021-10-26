import json
from api_endpoints import ApiEndpoints


class ProcessWorkflows:
    """Process all the workflows"""

    def __init__(self, api: ApiEndpoints):
        self.api = api

    def get_list_of_workflows(self):
        """Get the list of workflows from api"""
        self.workflows = self.api.get_workflows()

    def get_workflow_id(self, workflow_name, workflow_version):
        """Get the id of the workflow for given workflow name and version"""
        workflow_obj_with_id = list(
            filter(lambda workflow: (
                workflow['name'] == workflow_name and workflow['version'] == workflow_version),
                self.workflows['workflows']))[0]
        return workflow_obj_with_id['id']

    def load_workflows(self, workflow_path):
        """Load the workflow from given path"""
        with open(workflow_path, 'r') as f:
            workflow_obj = json.load(f)
        return workflow_obj

    def match_workflows(self, workflow_path, workflow_id):
        """Load the workflow from given path and get the id of the workflow
        and matches with the workflow_id"""
        workflow_obj = self.load_workflows(workflow_path)
        workflow_obj['id'] = self.get_workflow_id(
            workflow_obj['name'], workflow_obj['version'])

        return workflow_obj['id'] == workflow_id
