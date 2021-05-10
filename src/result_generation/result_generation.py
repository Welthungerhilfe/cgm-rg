import json
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parents[1]))
from api_endpoints import ApiEndpoints


class ResultGeneration:

    def __init__(
            self,
            api: ApiEndpoints,
            workflows,
            scan_metadata):
        self.api = api
        self.workflows = workflows
        self.scan_metadata = scan_metadata

    def bunch_object_to_json_object(self, bunch_object):
        """Convert given bunch object to json object"""
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)
        return json_object

    def get_input_path(self, directory, file_name):
        """Return input path for given directory name and file name"""
        return os.path.join(directory, file_name)
