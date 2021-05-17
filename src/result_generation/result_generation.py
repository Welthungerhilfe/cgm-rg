import json
import sys
from pathlib import Path

import numpy as np
from fastcore.basics import store_attr

sys.path.append(str(Path(__file__).parents[1]))
from api_endpoints import ApiEndpoints


class ResultGeneration:

    def __init__(
            self,
            api: ApiEndpoints,
            workflows,
            scan_metadata,
            scan_parent_dir):
        store_attr('api, workflows, scan_metadata, scan_parent_dir', self)

    def bunch_object_to_json_object(self, bunch_object):
        """Convert given bunch object to json object"""
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)
        return json_object

    def get_input_path(self, directory, file_name):
        """Return input path for given directory name and file name"""
        return Path(directory) / file_name

    def get_mean_scan_results(self, predictions):
        return str(np.mean(predictions))
