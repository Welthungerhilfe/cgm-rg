import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from bunch import Bunch

sys.path.append(str(Path(__file__).parents[1]))
import utils.preprocessing as preprocessing  # noqa: E402


class DepthMapImgFlow:
    """
    A class to visualise depthmap image in result generation
    """

    def __init__(
            self,
            api,
            workflows,
            workflow_path,
            artifacts,
            scan_parent_dir,
            scan_metadata):
        self.api = api
        self.workflows = workflows
        self.artifacts = artifacts
        self.workflow_path = workflow_path
        self.workflow_obj = self.workflows.load_workflows(self.workflow_path)
        self.scan_metadata = scan_metadata
        self.scan_parent_dir = scan_parent_dir
        if self.workflow_obj["data"]["input_format"] == 'application/zip':
            self.depth_input_format = 'depth'
        self.scan_directory = os.path.join(
            self.scan_parent_dir,
            self.scan_metadata['id'],
            self.depth_input_format)
        self.workflow_obj['id'] = self.workflows.get_workflow_id(
            self.workflow_obj['name'], self.workflow_obj['version'])
        self.colormap = plt.get_cmap('inferno')

    def run_flow(self):
        self.depthmap_img_artifacts()
        self.post_depthmap_image_files()
        self.post_result_object()

    def get_input_path(self, directory, file_name):
        return os.path.join(directory, file_name)

    def bunch_object_to_json_object(self, bunch_object):
        json_string = json.dumps(bunch_object, indent=2, separators=(',', ':'))
        json_object = json.loads(json_string)
        return json_object

    def preprocess_depthmap(self, input_path):
        data, width, height, depthScale, _max_confidence = preprocessing.load_depth(input_path)

        depthmap = preprocessing.prepare_depthmap(data, width, height, depthScale)

        # depthmap = preprocessing.preprocess(depthmap)
        # depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
        return depthmap, True

    def depthmap_img_artifacts(self):
        for artifact in self.artifacts:
            input_path = self.get_input_path(self.scan_directory, artifact['file'])
            depthmap, depthmap_status = self.preprocess_depthmap(input_path)
            scaled_depthmap = depthmap * 255.0
            if depthmap_status:
                artifact['depthmap_img'] = scaled_depthmap

    def post_depthmap_image_files(self):
        for artifact in self.artifacts:
            depthmap_img_id_from_post_request, post_status = self.api.post_files(artifact['depthmap_img'])
            if post_status == 201:
                artifact['depthmap_img_id_from_post_request'] = depthmap_img_id_from_post_request
                artifact['generated_timestamp'] = datetime.now().strftime(
                    '%Y-%m-%dT%H:%M:%SZ')

    def prepare_result_object(self):
        res = Bunch()
        res.results = []
        for artifact in self.artifacts:
            depthmap_img_result = Bunch()
            depthmap_img_result.id = f"{uuid.uuid4()}"
            depthmap_img_result.scan = self.scan_metadata['id']
            depthmap_img_result.workflow = self.workflow_obj["id"]
            depthmap_img_result.source_artifacts = [artifact['id']]
            depthmap_img_result.source_results = []
            depthmap_img_result.file = artifact['depthmap_img_id_from_post_request']
            depthmap_img_result.generated = artifact['generated_timestamp']
            res.results.append(depthmap_img_result)

        return res

    def post_result_object(self):
        depthmap_img_result = self.prepare_result_object()
        depthmap_img_result_object = self.bunch_object_to_json_object(
            depthmap_img_result)
        if self.api.post_results(depthmap_img_result_object) == 201:
            print(
                "successfully post Depthmap Image results: ",
                depthmap_img_result_object)
