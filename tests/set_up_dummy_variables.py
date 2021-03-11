import os
import json
import pathlib


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_obj = json.load(f)

    return json_obj


os.environ['APP_ENV'] = 'LOCAL'

current_working_directory = pathlib.Path.cwd()


scan_endpoint = '/api/scans/unprocessed?limit=1'
get_file_endpoint = '/api/files/'
post_file_endpoint = '/api/files'
result_endpoint = '/api/results'
workflow_endpoint = '/api/workflows'
person_detail_endpoint = '/api/persons/'

person_details = {'age_estimated': False, 'date_of_birth': '2020-11-12', 'id': '95dd6f94-7bde-11eb-898c-d7e313a8f30a', 'qr_code': 'IN_AAH_RJ_TEST_JAN_2021_00002', 'qr_scanned': '2021-01-13T07:43:41Z', 'sex': 'female'}

workflows_path = current_working_directory.joinpath('tests', 'static_files', 'workflows.json')
# workflows_path = 'static_files/workflows.json'

url = "http://localhost:5001"

workflows = load_json(workflows_path)

scan_metadata_name = 'scan_meta_0933fc94-e4e6-4d39-89bf-2ecc19d39e5a.json'

# scan_metadata_path = 'static_files/scan_meta_0933fc94-e4e6-4d39-89bf-2ecc19d39e5a.json'
scan_metadata_path = current_working_directory.joinpath('tests', 'static_files', 'scan_meta_0933fc94-e4e6-4d39-89bf-2ecc19d39e5a.json')

scan_metadata = load_json(scan_metadata_path)['scans'][0]

rgb_artifacts = load_json(current_working_directory.joinpath('tests', 'static_files', 'rgb_artifacts.json'))['rgb_artifacts']

depth_artifacts = load_json(current_working_directory.joinpath('tests', 'static_files', 'depth_artifacts.json'))['depth_artifacts']

scan_parent_dir = 'static_files/'
