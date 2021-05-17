import json
from pathlib import Path

from bunch import Bunch


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


CWD = Path.cwd()

def create_dummy_vars():
    person_details = {'age_estimated': False,
                      'date_of_birth': '2020-11-12',
                      'id': '95dd6f94-7bde-11eb-898c-d7e313a8f30a',
                      'qr_code': 'IN_AAH_RJ_TEST_JAN_2021_00002',
                      'qr_scanned': '2021-01-13T07:43:41Z', 'sex': 'female'
                      }
    scan_metadata_name = 'scan_meta_0933fc94-e4e6-4d39-89bf-2ecc19d39e5a.json'
    scan_metadata_path = CWD.joinpath('tests', 'static_files', 'scan_meta_0933fc94-e4e6-4d39-89bf-2ecc19d39e5a.json')
    scan_metadata = load_json(scan_metadata_path)['scans'][0]

    return Bunch({
        "person_details": person_details,
        "url": "http://localhost:5001",
        "scan_metadata_name": scan_metadata_name,
        "scan_metadata_path": scan_metadata_path,
        "scan_metadata": scan_metadata,
        "scan_version": scan_metadata['version'],
        "rgb_artifacts": load_json(CWD.joinpath('tests', 'static_files', 'rgb_artifacts.json'))['rgb_artifacts'],
        "depth_artifacts": load_json(CWD.joinpath('tests', 'static_files', 'depth_artifacts.json'))['depth_artifacts'],
        "scan_parent_dir": CWD.joinpath('tests', 'static_files'),
    })