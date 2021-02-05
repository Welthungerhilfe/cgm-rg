import sys
sys.path.append('./src')
import unittest
sys.path.insert(0, '/opt/api')


#from src.api_endpoints import ApiEndpoints
#from src.result_gen_with_api import ScanResults
import api_endpoints
import result_gen_with_api

class TestScanResults(unittest.TestCase):
    def setup(self):
        if os.environ['APP_ENV'] == 'LOCAL':
            url = "http://localhost:5001"
        elif os.environ['APP_ENV'] == 'SANDBOX':
            url = "https://cgm-be-ci-dev-scanner-api.azurewebsites.net"
        elif os.environ['APP_ENV'] == 'DEMO':
            url = "https://cgm-be-ci-qa-scanner-api.azurewebsites.net"

        scan_endpoint = '/api/scans/unprocessed?limit=1'
        get_file_endpoint = '/api/files/'
        post_file_endpoint = '/api/files'
        result_endpoint = '/api/results'
        workflow_endpoint = '/api/workflows'

        # Need to provide
        scan_parent_dir = "/app/data/scans/"
        blur_workflow_path = "/app/src/workflows/blur-workflow.json"
        height_workflow_path = "/app/src/workflows/height-workflow.json"
        weight_workflow_path = "/app/src/workflows/weight-workflow.json"

        self.cgm_api = ApiEndpoints(
            url,
            scan_endpoint,
            get_file_endpoint,
            post_file_endpoint,
            result_endpoint,
            workflow_endpoint)

        # Do we need to get the workflow
        self.workflows = cgm_api.get_workflows() # we can remove it and will assume no workflow has been registered

        scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
        scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

        # We can mock this
        cgm_api.get_scan(scan_metadata_path)

        # scan_metadata_path = './schema/scan_with_blur_artifact.json'
        with open(scan_metadata_path, 'r') as f:
            scan_metadata = f.read()
        self.scan_metadata_obj = json.loads(scan_metadata)



    def test_height(self):

        if len(self.scan_metadata_obj['scans']) > 0:

            print(" Starting Result Generation Workflow on a scan")
            # Taking a single scan at a time
            scan_metadata_obj = self.scan_metadata_obj['scans'][0]

            with open(blur_workflow_path, 'r') as f:
                blur_workflow_obj = json.load(f)

            with open(height_workflow_path, 'r') as f:
                height_workflow_obj = json.load(f)

            with open(weight_workflow_path, 'r') as f:
                weight_workflow_obj = json.load(f)

            # mock get_workflow_id by using uuid
            blur_workflow_obj['id'] = get_workflow_id(blur_workflow_obj['name'], blur_workflow_obj['version'], workflows)
            height_workflow_obj['id'] = get_workflow_id(height_workflow_obj['name'], height_workflow_obj['version'], workflows)
            weight_workflow_obj['id'] = get_workflow_id(weight_workflow_obj['name'], weight_workflow_obj['version'], workflows)

            scan_results = ScanResults(
                scan_metadata_obj,
                blur_workflow_obj,
                height_workflow_obj,
                weight_workflow_obj,
                scan_parent_dir,
                cgm_api)

            scan_results.process_scan_metadata()
            scan_results.create_scan_and_artifact_dir()
            scan_results.download_depth_artifact()
            scan_results.run_height_flow()

        else:
            print("No Scan found without Results")

    def test_weight(self):
        scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
        scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

        # We can mock this
        cgm_api.get_scan(scan_metadata_path)

        # scan_metadata_path = './schema/scan_with_blur_artifact.json'
        with open(scan_metadata_path, 'r') as f:
            scan_metadata = f.read()
        scan_metadata_obj = json.loads(scan_metadata)

        if len(scan_metadata_obj['scans']) > 0:

            print(" Starting Result Generation Workflow on a scan")
            # Taking a single scan at a time
            scan_metadata_obj = scan_metadata_obj['scans'][0]

            with open(blur_workflow_path, 'r') as f:
                blur_workflow_obj = json.load(f)

            with open(height_workflow_path, 'r') as f:
                height_workflow_obj = json.load(f)

            with open(weight_workflow_path, 'r') as f:
                weight_workflow_obj = json.load(f)

            blur_workflow_obj['id'] = get_workflow_id(blur_workflow_obj['name'], blur_workflow_obj['version'], workflows)
            height_workflow_obj['id'] = get_workflow_id(height_workflow_obj['name'], height_workflow_obj['version'], workflows)
            weight_workflow_obj['id'] = get_workflow_id(weight_workflow_obj['name'], weight_workflow_obj['version'], workflows)

            scan_results = ScanResults(
                scan_metadata_obj,
                blur_workflow_obj,
                height_workflow_obj,
                weight_workflow_obj,
                scan_parent_dir,
                cgm_api)

            scan_results.process_scan_metadata()
            scan_results.create_scan_and_artifact_dir()
            scan_results.download_depth_artifact()
            scan_results.run_weight_flow()

        else:
            print("No Scan found without Results")


    def test_blur(self):
        scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
        scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

        # We can mock this
        cgm_api.get_scan(scan_metadata_path)


        # scan_metadata_path = './schema/scan_with_blur_artifact.json'
        with open(scan_metadata_path, 'r') as f:
            scan_metadata = f.read()
        scan_metadata_obj = json.loads(scan_metadata)

        if len(scan_metadata_obj['scans']) > 0:

            print(" Starting Result Generation Workflow on a scan")
            # Taking a single scan at a time
            scan_metadata_obj = scan_metadata_obj['scans'][0]

            with open(blur_workflow_path, 'r') as f:
                blur_workflow_obj = json.load(f)


            blur_workflow_obj['id'] = get_workflow_id(blur_workflow_obj['name'], blur_workflow_obj['version'], workflows)
            height_workflow_obj['id'] = get_workflow_id(height_workflow_obj['name'], height_workflow_obj['version'], workflows)
            weight_workflow_obj['id'] = get_workflow_id(weight_workflow_obj['name'], weight_workflow_obj['version'], workflows)

            scan_results = ScanResults(
                scan_metadata_obj,
                blur_workflow_obj,
                height_workflow_obj,
                weight_workflow_obj,
                scan_parent_dir,
                cgm_api)

            scan_results.process_scan_metadata()
            scan_results.create_scan_and_artifact_dir()
            scan_results.download_blur_flow_artifact()
            scan_results.run_blur_flow()

        else:
            print("No Scan found without Results")