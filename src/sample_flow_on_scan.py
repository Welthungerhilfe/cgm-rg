import os
import log
# import uuid

logger = log.setup_custom_logger(__name__)

from api_endpoints import ApiEndpoints
from process_workflows import ProcessWorkflows
# from get_scan_metadata import GetScanMetadata
from prepare_artifacts import PrepareArtifacts
from result_generation.result_generation import ResultGeneration
from result_generation.mlkit_pose_visual import MLkitPoseVisualise


def person(api, person_id):
    return api.get_person_details(person_id)


def scan_id_meta_data(api, scan_id):
    return api.get_scan_meta(scan_id)


def run_flow_on_single_scan(scan_id):

    app_pose_workflow_path = '/app/src/workflows/app_pose_workflow_path.json'
    mlkit_pose_visualize_pose_workflow_path = '/app/src/workflows/mlkit_pose_visualize_pose_workflow.json'
    # pose_visualization_workflow_path = '/app/src/workflows/pose-visualize-workflows.json'
    scan_parent_dir = '/app/data/scans/'
    # scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    # scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

    # URL
    url = os.getenv('APP_URL', 'http://localhost:5001')
    logger.info("%s %s", "App URL:", url)

    cgm_api = ApiEndpoints(url)
    workflow = ProcessWorkflows(cgm_api)
    # get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)
    # if get_scan_metadata.get_unprocessed_scans() <= 0:
    #     return
    # scan_metadata = get_scan_metadata.get_scan_metadata()

    scan_metadata = cgm_api.get_scan_metadata(scan_id)

    scan_version = scan_metadata['version']
    scan_type = scan_metadata["type"]
    logger.info("%s %s", "Scan Type Version:", scan_version)
    workflow.get_list_of_workflows()

    data_processing = PrepareArtifacts(cgm_api, scan_metadata, scan_parent_dir)
    data_processing.process_scan_metadata()
    data_processing.create_scan_dir()
    data_processing.create_artifact_dir()
    rgb_artifacts = data_processing.download_artifacts('img')
    # depth_artifacts = data_processing.download_artifacts('depth')
    # person_details = person(cgm_api, scan_metadata['person'])
    # scan_meta_data_details = scan_id_meta_data(cgm_api, scan_metadata['id'])

    # flows = []

    result_generation = ResultGeneration(cgm_api, workflow, scan_metadata, scan_parent_dir)

    # flow = PoseAndBlurFlow(
    #     result_generation,
    #     blur_workflow_path,
    #     blur_faces_workflow_path,
    #     pose_workflow_path,
    #     pose_visualization_workflow_path,
    #     rgb_artifacts,
    #     scan_version,
    #     scan_type, ['POSE', 'BLUR'])
    # flows.append(flow)

    flow = MLkitPoseVisualise(
        result_generation,
        app_pose_workflow_path,
        mlkit_pose_visualize_pose_workflow_path,
        rgb_artifacts,
        scan_version,
        scan_type)

    flow.run_flow()


if __name__ == '__main__':
    scan_id = '01d77020-6f01-11ed-ad18-ff1a7240801c'
    run_flow_on_single_scan(scan_id)
