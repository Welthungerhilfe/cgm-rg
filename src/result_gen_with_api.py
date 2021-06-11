import argparse
import base64
import json
import os
import uuid
from azure.storage.queue import QueueService

from api_endpoints import ApiEndpoints
from get_scan_metadata import GetScanMetadata
from process_workflows import ProcessWorkflows
from prepare_artifacts import PrepareArtifacts
from result_generation.blur import BlurFlow
from result_generation.result_generation import ResultGeneration
from result_generation.depthmap_image import DepthMapImgFlow
from result_generation.height.height_plaincnn import HeightFlowPlainCnn
from result_generation.height.height_multiartifact import HeightFlowMultiArtifact
from result_generation.standing import StandingLaying
from result_generation.weight import WeightFlow
from result_generation.height.height_rgbd import HeightFlowRGBD


def person(api, person_id):
    return api.get_person_details(person_id)


def parse_args():
    parser = argparse.ArgumentParser()
    workflow_dir = '/app/src/workflows'
    parser.add_argument('--scan_parent_dir', default="data/scans/", help='Parent directory in which scans will be stored')  # noqa: E501
    parser.add_argument('--blur_workflow_path', default=f"{workflow_dir}/blur-workflow.json")  # noqa: E501
    parser.add_argument('--standing_laying_workflow_path', default=f"{workflow_dir}/standing_laying-workflow.json")  # noqa: E501
    parser.add_argument('--depthmap_img_workflow_path', default=f"{workflow_dir}/depthmap-img-workflow.json")  # noqa: E501
    parser.add_argument('--height_workflow_artifact_path', default=f"{workflow_dir}/height-plaincnn-workflow-artifact.json", help='Height Workflow Artifact path')  # noqa: E501
    parser.add_argument('--height_depthmapmultiartifactlatefusion_workflow_path', default=f"{workflow_dir}/height-depthmapmultiartifactlatefusion-workflow.json")  # noqa: E501
    parser.add_argument('--height_workflow_scan_path', default=f"{workflow_dir}/height-plaincnn-workflow-scan.json")  # noqa: E501
    parser.add_argument('--weight_workflow_artifact_path', default=f"{workflow_dir}/weight-workflow-artifact-run_05.json")  # noqa: E501
    parser.add_argument('--weight_workflow_scan_path', default=f"{workflow_dir}/weight-workflow-scan-run_05.json")  # noqa: E501
    parser.add_argument('--height_rgbd_workflow_artifact_path', default=f"{workflow_dir}/height-rgbd-workflow-artifact.json")  # noqa: E501
    parser.add_argument('--height_rgbd_workflow_scan_path', default=f"{workflow_dir}/height-rgbd-workflow-scan.json")  # noqa: E501
    args = parser.parse_args()
    return args


def run_normal_flow():
    args = parse_args()
    scan_parent_dir = args.scan_parent_dir
    blur_workflow_path = args.blur_workflow_path
    standing_laying_workflow_path = args.standing_laying_workflow_path
    depthmap_img_workflow_path = args.depthmap_img_workflow_path
    height_workflow_artifact_path = args.height_workflow_artifact_path
    height_workflow_scan_path = args.height_workflow_scan_path
    height_depthmapmultiartifactlatefusion_workflow_path = args.height_depthmapmultiartifactlatefusion_workflow_path
    weight_workflow_artifact_path = args.weight_workflow_artifact_path
    weight_workflow_scan_path = args.weight_workflow_scan_path
    height_rgbd_workflow_artifact_path = args.height_rgbd_workflow_artifact_path
    height_rgbd_workflow_scan_path = args.height_rgbd_workflow_scan_path

    scan_metadata_name = 'scan_meta_' + str(uuid.uuid4()) + '.json'
    scan_metadata_path = os.path.join(scan_parent_dir, scan_metadata_name)

    # URL
    url = os.getenv('APP_URL', 'http://localhost:5001')
    print(f"App URL: {url}")
    cgm_api = ApiEndpoints(url)

    workflow = ProcessWorkflows(cgm_api)
    get_scan_metadata = GetScanMetadata(cgm_api, scan_metadata_path)

    if get_scan_metadata.get_unprocessed_scans() <= 0:
        return

    scan_metadata = get_scan_metadata.get_scan_metadata()
    scan_version = scan_metadata['version']
    print("Scan Type Version: ", scan_version)
    workflow.get_list_of_worflows()

    data_processing = PrepareArtifacts(cgm_api, scan_metadata, scan_parent_dir)
    data_processing.process_scan_metadata()
    data_processing.create_scan_dir()
    data_processing.create_artifact_dir()
    rgb_artifacts = data_processing.download_artifacts('img')
    depth_artifacts = data_processing.download_artifacts('depth')
    person_details = person(cgm_api, scan_metadata['person'])

    flows = []

    result_generation = ResultGeneration(cgm_api, workflow, scan_metadata, scan_parent_dir)

    flow = BlurFlow(
        result_generation,
        blur_workflow_path,
        rgb_artifacts,
        scan_version)
    flows.append(flow)

    flow = StandingLaying(
        result_generation,
        standing_laying_workflow_path,
        rgb_artifacts)
    flows.append(flow)

    flow = DepthMapImgFlow(
        result_generation,
        depthmap_img_workflow_path,
        depth_artifacts)
    flows.append(flow)

    flow = HeightFlowPlainCnn(
        result_generation,
        height_workflow_artifact_path,
        height_workflow_scan_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    flow = HeightFlowMultiArtifact(
        result_generation,
        height_workflow_artifact_path,
        height_depthmapmultiartifactlatefusion_workflow_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    flow = WeightFlow(
        result_generation,
        weight_workflow_artifact_path,
        weight_workflow_scan_path,
        depth_artifacts,
        person_details)
    flows.append(flow)

    flow = HeightFlowRGBD(
        result_generation,
        height_rgbd_workflow_artifact_path,
        height_rgbd_workflow_scan_path,
        depth_artifacts,
        person_details,
        rgb_artifacts)
    flows.append(flow)

    for flow in flows:
        try:
            flow.run_flow()
        except Exception as e:
            print(e)


def run_retroactive_flow():

    args = parse_args()
    # scan_parent_dir = args.scan_parent_dir
    blur_workflow_path = args.blur_workflow_path
    standing_laying_workflow_path = args.standing_laying_workflow_path
    depthmap_img_workflow_path = args.depthmap_img_workflow_path
    height_workflow_artifact_path = args.height_workflow_artifact_path
    height_workflow_scan_path = args.height_workflow_scan_path
    height_depthmapmultiartifactlatefusion_workflow_path = args.height_depthmapmultiartifactlatefusion_workflow_path
    weight_workflow_artifact_path = args.weight_workflow_artifact_path
    weight_workflow_scan_path = args.weight_workflow_scan_path
    height_rgbd_workflow_artifact_path = args.height_rgbd_workflow_artifact_path
    height_rgbd_workflow_scan_path = args.height_rgbd_workflow_scan_path

    print("Started Retroactive Flow")
    # Retrieve the connection string from an environment
    # variable named AZURE_STORAGE_CONNECTION_STRING
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    # URL
    url = os.getenv('APP_URL', 'http://localhost:5001')
    print(f"App URL: {url}")
    queue_name = "retroactive-scan-process"
    retroactive_scan_dir = '/api/data/retroactive_scans/'

    cgm_api = ApiEndpoints(url)
    workflow = ProcessWorkflows(cgm_api)
    workflow.get_list_of_worflows()
    queue_service = QueueService(connection_string=connect_str)
    messages = queue_service.get_messages(queue_name, num_messages=1, visibility_timeout=5 * 60)
    print("Length of messages : ", len(messages))

    for message in messages:
        original_msg = base64.b64decode(message.content)
        print("message : ", original_msg)

        scan_metadata_with_workflow_obj = json.loads(original_msg)
        print("Scan Metadata with Workflow")
        print(scan_metadata_with_workflow_obj)

        scan_metadata = scan_metadata_with_workflow_obj['scans'][0]
        scan_version = scan_metadata['version']
        print("Scan Type Version: ", scan_version)

        workflow_id = scan_metadata_with_workflow_obj['workflow_id']
        print("Workflow ID : ", workflow_id)

        data_processing = PrepareArtifacts(cgm_api, scan_metadata, retroactive_scan_dir)
        data_processing.process_scan_metadata()
        data_processing.create_scan_dir()
        data_processing.create_artifact_dir()
        rgb_artifacts = data_processing.download_artifacts('img')
        depth_artifacts = data_processing.download_artifacts('depth')
        person_details = person(cgm_api, scan_metadata['person'])

        result_generation = ResultGeneration(cgm_api, workflow, scan_metadata, retroactive_scan_dir)

        workflow_matched = True

        if workflow.match_workflows(blur_workflow_path, workflow_id):
            print("BlurFlow")
            flow = BlurFlow(
                result_generation,
                blur_workflow_path,
                rgb_artifacts,
                scan_version)

        elif workflow.match_workflows(standing_laying_workflow_path, workflow_id):
            print("StandingLaying")
            flow = StandingLaying(
                result_generation,
                standing_laying_workflow_path,
                rgb_artifacts)

        elif workflow.match_workflows(depthmap_img_workflow_path, workflow_id):
            print("DepthMapImgFlow")
            flow = DepthMapImgFlow(
                result_generation,
                depthmap_img_workflow_path,
                depth_artifacts)

        elif workflow.match_workflows(height_workflow_scan_path, workflow_id):
            print("HeightFlowPlainCnn")
            flow = HeightFlowPlainCnn(
                result_generation,
                height_workflow_artifact_path,
                height_workflow_scan_path,
                depth_artifacts,
                person_details)

        elif workflow.match_workflows(height_depthmapmultiartifactlatefusion_workflow_path, workflow_id):
            print("HeightFlowMultiArtifact")
            flow = HeightFlowMultiArtifact(
                result_generation,
                height_workflow_artifact_path,
                height_depthmapmultiartifactlatefusion_workflow_path,
                depth_artifacts,
                person_details)

        elif workflow.match_workflows(weight_workflow_scan_path, workflow_id):
            print("WeightFlow")
            flow = WeightFlow(
                result_generation,
                weight_workflow_artifact_path,
                weight_workflow_scan_path,
                depth_artifacts,
                person_details)

        elif workflow.match_workflows(height_rgbd_workflow_scan_path, workflow_id):
            print("HeightFlowRGBD")
            flow = HeightFlowRGBD(
                result_generation,
                height_rgbd_workflow_artifact_path,
                height_rgbd_workflow_scan_path,
                depth_artifacts,
                person_details,
                rgb_artifacts)
        else:
            workflow_matched = False
            print("workflow id does not match with any of the workflow id")

        if workflow_matched:
            try:
                flow.run_flow()
            except Exception as e:
                print(e)

        queue_service.delete_message(queue_name, message.id, message.pop_receipt)


def main():
    run_normal_flow()
    run_retroactive_flow()


if __name__ == "__main__":
    main()
