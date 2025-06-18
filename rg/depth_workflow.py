import logging
import anyio
import asyncer
from asyncer import asyncify
import traceback


from utils.constants import *
from utils.processing import get_workflow
from utils.depth_preprocessing import get_depthmaps, compute_depth_metadata, depth_visualization, compute_angle
from utils.constants import *
from rg.results_utils import get_files_results, post_result_files, get_depth_feature_json_results, get_json_results
from utils.inference import call_mn_height, call_pcnn_height, call_pcnn_weight


async def run_depth_img_flow(cgm_api, session, artifacts, workflows, scan_version, results, scan_type):
    try:
        logging.info("starting depth img flow")
        depth_img_workflow = get_workflow(workflows, DEPTH_IMG_WORKFLOW_NAME, DEPTH_IMG_WORKFLOW_VERSION)
        depth_feature_workflow = get_workflow(workflows, DEPTH_FEATURE_WORKFLOW_NAME, DEPTH_FEATURE_WORKFLOW_VERSION)
        pcnn_height_workflow = get_workflow(workflows, PLAINCNN_HEIGHT_WORKFLOW_NAME, PLAINCNN_HEIGHT_WORKFLOW_VERSION)
        pcnn_weight_workflow = get_workflow(workflows, PLAINCNN_WEIGHT_WORKFLOW_NAME, PLAINCNN_WEIGHT_WORKFLOW_VERSION)
        mn_height_workflow = get_workflow(workflows, MOBILENET_HEIGHT_WORKFLOW_NAME, MOBILENET_HEIGHT_WORKFLOW_VERSION)
        logging.info("starting preprocessing")
        depthmaps, in_depthmaps, pc_dmaps, mn_depthmaps, device_poses = await asyncify(get_depthmaps)(artifacts, scan_version)
        logging.info("starting no of zeroes")
        no_of_zeroes_results = await asyncify(compute_depth_metadata)(depthmaps)
        logging.info("starting angle")
        angle_results = await asyncify(compute_angle)(device_poses)
        logging.info("starting depth viz")
        depth_viz = await asyncify(depth_visualization)(artifacts, depthmaps, scan_type)
        logging.info("starting height and weight models")
        async with asyncer.create_task_group() as task_group:
            pcnn_height_results = task_group.soonify(call_pcnn_height)(session, pc_dmaps)
            pcnn_weight_results = task_group.soonify(call_pcnn_weight)(session, pc_dmaps)
            mn_height_results = task_group.soonify(call_mn_height)(session, mn_depthmaps)
        pcnn_height_results = pcnn_height_results.value
        pcnn_weight_results = pcnn_weight_results.value
        mn_height_results = mn_height_results.value
        logging.info("starting posting")
        file_ids = await post_result_files(cgm_api, depth_viz)
        depth_img_results_dicts = get_files_results(file_ids, depth_img_workflow['id'], results.get(depth_img_workflow['id'], []))
        depth_feature_json_results_dicts = get_depth_feature_json_results(artifacts, no_of_zeroes_results, angle_results, depth_feature_workflow['id'], results.get(depth_feature_workflow['id'], []))
        pcnn_height_json_results_dicts = get_json_results(artifacts, pcnn_height_results, pcnn_height_workflow['id'], results.get(pcnn_height_workflow['id'], []), 'height')
        pcnn_weight_json_results_dicts = get_json_results(artifacts, pcnn_weight_results, pcnn_weight_workflow['id'], results.get(pcnn_weight_workflow['id'], []), 'weight')
        mn_height_json_results_dicts = get_json_results(artifacts, mn_height_results, mn_height_workflow['id'], results.get(mn_height_workflow['id'], []), 'height')

        logging.info("uploading results")
        async with asyncer.create_task_group() as task_group:
            depth_img_post_status = task_group.soonify(cgm_api.post_results)({"results": depth_img_results_dicts})
            depth_feature_post_status = task_group.soonify(cgm_api.post_results)({"results": depth_feature_json_results_dicts})
            pcnn_height_post_status = task_group.soonify(cgm_api.post_results)({"results": pcnn_height_json_results_dicts})
            pcnn_weight_post_status = task_group.soonify(cgm_api.post_results)({"results": pcnn_weight_json_results_dicts})
            # if 'ir' not in scan_version:
            mn_height_post_status = task_group.soonify(cgm_api.post_results)({"results": mn_height_json_results_dicts})
        logging.info(f"{depth_img_post_status.value}, {depth_feature_post_status.value}, {pcnn_height_post_status.value}, {pcnn_weight_post_status.value}, {mn_height_post_status.value}")
        return True, depthmaps
    except Exception as e:
        logging.error(f"Error in run_depth_img_flow: {e} {traceback.format_exc()}")
        raise e
