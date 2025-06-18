import logging
import anyio
import asyncer
from asyncer import asyncify

from utils.constants import *
from utils.processing import get_workflow, load_rgb_images, encode_rgb_images, pose_and_blur_visualsation, blur_rgb_images
from rg.workflows import run_face_workflow, run_pose_workflow
from rg.results_utils import get_files_results, post_result_files, get_blur_json_results, get_pose_json_results


async def run_rgb_flow(cgm_api, session, artifacts, workflows, results):
    try:
        logging.info("Starting rgb flow")
        pose_workflow = get_workflow(workflows, POSE_WORKFLOW_NAME, POSE_WORKFLOW_VERSION)
        pose_visualize_workflow = get_workflow(workflows, POSE_VISUALIZE_WORKFLOW_NAME, POSE_VISUALIZE_WORKFLOW_VERSION)
        blur_workflow = get_workflow(workflows, BLUR_WORKFLOW_NAME, BLUR_WORKFLOW_VERSION)
        faces_workflow = get_workflow(workflows, FACE_DETECTION_WORKFLOW_NAME, FACE_DETECTION_WORKFLOW_VERSION)
        rgb_input_images = await asyncify(load_rgb_images)(artifacts)
        encoded_images = await asyncify(encode_rgb_images)(rgb_input_images)
        logging.info("generating predictions")
        async with asyncer.create_task_group() as task_group:
            pose_results = task_group.soonify(run_pose_workflow)(session, encoded_images)
            face_results = task_group.soonify(run_face_workflow)(session, encoded_images)
        pose_results = pose_results.value
        face_results = face_results.value
        logging.info("generating images")
        blurred_images, blurred_images_to_post = await asyncify(blur_rgb_images)(artifacts, face_results, rgb_input_images)
        pose_viz = await asyncify(pose_and_blur_visualsation)(artifacts, pose_results, blurred_images)
        logging.info("uploading images")
        blur_file_ids = await post_result_files(cgm_api, blurred_images_to_post)
        pose_vis_file_ids = await post_result_files(cgm_api, pose_viz)
        blur_results_dicts = get_files_results(blur_file_ids, blur_workflow['id'], results.get(blur_workflow['id'], []))
        pose_results_dicts = get_files_results(pose_vis_file_ids, pose_visualize_workflow['id'], results.get(pose_visualize_workflow['id'], []))
        pose_json_results_dicts = get_pose_json_results(artifacts, pose_results, pose_workflow['id'], results.get(pose_workflow['id'], []))
        face_json_results_dicts = get_blur_json_results(artifacts, face_results, faces_workflow['id'], results.get(faces_workflow['id'], []))
        logging.info("uploading results")
        async with asyncer.create_task_group() as task_group:
            blur_result_post_status = task_group.soonify(cgm_api.post_results)({"results": blur_results_dicts})
            pose_result_post_status = task_group.soonify(cgm_api.post_results)({"results": pose_results_dicts})
            pose_json_result_post_status = task_group.soonify(cgm_api.post_results)({"results": pose_json_results_dicts})
            blur_json_result_post_status = task_group.soonify(cgm_api.post_results)({"results": face_json_results_dicts})
        logging.info(f"{blur_result_post_status.value}, {pose_result_post_status.value}, {pose_json_result_post_status.value}, {blur_json_result_post_status.value}")
        return True, rgb_input_images
    except Exception as e:
        logging.error(f"Error in run_rgb_flow: {e}")
        raise e
