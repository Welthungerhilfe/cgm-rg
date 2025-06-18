import os
import logging
import anyio
import asyncer
from asyncer import asyncify
import aiohttp
import numpy as np
import pickle
import tensorflow as tf


from utils.rest_api import CgmApi
from utils.constants import *
from utils.processing import download_artifacts, get_scan_by_format, get_workflow, check_rgb_depth_alignment, load_rgb_image, plot_with_masks_on_image
from rg.rgb_workflows import run_rgb_flow
from rg.depth_workflow import run_depth_img_flow
from rg.results_utils import get_mean_result, get_mean_results, filter_results, pose_score_filter, depth_feature_filter, distance_to_child_filter, get_rgb_depth_allignment_result, get_result_dict, get_json_results
from utils.inference import call_sam_api, call_mn_height
from utils.depth_preprocessing import get_raw_depthmap, inpaint_depth_all_masks, save_plot_as_binary_new, IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, NORMALIZATION_VALUE


async def run_rg(scan_ids):
    try:
        cgm_api = CgmApi()
        session = aiohttp.ClientSession()
        workflows = await cgm_api.get_workflows()
        async with asyncer.create_task_group() as task_group:
            scan_metadata_values = [task_group.soonify(cgm_api.get_scan_metadata)(scan_id) for scan_id in scan_ids]
        scan_id_metadata = [scan_metadata.value for scan_metadata in scan_metadata_values]
        version = scan_id_metadata[0]['version']
        artifacts = [{**artifact, "scan_id": data["id"]} for data in scan_id_metadata for artifact in data["artifacts"]]
        if scan_id_metadata[0]['type'] in STANDING_SCAN_TYPE:
            scan_type = STANDING_TYPE
        elif scan_id_metadata[0]['type'] in LAYING_SCAN_TYPE:
            scan_type = LYING_TYPE
        else:
            raise Exception('unknown scan type')
        logging.info("downloading artifacts")
        await download_artifacts(cgm_api, artifacts, version)
        logging.info("finished downloading artifacts")
        depth_artifacts = await asyncify(get_scan_by_format)(artifacts, depth_format)
        rgb_artifacts = await asyncify(get_scan_by_format)(artifacts, rgb_format)
        results = [result for data in scan_id_metadata for result in data["results"]]
        results_workflow_dict = {}
        for r in results:
            k = r['workflow']
            if k not in results_workflow_dict:
                results_workflow_dict[k] = []
            results_workflow_dict[k].extend(r['source_artifacts'])
        logging.info("Starting flow")
        async with asyncer.create_task_group() as task_group:
            rgb_workflow_status = task_group.soonify(run_rgb_flow)(cgm_api, session, rgb_artifacts, workflows, results_workflow_dict)
            depth_workflow_status = task_group.soonify(run_depth_img_flow)(cgm_api, session, depth_artifacts, workflows, version, results_workflow_dict, scan_type)
        rgb_wf_status, rgb_images = rgb_workflow_status.value
        depth_wf_status, depthmaps = depth_workflow_status.value
        allignment_workflow = get_workflow(workflows, RGB_DEPTH_ALLIGNMENT_WORKFLOW_NAME, RGB_DEPTH_ALLIGNMENT_WORKFLOW_VERSION)
        rgb_artifact_to_ord_mapping = {(ra['scan_id'], ra['order']): ra['id'] for ra in rgb_artifacts}
        if scan_type == STANDING_TYPE:
            max_depth = 3
        elif scan_type == LYING_TYPE:
            max_depth = 1.5
        alignment_rds = []
        for da, dm in zip(depth_artifacts, depthmaps):
            rgb_a_id = rgb_artifact_to_ord_mapping[(da['scan_id'], da['order'])]
            similarity_index, alignment_status = check_rgb_depth_alignment(rgb_images[rgb_a_id], dm, max_depth, threshold=0.75)
            alignment_rds.append(get_rgb_depth_allignment_result(da['scan_id'], [da['id']], allignment_workflow['id'], similarity_index, alignment_status))
        post_allignment_status = await cgm_api.post_results({"results": alignment_rds})
        logging.info(f"allignment post status {post_allignment_status}")
    finally:
        await cgm_api.session.close()
        await session.close()


async def run_mean_rg(scan_ids):
    try:
        cgm_api = CgmApi()
        workflows = await cgm_api.get_workflows()
        results_dicts = []
        scan_preds = []
        for scan_id in scan_ids:
            result_dict, scan_pred, child_visit_id = await process_scan_id(cgm_api, scan_id, workflows)
            results_dicts.extend(result_dict)
            if scan_pred:
                scan_preds.append(scan_pred)
        post_result_status_code = await cgm_api.post_results({"results": results_dicts})
        estimate = get_mean_result(scan_preds) if scan_preds else None
        if estimate:
            result_data = {'mean_height': f'{estimate:.3f}'}
        else:
            result_data = {'mean_height': estimate}
        status_code = await cgm_api.put_child_visit_result(child_visit_id, result_data)
        logging.info(f"{post_result_status_code}, {status_code}")
    finally:
        await cgm_api.session.close()


async def process_scan_id(cgm_api, scan_id, workflows):
    result_dicts = []
    sm = await cgm_api.get_scan_metadata(scan_id)
    app_child_distance_workflow = get_workflow(workflows, 'app_child_distance', '1.0')
    depth_feature_workflow = get_workflow(workflows, DEPTH_FEATURE_WORKFLOW_NAME, DEPTH_FEATURE_WORKFLOW_VERSION)
    pose_workflow = get_workflow(workflows, POSE_WORKFLOW_NAME, POSE_WORKFLOW_VERSION)
    filter_functions_dict = {
        pose_workflow['id']: pose_score_filter,
        depth_feature_workflow['id']: depth_feature_filter,
        app_child_distance_workflow['id']: distance_to_child_filter,
    }

    pcnn_height_workflow = get_workflow(workflows, PLAINCNN_HEIGHT_WORKFLOW_NAME, PLAINCNN_HEIGHT_WORKFLOW_VERSION)
    pcnn_weight_workflow = get_workflow(workflows, PLAINCNN_WEIGHT_WORKFLOW_NAME, PLAINCNN_WEIGHT_WORKFLOW_VERSION)
    mn_height_workflow = get_workflow(workflows, MOBILENET_HEIGHT_WORKFLOW_NAME, MOBILENET_HEIGHT_WORKFLOW_VERSION)
    pcnn_height_mean_workflow = get_workflow(workflows, MEAN_PLAINCNN_HEIGHT_WORKFLOW_NAME, MEAN_PLAINCNN_HEIGHT_WORKFLOW_VERSION)
    pcnn_weight_mean_workflow = get_workflow(workflows, MEAN_PLAINCNN_WEIGHT_WORKFLOW_NAME, MEAN_PLAINCNN_WEIGHT_WORKFLOW_VERSION)
    mn_height_mean_workflow = get_workflow(workflows, MEAN_MOBILENET_HEIGHT_WORKFLOW_NAME, MEAN_MOBILENET_HEIGHT_WORKFLOW_VERSION)

    artifacts = sm['artifacts']
    artifact_order_mapping = {a['id']: (scan_id, a['order']) for a in artifacts}
    depth_artifact_ids = [a['id'] for a in artifacts if a['format'] in depth_format]
    results = sm['results']
    scan_type = sm['type']
    filtered_data, results_workflow_dict = filter_results(results, artifact_order_mapping, filter_functions_dict)
    rd, _ = get_mean_results(scan_id, filtered_data, results_workflow_dict, pcnn_height_workflow['id'], pcnn_height_mean_workflow['id'], 'height', depth_artifact_ids)
    result_dicts.append(rd)
    rd, _ = get_mean_results(scan_id, filtered_data, results_workflow_dict, pcnn_weight_workflow['id'], pcnn_weight_mean_workflow['id'], 'weight', depth_artifact_ids)
    result_dicts.append(rd)
    rd, scan_pred = get_mean_results(scan_id, filtered_data, results_workflow_dict, mn_height_workflow['id'], mn_height_mean_workflow['id'], 'height', depth_artifact_ids)
    result_dicts.append(rd)
    return result_dicts, scan_pred, sm['child_visit_id']


async def generate_new_visualizations(scan_id):
    try:
        cgm_api = CgmApi()
        session = aiohttp.ClientSession()
        workflows = await cgm_api.get_workflows()
        sm = await cgm_api.get_scan_metadata(scan_id)
        rgb_overlay_workflow = get_workflow(workflows, RGB_OVERLAY_WORKFLOW_NAME, RGB_OVERLAY_WORKFLOW_VERSION)
        in_depth_img_workflow = get_workflow(workflows, IN_DEPTH_IMG_WORKFLOW_NAME, IN_DEPTH_IMG_WORKFLOW_VERSION)
        artifacts = sm['artifacts']
        results = sm['results']
        results_workflow_dict = {}
        for r in results:
            k = r['workflow']
            if k not in results_workflow_dict:
                results_workflow_dict[k] = []
            results_workflow_dict[k].extend(r['source_artifacts'])
        scan_type = sm['type']
        if scan_type in STANDING_SCAN_TYPE:
            child_position = STANDING_TYPE
            scan_type = STANDING_TYPE
        elif scan_type in LAYING_SCAN_TYPE:
            child_position = LYING_TYPE
            scan_type = LYING_TYPE
        else:
            raise "Invalid scan type"

        await download_artifacts(cgm_api, artifacts)
        logging.info("finished downloading artifacts")
        depth_artifacts = await asyncify(get_scan_by_format)(artifacts, depth_format)
        rgb_artifacts = await asyncify(get_scan_by_format)(artifacts, rgb_format)
        blur_workflow = get_workflow(workflows, BLUR_WORKFLOW_NAME, BLUR_WORKFLOW_VERSION)
        blur_workflow_id = blur_workflow['id']
        blur_results = [r for r in results if r['workflow'] == blur_workflow_id]
        rgb_overlay_workflow = get_workflow(workflows, RGB_OVERLAY_WORKFLOW_NAME, RGB_OVERLAY_WORKFLOW_VERSION)
        rgb_overlay_results = results_workflow_dict[rgb_overlay_workflow['id']]
        in_depth_img_workflow = get_workflow(workflows, IN_DEPTH_IMG_WORKFLOW_NAME, IN_DEPTH_IMG_WORKFLOW_VERSION)
        in_depth_results = results_workflow_dict[in_depth_img_workflow['id']]
        depth_artifacts_by_order = {a['order']: a for a in depth_artifacts}
        rgb_artifacts_by_order = {a['order']: a for a in rgb_artifacts}
        results_li = []
        mn_depthmaps = []
        for i in range(1, 10):
            rgb_artifact = rgb_artifacts_by_order[i]
            depth_artifact = depth_artifacts_by_order[i]
            blue_file_id = [r['file'] for r in blur_results if r['source_artifacts'][0] == rgb_artifact['id']][0]
            rgb = load_rgb_image(rgb_artifact['raw_file'])
            depth = get_raw_depthmap(depth_artifact['raw_file'])
            raw_blur_artifact, status = await cgm_api.get_files(blue_file_id)
            blur_rgb = load_rgb_image(raw_blur_artifact)
            if child_position == "lying":
                max_depth = 1.5
            else:
                max_depth = 3.0
            depth = np.clip(depth, 0, max_depth)
            payload = pickle.dumps([rgb, child_position])
            out = await call_sam_api(session, payload)
            if child_position == "standing":
                child_mask, wall_mask, floor_mask = out
            else:
                child_mask, foot_mask, floor_mask = out
            if child_position == "lying":
                depth_inpainted = inpaint_depth_all_masks(depth, pose_type=child_position, child_mask=child_mask, floor_mask=floor_mask, foot_mask=foot_mask, max_depth=max_depth)
                overlaid_image_rgb = plot_with_masks_on_image(blur_rgb, None, floor_mask, child_mask, foot_mask=foot_mask, is_depth=False, is_standing=False)
            else:
                depth_inpainted = inpaint_depth_all_masks(depth, pose_type=child_position, child_mask=child_mask, floor_mask=floor_mask, wall_mask=wall_mask, max_depth=max_depth)
                overlaid_image_rgb = plot_with_masks_on_image(blur_rgb, wall_mask, floor_mask, child_mask, is_depth=False, is_standing=True)
            bin_file = save_plot_as_binary_new(depth_inpainted, scan_type)
            depth_inpainted = np.expand_dims(depth_inpainted, axis=2)
            depth_inpainted = depth_inpainted / NORMALIZATION_VALUE
            mn_dmap = depth_inpainted.copy()
            if mn_dmap.shape[:2] != (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH):
                mn_dmap = tf.image.resize(mn_dmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
            mn_dmap.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))
            mn_depthmaps.append(mn_dmap)
            if depth_artifact['id'] not in in_depth_results:
                in_depth_file_id, status = await cgm_api.post_files(bin_file, "rgb")
                results_li.append(get_result_dict(scan_id, in_depth_img_workflow['id'], source_artifacts=[depth_artifact['id']], file=in_depth_file_id))
            if rgb_artifact['id'] not in rgb_overlay_results:
                rgb_overlay_file_id, status = await cgm_api.post_files(overlaid_image_rgb, "rgb")
                results_li.append(get_result_dict(scan_id, rgb_overlay_workflow['id'], source_artifacts=[rgb_artifact['id']], file=rgb_overlay_file_id))
            if results_li:
                post_result_status_code = await cgm_api.post_results({"results": results_li})
                logging.info(f"post result status is {post_result_status_code}")
        mn_height_workflow = get_workflow(workflows, MOBILENET_HEIGHT_WORKFLOW_NAME, MOBILENET_HEIGHT_WORKFLOW_VERSION)
        mn_height_results = await call_mn_height(session, mn_depthmaps)
        mn_height_json_results_dicts = get_json_results(artifacts, mn_height_results, mn_height_workflow['id'], results.get(mn_height_workflow['id'], []), 'height')
        mn_height_post_status = await cgm_api.post_results({"results": mn_height_json_results_dicts})
        logging.info(f"Mobilenet pose status is {mn_height_post_status}")
    finally:
        await cgm_api.session.close()
        await session.close()
