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
from rg.results_utils import get_mean_result, get_mean_results, filter_results, pose_score_filter, depth_feature_filter, distance_to_child_filter, get_rgb_depth_allignment_result, get_result_dict, get_json_results, get_mediapipe_distance_result
from utils.inference import call_sam_api, call_mn_height
from utils.depth_preprocessing import get_raw_depthmap, inpaint_depth_all_masks, save_plot_as_binary_new, IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, NORMALIZATION_VALUE
from utils.mediapipe_distance_calculation import estimate_distance_from_segmentation, estimate_distance_from_pose
from utils.file_decryption import decrypt_xor_response_content, decrypt_aes_response_content


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
        for artifact in artifacts:
            if 'raw_file' not in artifact:
                logging.info(f"no raw file")
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
        mediapipe_seg_distance_workflow = get_workflow(workflows, MEDIAPIPE_SEG_DISTANCE_WORKFLOW_NAME, MEDIAPIPE_SEG_DISTANCE_WORKFLOW_VERSION)
        medispipe_pose_distance_workflow = get_workflow(workflows, MEDIAPIPE_POSE_DISTANCE_WORKFLOW_NAME, MEDIAPIPE_POSE_DISTANCE_WORKFLOW_VERSION)
        rgb_artifact_to_ord_mapping = {(ra['scan_id'], ra['order']): ra['id'] for ra in rgb_artifacts}
        if scan_type == STANDING_TYPE:
            max_depth = 3
        elif scan_type == LYING_TYPE:
            max_depth = 1.5
        alignment_rds = []
        for da, dm in zip(depth_artifacts, depthmaps):
            rgb_a_id = rgb_artifact_to_ord_mapping.get((da['scan_id'], da['order']), None)
            if rgb_a_id is None:
                logging.warning(f"No RGB artifact found for depth artifact {da['order']} in scan {da['scan_id']}")
                continue
            similarity_index, alignment_status = check_rgb_depth_alignment(rgb_images[rgb_a_id], dm, max_depth, threshold=0.75)
            alignment_rds.append(get_rgb_depth_allignment_result(da['scan_id'], [da['id']], allignment_workflow['id'], similarity_index, alignment_status))
            distance, center, _ = estimate_distance_from_segmentation(rgb_images[rgb_a_id], dm)
            if distance is not None:
                alignment_rds.append(get_mediapipe_distance_result(da['scan_id'], [da['id']], mediapipe_seg_distance_workflow['id'], distance, center))
            else:
                alignment_rds.append(get_mediapipe_distance_result(da['scan_id'], [da['id']], mediapipe_seg_distance_workflow['id'], 'NA', 
                                                                   'NA'))
                logging.warning(f"Distance estimation failed for scan {da['scan_id']} at order {da['order']}")
            distance, center = estimate_distance_from_pose(rgb_images[rgb_a_id], dm)
            if distance is not None:
                alignment_rds.append(get_mediapipe_distance_result(da['scan_id'], [da['id']], medispipe_pose_distance_workflow['id'], distance, center))
            else:
                alignment_rds.append(get_mediapipe_distance_result(da['scan_id'], [da['id']], medispipe_pose_distance_workflow['id'], 'NA', 
                                                                   'NA'))
                logging.warning(f"Pose distance estimation failed for scan {da['scan_id']} at order {da['order']}")
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


def check_rgb_depth_alignment_by_overlap(
    rgb_image, depth_image, child_mask, max_depth=3.0,
    edge_thresh=(30, 180), overlap_thresh=0.15, show_edges=True,
    missing_thresh=0.10
):
    depth_h, depth_w = depth_image.shape[:2]
    rgb_image = cv2.resize(rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
    child_mask = cv2.resize(child_mask.astype(np.uint8), (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

    regions = {
        "upper": np.zeros_like(child_mask),
        "lower": np.zeros_like(child_mask),
        "left": np.zeros_like(child_mask),
        "right": np.zeros_like(child_mask)
    }

    mid_h, mid_w = depth_h // 2, depth_w // 2
    regions["upper"][:mid_h, :] = 1
    regions["lower"][mid_h:, :] = 1
    regions["left"][:, :mid_w] = 1
    regions["right"][:, mid_w:] = 1

    insufficient_regions = []
    valid_mask = (depth_image > 0).astype(np.uint8) * child_mask

    for name, region_mask in regions.items():
        region_mask = region_mask.astype(np.uint8) * child_mask
        total = region_mask.sum()
        if total == 0:
            continue
        valid = (valid_mask * region_mask).sum()
        missing_ratio = 1 - valid / (total + 1e-6)
        if missing_ratio > missing_thresh:
            insufficient_regions.append(name)

    if len(insufficient_regions) == 4:
        return None, f"Insufficient Depth in All Regions"

    # Build final mask using only sufficient regions
    combined_region = np.zeros_like(child_mask)
    for name, region_mask in regions.items():
        if name not in insufficient_regions:
            combined_region = np.logical_or(combined_region, region_mask * child_mask)

    combined_region = combined_region.astype(np.uint8)

    # Get child outline
    child_outline = get_child_outline(combined_region)

    # Convert RGB to grayscale
    rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Normalize and convert depth to 8-bit
    depth_image = np.clip(depth_image, 0, max_depth)
    depth_8bit = (depth_image / max_depth * 255).astype(np.uint8)

    # Edge detection
    edges_rgb = cv2.Canny(rgb_gray, *edge_thresh)
    edges_depth = cv2.Canny(depth_8bit, *edge_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_rgb_thick = cv2.dilate(edges_rgb, kernel, iterations=5)
    edges_depth_thick = cv2.dilate(edges_depth, kernel, iterations=5)

    rgb_outline_edges = cv2.bitwise_and(edges_rgb_thick, edges_rgb_thick, mask=child_outline)
    depth_outline_edges = fill_missing_depth_edges(edges_depth_thick, child_outline)

    intersection = np.logical_and(rgb_outline_edges > 0, depth_outline_edges > 0).sum()
    denominator = (rgb_outline_edges > 0).sum()
    overlap_ratio = intersection / (denominator + 1e-6)
    overlap_ratio = round(overlap_ratio, 2)
    alignment_status = "Aligned" if overlap_ratio >= overlap_thresh else "Misaligned"