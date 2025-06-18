import os
import json
from copy import deepcopy
import anyio
import asyncer
from asyncer import asyncify
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from utils.constants import *
from skimage.metrics import structural_similarity as ssim
from utils.file_decryption import decrypt_file_data


def get_scan_by_format(artifacts, file_format):
    return [artifact for artifact in artifacts if artifact['format'] in file_format]


async def download_artifacts(cgm_api, artifacts, scan_version):
    try:
        async with asyncer.create_task_group() as task_group:
            soon_values = [task_group.soonify(cgm_api.get_files)(artifact['file']) for artifact in artifacts]

    # 🔹 Ensure results are properly accessed after task_group exits
        results = [soon.value for soon in soon_values]
        for result, artifact in zip(results, artifacts):
            content, status = result
            if status == 200:
                if scan_version.startswith('ir-2.') or scan_version.startswith('v3.'):
                    artifact['raw_file'] = await asyncify(decrypt_file_data)(content)
                else:
                    artifact['raw_file'] = content
            else:
                content, status = await cgm_api.get_files(artifact['file'])
                if scan_version.startswith('ir-2.') or scan_version.startswith('v3.'):
                    artifact['raw_file'] = await asyncify(decrypt_file_data)(content)
                else:
                    artifact['raw_file'] = content
    except Exception as error:
        print(error)


def get_workflow(workflows, workflow_name, workflow_version):
    workflow = [w for w in workflows if w['name'] == workflow_name and w['version'] == workflow_version]
    
    return workflow[0]

def load_rgb_images(artifacts):
    input_images = {}
    for artifact in artifacts:
        image_rgb = np.asarray(Image.open(BytesIO(artifact['raw_file'])))
        image = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)

        input_images[artifact['id']] = image
    return input_images


def load_rgb_image(rgb_binary_file):
    image_rgb = np.asarray(Image.open(BytesIO(rgb_binary_file)))
    rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    return rgb


def encode_rgb_images(images):
    return [encode_image(img) for img in images.values()]

def encode_image(img):
    _, bin_file = cv2.imencode('.JPEG', img)
    bin_file = bin_file.tobytes()

    return bin_file

def blur_img(im,height,width,origin):
    #Create ROI
    h,w = height,width
    x,y = origin[0],origin[1]
    roi = im[int(y):int(y)+int(h),int(x):int(x)+int(w)]
    #Blur image in ROI
    blurred_img = cv2.GaussianBlur(roi,(91,91),0)
    #Add blur to the overall img
    im[int(y):int(y)+int(h),int(x):int(x)+int(w)] = blurred_img
    return im


def blur_face(input_image, detected_faces):
    for face in detected_faces:
        fr = face['faceRectangle']
        origin = (fr['left'], fr['top'])
        input_image = blur_img(input_image,fr['height'],fr['width'],origin)
    image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image[:, :, ::-1]

def blur_rgb_images(artifacts, face_results, images):
    blurred_images = {}
    blurred_images_to_post = {}
    for artifact, face_result in zip(artifacts, face_results):
        blurred_image = blur_face(images[artifact['id']], face_result)
        blurred_images[artifact['id']] = blurred_image
        _, bin_file = cv2.imencode('.JPEG', blurred_image)
        bin_file = bin_file.tostring()
        blurred_images_to_post[(artifact['scan_id'], artifact['id'])] = bin_file
    return blurred_images, blurred_images_to_post

def pose_and_blur_visualsation(artifacts, predictions, blurred_images):
    pose_vis = {}
    for (artifact, pose_prediction) in zip(artifacts, predictions):
        img = deepcopy(blurred_images[artifact['id']])
        no_of_pose_detected, pose_score, pose_result = pose_prediction[0]
        if no_of_pose_detected > 0:
            rotated_pose_preds = pose_result[0]['draw_kpt']
            for kpt in rotated_pose_preds:
                img = draw_pose(kpt, img)
        _, bin_file = cv2.imencode('.JPEG', img)
        bin_file = bin_file.tostring()
        pose_vis[(artifact['scan_id'], artifact['id'])] = bin_file
    return pose_vis


def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert (len(keypoints), len(keypoints[0])) == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        img = cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        img = cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        img = cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
    return img


def check_rgb_depth_alignment(rgb_image, depth_image, max_depth, threshold=0.75):
    """
    Checks the alignment of objects within an RGB image and a depth map using edge detection and SSIM.
    
    Parameters:
        rgb_image (numpy array): The RGB image.
        depth_image (numpy array): The depth map.
        max_depth (float): Maximum depth value (e.g., 3m or 1.5m).
        threshold (float): The SSIM threshold to determine alignment.
        - Currently set to 0.75 as the default.
    
    Returns:
        tuple: (Rounded SSIM score, 'Aligned' or 'Misaligned')
    """

    # Resize RGB to match Depth Map size
    depth_h, depth_w = depth_image.shape[:2]
    rgb_image = cv2.resize(rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)

    # Convert RGB image to grayscale
    rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Ensure depth values are within the specified max_depth range
    depth_image = np.clip(depth_image, 0, max_depth)

    # Convert depth map to 8-bit without normalization
    depth_8bit = (depth_image / max_depth * 255).astype(np.uint8)  # Scale based on max_depth

    # Apply Canny edge detection
    edges_rgb = cv2.Canny(rgb_gray, 50, 150)
    edges_depth = cv2.Canny(depth_8bit, 50, 150)

    # Compute SSIM between edge images
    similarity_index = ssim(edges_rgb, edges_depth)
    similarity_index = round(similarity_index, 2)  # Round SSIM to 2 decimal places

    # Determine alignment status
    alignment_status = "Aligned" if similarity_index >= threshold else "Misaligned"

    return similarity_index, alignment_status


def overlay_mask(image, mask, color, alpha=0.3):
    # Convert grayscale/depth to 3-channel if needed
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    overlay = image.copy()
    for c in range(3):
        overlay[..., c] = np.where(mask,
                                   (1 - alpha) * overlay[..., c] + alpha * color[c] * 255,
                                   overlay[..., c])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_with_masks_on_image(image, wall_mask, floor_mask, child_mask, foot_mask=None, is_depth=False, is_standing=True):
    # Normalize depth to [0, 255] if it's depthmap
    if is_depth and image.ndim == 2:
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
        image = image.astype(np.uint8)

    # Define light colors
    wall_color = (0.4, 0.7, 1.0)   # light blue
    floor_color = (0.6, 1.0, 0.6)  # light green
    child_color = (1.0, 0.7, 0.8)  # light pink
    foot_color = (0.9, 0.9, 0.1)   # yellow for foot
    blue = (0, 0, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    yellow = (255, 255, 0)

    # Start with the child and floor masks
    combined = overlay_mask(image, floor_mask, floor_color)
    combined = overlay_mask(combined, child_mask, child_color)

    # For standing child, overlay wall_mask, foot_mask is ignored
    if is_standing:
        combined = overlay_mask(combined, wall_mask, wall_color)
    
    # For lying child, overlay foot_mask (if available)
    elif foot_mask is not None:
        combined = overlay_mask(combined, foot_mask, foot_color)

    combined = add_label_cv(combined, child_mask, "Child", color=red)
    combined = add_label_cv(combined, floor_mask, "Floor", color=green)  # Green
    # if is_standing:
    #     combined = add_label_cv(combined, wall_mask, "Wall", color=blue)
    #     if foot_mask is not None:
    #         combined = add_label_cv(combined, foot_mask, "Foot", color=yellow)
    # else:
    #     if foot_mask is not None:
    #         combined = add_label_cv(combined, foot_mask, "Foot", color=yellow)
    combined_rotated = cv2.rotate(combined, cv2.ROTATE_90_COUNTERCLOCKWISE)
    encoded_image = encode_image(combined_rotated[:, :, ::-1])
    # Return the overlaid image
    return encoded_image


def add_label_cv(image, mask, label, color=(0, 0, 255), margin = 50):
    H, W = mask.shape[:2]
    pos = np.argwhere(mask)
    filtered_pos = pos[
        (pos[:, 0] > margin) & (pos[:, 0] < H - margin) &  # y-coordinate
        (pos[:, 1] > margin) & (pos[:, 1] < W - margin)    # x-coordinate
    ]
    if len(filtered_pos) > 0:
        y, x = filtered_pos[len(filtered_pos)//2]  # Get center point
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=2)

    return image
