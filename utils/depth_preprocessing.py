from typing import Tuple
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from matplotlib import pyplot as plt
import math
import tensorflow as tf
from skimage.transform import resize
from skimage.restoration import inpaint
from utils.constants import STANDING_TYPE, LYING_TYPE
import cv2
import traceback
import logging


IMAGE_TARGET_HEIGHT= int(224)
IMAGE_TARGET_WIDTH= int(224)
NORMALIZATION_VALUE=3.0
PCC_IMAGE_TARGET_HEIGHT = 240
PCC_IMAGE_TARGET_WIDTH = 180
PCC_NORMALIZATION_VALUE = 7.5

IDENTITY_MATRIX_4D = [1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.]

def replace_values_above_threshold(depth_map, threshold):
    """
    Replace depth values in a depth map above a given threshold with the average
    of their four non-zero neighbors.

    Args:
    - depth_map (numpy.ndarray): The input depth map.
    - threshold (float): The threshold value.

    Returns:
    - numpy.ndarray: The depth map with values above the threshold replaced by
                    the average of their four non-zero neighbors.
    """
    # Ensure depth_map is a 3D array (240, 180, 1)
    # if depth_map.shape != (240, 180, 1):
    #     raise ValueError("Input depth_map should have a shape of (240, 180, 1)")

    # Create a binary mask for values above the threshold
    above_threshold_mask = depth_map > threshold

    # Get the indices of the values above the threshold
    above_threshold_indices = np.argwhere(above_threshold_mask)

    # Iterate over the indices and replace values with the average of neighbors
    for i, j, k in above_threshold_indices:
        neighbors = []
        if i > 0:
            neighbors.append(depth_map[i - 1, j, k])  
        if i < IMAGE_TARGET_HEIGHT - 1:
            neighbors.append(depth_map[i + 1, j, k])
        if j > 0:
            neighbors.append(depth_map[i, j - 1, k])  
        if j < IMAGE_TARGET_WIDTH - 1:
            neighbors.append(depth_map[i, j + 1, k])

        # Filter out zero values
        non_zero_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]

        if non_zero_neighbors:
            # Calculate the average of non-zero neighbors
            avg_neighbor_value = np.mean(non_zero_neighbors)
            depth_map[i, j, k] = avg_neighbor_value
        else:
            # If all neighbors are zero, set the value to zero
            depth_map[i, j, k] = 0

    return depth_map


def fill_zeros_inpainting(depth_map):
    """
    Fill zero values in a depth map using inpainting.

    Args:
        depth_map (numpy.ndarray): Input depth map with zero values.

    Returns:
        numpy.ndarray: Depth map with zero values filled using inpainting.
    """
    # Apply inpainting to fill in zero values using biharmonic interpolation
    depth_map_filled = inpaint.inpaint_biharmonic(depth_map, mask=(depth_map == 0))
    
    return depth_map_filled


def load_depth(response) -> Tuple[bytes, int, int, float, float]:
    """Take ZIP file and extract depth and metadata
    Args:
        fpath (str): File path to the ZIP
    Returns:
        depth_data (bytes): depthmap data
        width(int): depthmap width in pixel
        height(int): depthmap height in pixel
        depth_scale(float)
        max_confidence(float)
    """

    zipfile = ZipFile(BytesIO(response))
    with zipfile.open('data') as f:
        # Example for a first_line: '180x135_0.001_7_0.57045287_-0.0057296_0.0022602521_0.82130724_-0.059177425_0.0024800065_0.030834956'
        first_line = f.readline().decode().strip()
        width, height, depth_scale, max_confidence, device_pose = parse_header(first_line)
        depth_data = f.read()
    return depth_data, width, height, depth_scale, max_confidence, device_pose


def parse_depth(tx: int, ty: int, data: bytes, depth_scale: float, width: int) -> float:
    assert isinstance(tx, int)
    assert isinstance(ty, int)

    depth = data[(ty * width + tx) * 3 + 0] << 8
    depth += data[(ty * width + tx) * 3 + 1]

    depth *= depth_scale
    return depth


def prepare_depthmap(data: bytes, width: int, height: int, depth_scale: float) -> np.array:
    """Convert bytes array into np.array"""
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depth_scale, width)
    arr = np.array(output, dtype='float32')
    return arr.reshape(width, height)


def get_inpainted_depthmaps(artifacts):
    in_depthmaps = []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth(artifact['raw_file'])
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = depthmap.astype("float32")
        depthmap = np.expand_dims(depthmap, axis=2)
        in_depthmap = fill_zeros_inpainting(replace_values_above_threshold(depthmap, NORMALIZATION_VALUE))
        in_depthmap = in_depthmap / NORMALIZATION_VALUE
        if in_depthmap.shape[:2] != (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH):
            in_depthmap = tf.image.resize(in_depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
        in_depthmaps.append(in_depthmap)
    return in_depthmaps


def eval_preprocessing(depthmap):
    depthmap = depthmap.astype("float32")
    depthmap = depthmap / PCC_NORMALIZATION_VALUE
    depthmap = tf.image.resize(depthmap, (PCC_IMAGE_TARGET_HEIGHT, PCC_IMAGE_TARGET_WIDTH))
    depthmap.set_shape((PCC_IMAGE_TARGET_HEIGHT, PCC_IMAGE_TARGET_WIDTH, 1))
    return depthmap


def get_depthmaps_old(artifacts, scan_version):
    depthmaps, in_depthmaps = [], []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth(artifact['raw_file'])
        if 'ir' in scan_version:
            depthmap = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            depthmap = depthmap * depth_scale
            depthmap = np.rot90(depthmap, k=-1)
        else:
            depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = depthmap.astype("float32")
        depthmap = np.expand_dims(depthmap, axis=2)
        in_depthmap = depthmap.copy()
        in_depthmap = fill_zeros_inpainting(replace_values_above_threshold(in_depthmap, NORMALIZATION_VALUE))
        in_depthmap = in_depthmap / NORMALIZATION_VALUE
        if in_depthmap.shape[:2] != (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH):
            in_depthmap = tf.image.resize(in_depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
        in_depthmap.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))
        in_depthmaps.append(in_depthmap)
        ev_depthmap = eval_preprocessing(depthmap)
        depthmaps.append(np.array(ev_depthmap))
    return depthmaps, in_depthmaps


def get_depthmaps(artifacts, scan_version):
    depthmaps, in_depthmaps, pc_dmaps, mn_depthmaps, device_poses = [], [], [], [], []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence, device_pose = load_depth(artifact['raw_file'])
        if 'ir' in scan_version:
            depthmap = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            depthmap = depthmap * depth_scale
            depthmap = np.rot90(depthmap, k=-1)
        else:
            depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = np.expand_dims(depthmap, axis=2)
        depthmaps.append(depthmap)
        device_poses.append(device_pose)
        if 'ir' not in scan_version:
            in_depthmap = depthmap.copy()
            in_depthmap = fill_zeros_inpainting(replace_values_above_threshold(in_depthmap, NORMALIZATION_VALUE))
            in_depthmap = in_depthmap / NORMALIZATION_VALUE
            in_depthmaps.append(in_depthmap)
            mn_dmap = in_depthmap.copy()
            if mn_dmap.shape[:2] != (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH):
                mn_dmap = tf.image.resize(mn_dmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
            mn_dmap.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))
            mn_depthmaps.append(mn_dmap)
        else:
            in_depthmap = depthmap.copy()
            in_depthmap = tf.image.resize(in_depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
            in_depthmap = fill_zeros_inpainting(replace_values_above_threshold(np.array(in_depthmap), NORMALIZATION_VALUE))
            in_depthmap = in_depthmap / NORMALIZATION_VALUE
            in_depthmaps.append(in_depthmap)
            mn_dmap = in_depthmap.copy()
            mn_dmap = tf.image.resize(mn_dmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
            mn_dmap.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))
            mn_depthmaps.append(mn_dmap)
        pc_dmap = depthmap.copy()
        pc_dmap = eval_preprocessing(pc_dmap)
        pc_dmaps.append(np.array(pc_dmap))
    return depthmaps, in_depthmaps, pc_dmaps, mn_depthmaps, device_poses


def get_raw_depthmaps(artifacts, scan_version):
    depthmaps = []
    device_poses = []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence, device_pose = load_depth(artifact['raw_file'])
        if 'ir' in scan_version:
            depthmap = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            depthmap = depthmap * depth_scale
            depthmap = np.rot90(depthmap, k=-1)
        else:
            depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = np.expand_dims(depthmap, axis=2)
        depthmaps.append(depthmap)
        device_poses.append(device_pose)
    return depthmaps, device_poses


def get_raw_depthmap(depth_binary_file):
    data, width, height, depth_scale, _max_confidence, device_pose = load_depth(depth_binary_file)
    depthmap = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
    depthmap = depthmap * depth_scale
    depthmap = np.rot90(depthmap, k=-1)
    return depthmap


def compute_depth_metadata(depthmaps):
    no_of_zeroes = []
    for depthmap in depthmaps:
        no_of_zero = np.count_nonzero(depthmap == 0)
        depthmap_size = depthmap.size
        percentage_of_zero = no_of_zero * 100 / depthmap_size
        no_of_zeroes.append(percentage_of_zero)
    return no_of_zeroes


def save_plot_as_binary(depthmap, scan_type):
    if scan_type == STANDING_TYPE:
        plt.imshow(depthmap, cmap='jet', vmin=0, vmax=3)
    elif scan_type == LYING_TYPE:
        plt.imshow(depthmap, cmap='jet', vmin=0, vmax=1.5)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Move the buffer cursor to the beginning
    buffer.seek(0)

    # Get the binary data
    binary_data = buffer.read()

    return binary_data


def save_plot_as_binary_new(depthmap, scan_type):
    buffer = BytesIO()
    dpi = 100  # Adjust DPI to maintain 1280x720 resolution
    fig, ax = plt.subplots(figsize=(720 / dpi, 480 / dpi), dpi=dpi, frameon=False)
    rotated_depthmap = np.rot90(depthmap, k=1)
    if scan_type == STANDING_TYPE:
        ax.imshow(rotated_depthmap, cmap='jet', vmin=0, vmax=3)
    elif scan_type == LYING_TYPE:
        ax.imshow(rotated_depthmap, cmap='jet', vmin=0, vmax=1.5)

    # Remove all axes and margins
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Adjust layout to remove any whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the image
    plt.savefig(buffer, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    buffer.seek(0)
    # Get the binary data
    binary_data = buffer.read()

    return binary_data


def depth_visualization(artifacts, depthmaps, scan_type):
    depth_viz = {}
    for artifact, depthmap in zip(artifacts, depthmaps):
        bin_file = save_plot_as_binary_new(depthmap, scan_type)
        depth_viz[(artifact['scan_id'], artifact['id'])] = bin_file
    return depth_viz


def matrix_calculate(position: list[float], rotation: list[float]) -> list[float]:
    """Calculate a matrix image->world from device position and rotation"""

    output = IDENTITY_MATRIX_4D

    sqw = rotation[3] * rotation[3]
    sqx = rotation[0] * rotation[0]
    sqy = rotation[1] * rotation[1]
    sqz = rotation[2] * rotation[2]

    invs = 1 / (sqx + sqy + sqz + sqw)
    output[0] = (sqx - sqy - sqz + sqw) * invs
    output[5] = (-sqx + sqy - sqz + sqw) * invs
    output[10] = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = rotation[0] * rotation[1]
    tmp2 = rotation[2] * rotation[3]
    output[1] = 2.0 * (tmp1 + tmp2) * invs
    output[4] = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = rotation[0] * rotation[2]
    tmp2 = rotation[1] * rotation[3]
    output[2] = 2.0 * (tmp1 - tmp2) * invs
    output[8] = 2.0 * (tmp1 + tmp2) * invs

    tmp1 = rotation[1] * rotation[2]
    tmp2 = rotation[0] * rotation[3]
    output[6] = 2.0 * (tmp1 + tmp2) * invs
    output[9] = 2.0 * (tmp1 - tmp2) * invs

    output[12] = -position[0]
    output[13] = -position[1]
    output[14] = -position[2]
    return output


def matrix_transform_point(point: np.ndarray, device_pose_arr: np.ndarray) -> np.ndarray:
    """Transformation of point by device pose matrix

    point(np.array of float): 3D point
    device_pose: flattened 4x4 matrix

    Returns:
        3D point(np.array of float)
    """
    point_4d = np.append(point, 1.)
    output = np.matmul(device_pose_arr, point_4d)
    output[0:2] = output[0:2] / abs(output[3])
    return output[0:-1]


def get_angle_between_camera_and_floor(device_pose_arr) -> float:
    """Calculate an angle between camera and floor based on device pose

    The angle is often a negative values because the phone is pointing down.

    Angle examples:
    angle=-90deg: The phone's camera is fully facing the floor
    angle=0deg: The horizon is in the center
    angle=90deg: The phone's camera is facing straight up to the sky.
    """
    forward = matrix_transform_point([0, 0, 1], device_pose_arr)
    camera = matrix_transform_point([0, 0, 0], device_pose_arr)
    return math.degrees(math.asin(camera[1] - forward[1]))


def parse_header(header_line: str) -> Tuple:
    header_parts = header_line.split('_')
    res = header_parts[0].split('x')
    width = int(res[0])
    height = int(res[1])
    depth_scale = float(header_parts[1])
    max_confidence = float(header_parts[2])
    if len(header_parts) >= 10:
        position = (float(header_parts[7]), float(header_parts[8]), float(header_parts[9]))
        rotation = (float(header_parts[3]), float(header_parts[4]),
                    float(header_parts[5]), float(header_parts[6]))
        if position == (0., 0., 0.):
            device_pose = None
        else:
            device_pose = matrix_calculate(position, rotation)
    else:
        device_pose = IDENTITY_MATRIX_4D
    return width, height, depth_scale, max_confidence, device_pose


def compute_angle(device_poses):
    angles = []
    for device_pose in device_poses:
        if device_pose:
            device_pose_arr = np.array(device_pose).reshape(4, 4).T
            angles.append(get_angle_between_camera_and_floor(device_pose_arr))
        else:
            angles.append('NA')
    return angles


def inpaint_depth_by_rowwise_mean(depth, mask, max_depth=3.0):
    """
    Inpaint missing values in depth using row-wise IQR mean.
    - If >10% of row is missing, sample randomly in (mean-0.1, mean+0.1).
    - Otherwise, use normal IQR mean.
    - If the entire row is missing, check adjacent rows (left or right) for valid depth and adjust.
    """
    inpainted = depth.copy()
    height, width = depth.shape

    mask_valid = (mask > 0)
    depth_valid = (depth > 0) & (depth <= max_depth)
    overall_valid_mask = mask_valid & depth_valid
    overall_mask_mean = depth[overall_valid_mask].mean() if np.any(overall_valid_mask) else 0

    rng = np.random.default_rng()

    for row in range(height):
        row_mask = mask_valid[row, :]
        row_depth = depth[row, :]
        valid_row = row_mask & (row_depth > 0) & (row_depth <= max_depth)
        
        missing_row = row_mask & ((row_depth == 0) | (row_depth > max_depth))
        missing_ratio = np.sum(missing_row) / np.sum(row_mask) if np.sum(row_mask) > 0 else 1.0

        if np.any(valid_row):
            row_valid_depths = row_depth[valid_row]
            q1 = np.percentile(row_valid_depths, 25)
            q3 = np.percentile(row_valid_depths, 75)
            iqr_values = row_valid_depths[(row_valid_depths >= q1) & (row_valid_depths <= q3)]
            row_mean = iqr_values.mean() if len(iqr_values) > 0 else row_valid_depths.mean()

            if missing_ratio > 0.1:
                # Sample randomly around the overall_mask_mean
                low = overall_mask_mean - 0.1
                high = overall_mask_mean + 0.1
                sampled_values = rng.uniform(low, high, size=np.sum(missing_row))
            else:
                sampled_values = np.full(np.sum(missing_row), row_mean)
        else:
            # Entire row invalid — check adjacent rows (left or right)
            above = row - 1 if row > 0 else None
            below = row + 1 if row < height - 1 else None
            means = []
            if above is not None:
                valid_above = mask_valid[above, :] & (depth[above, :] > 0) & (depth[above, :] < max_depth)
                if np.any(valid_above):
                    row_valid_above = depth[above, :][valid_above]
                    q1_above = np.percentile(row_valid_above, 25)
                    q3_above = np.percentile(row_valid_above, 75)
                    iqr_above = row_valid_above[(row_valid_above >= q1_above) & (row_valid_above <= q3_above)]
                    means.append(iqr_above.mean() if len(iqr_above) > 0 else row_valid_above.mean())
            if below is not None:
                valid_below = mask_valid[below, :] & (depth[below, :] > 0) & (depth[below, :] < max_depth)
                if np.any(valid_below):
                    row_valid_below = depth[below, :][valid_below]
                    q1_below = np.percentile(row_valid_below, 25)
                    q3_below = np.percentile(row_valid_below, 75)
                    iqr_below = row_valid_below[(row_valid_below >= q1_below) & (row_valid_below <= q3_below)]
                    means.append(iqr_below.mean() if len(iqr_below) > 0 else row_valid_below.mean())

            if means:
                row_mean = np.mean(means)
            else:
                row_mean = overall_mask_mean

            sampled_values = rng.uniform(row_mean - 0.1, row_mean + 0.1, size=np.sum(missing_row))

        missing_indices = np.where(missing_row)[0]
        inpainted[row, missing_indices] = sampled_values

    return inpainted


def inpaint_depth_by_columnwise_mean(depth, mask, max_depth=1.5):
    """
    Inpaint missing values in depth using column-wise IQR mean for lying children.
    - If >10% of column is missing, sample randomly in (mean-0.05, mean+0.05).
    - Otherwise, use normal IQR mean.
    - If the entire column is missing, check adjacent columns (left or right) for valid depth and adjust.
    """
    inpainted = depth.copy()
    height, width = depth.shape

    mask_valid = (mask > 0)
    depth_valid = (depth > 0) & (depth <= max_depth)
    overall_valid_mask = mask_valid & depth_valid
    overall_mask_mean = depth[overall_valid_mask].mean() if np.any(overall_valid_mask) else 0

    rng = np.random.default_rng()

    for col in range(width):
        col_mask = mask_valid[:, col]
        col_depth = depth[:, col]
        valid_col = col_mask & (col_depth > 0) & (col_depth <= max_depth)

        missing_col = col_mask & ((col_depth == 0) | (col_depth > max_depth))
        missing_ratio = np.sum(missing_col) / np.sum(col_mask) if np.sum(col_mask) > 0 else 1.0

        if np.any(valid_col):
            col_valid_depths = col_depth[valid_col]
            q1 = np.percentile(col_valid_depths, 25)
            q3 = np.percentile(col_valid_depths, 75)
            iqr_values = col_valid_depths[(col_valid_depths >= q1) & (col_valid_depths <= q3)]
            col_mean = iqr_values.mean() if len(iqr_values) > 0 else col_valid_depths.mean()

            if missing_ratio > 0.1:
                # Sample randomly around the overall_mask_mean
                low = overall_mask_mean - 0.05
                high = overall_mask_mean + 0.05
                sampled_values = rng.uniform(low, high, size=np.sum(missing_col))
            else:
                sampled_values = np.full(np.sum(missing_col), col_mean)
        else:
            # Entire column invalid — check adjacent columns (left or right)
            left = col - 1 if col > 0 else None
            right = col + 1 if col < width - 1 else None
            means = []
            if left is not None:
                valid_left = mask_valid[:, left] & (depth[:, left] > 0) & (depth[:, left] < max_depth)
                if np.any(valid_left):
                    left_depths = depth[:, left][valid_left]
                    q1_left = np.percentile(left_depths, 25)
                    q3_left = np.percentile(left_depths, 75)
                    iqr_left = left_depths[(left_depths >= q1_left) & (left_depths <= q3_left)]
                    means.append(iqr_left.mean() if len(iqr_left) > 0 else left_depths.mean())
            if right is not None:
                valid_right = mask_valid[:, right] & (depth[:, right] > 0) & (depth[:, right] < max_depth)
                if np.any(valid_right):
                    right_depths = depth[:, right][valid_right]
                    q1_right = np.percentile(right_depths, 25)
                    q3_right = np.percentile(right_depths, 75)
                    iqr_right = right_depths[(right_depths >= q1_right) & (right_depths <= q3_right)]
                    means.append(iqr_right.mean() if len(iqr_right) > 0 else right_depths.mean())

            if means:
                col_mean = np.mean(means)
            else:
                col_mean = overall_mask_mean

            sampled_values = rng.uniform(col_mean - 0.05, col_mean + 0.05, size=np.sum(missing_col))

        missing_indices = np.where(missing_col)[0]
        inpainted[missing_indices, col] = sampled_values

    return inpainted


def inpaint_depth_by_interpolation(depth, child_mask, max_depth):
    """
    Inpaint depth values using interpolation within the child mask area.

    Parameters:
        depth (np.ndarray): The depth map to inpaint.
        child_mask (np.ndarray): A binary mask where non-zero values represent the region to inpaint.
        max_depth (float): Maximum depth value to constrain inpainting.

    Returns:
        np.ndarray: Inpainted depth map.
    """
    # Ensure the mask is a binary mask (0 or 255)
    child_mask = np.uint8(child_mask)
    
    # Create the inverse mask (non-child areas)
    inverse_mask = cv2.bitwise_not(child_mask)

    # Inpainting using OpenCV's inpainting function (method = 1 for inpainting using telea)
    inpainted_depth = cv2.inpaint(depth, child_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Apply max depth constraint (ensuring the depth values do not exceed max_depth)
    inpainted_depth = np.minimum(inpainted_depth, max_depth)
    
    return inpainted_depth

def inpaint_depth_all_masks(depth, pose_type, child_mask, floor_mask=None, wall_mask=None, foot_mask=None, max_depth=3.0, foot_area_threshold=0.3):
    """
    Generalized depth inpainting function for both lying and standing children.
    """
    # Calculate percentage of missing values in child mask
    child_mask_missing_percentage = np.sum(child_mask == 0) / child_mask.size * 100

    if pose_type == "standing":

        # If child mask has more than 10% missing values, perform row-wise inpainting; otherwise, use interpolation
        if child_mask_missing_percentage > 10:
            depth = inpaint_depth_by_rowwise_mean(depth, child_mask, max_depth)
        else:
            depth = inpaint_depth_by_interpolation(depth, child_mask, max_depth)
        # Inpainting for standing children: Use row-wise mean for floor_mask, and wall_mask
        if floor_mask is not None:
            depth = inpaint_depth_by_rowwise_mean(depth, floor_mask, max_depth)
        if wall_mask is not None:
            depth = inpaint_depth_by_rowwise_mean(depth, wall_mask, max_depth)

    elif pose_type == "lying":

        # If child mask has more than 10% missing values, perform column-wise inpainting; otherwise, use interpolation
        if child_mask_missing_percentage > 10:
            depth = inpaint_depth_by_columnwise_mean(depth, child_mask, max_depth)
        else:
            depth = inpaint_depth_by_interpolation(depth, child_mask, max_depth)
        # Inpainting for lying children: Use column-wise mean for child_mask and floor_mask
        if floor_mask is not None:
            depth = inpaint_depth_by_columnwise_mean(depth, floor_mask, max_depth=max_depth)
        # Foot region inpainting for lying children using floor_mask column mean
        if foot_mask is not None:
            foot_mask = (foot_mask == 1).astype(np.uint8)
            foot_area = np.sum(foot_mask)
            # Check if foot area is smaller than a threshold
            child_area = np.sum(child_mask)
            if foot_area < foot_area_threshold * child_area:
                # Inpaint using the floor mask (foot region is too small)
                depth[foot_mask == 1] = 0
                remaining_mask = 1 - child_mask
                depth = inpaint_depth_by_columnwise_mean(depth, remaining_mask, max_depth=max_depth)
                print("foot inpaint")
            else:
                # If foot area is too large, return None (image not usable)
                print("foot inpaint not possible")
                return None  # Image not usable due to too much overlap.
    return depth