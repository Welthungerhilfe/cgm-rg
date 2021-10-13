import zipfile
from typing import Tuple
import cv2

import numpy as np
import tensorflow as tf
from skimage.transform import resize

IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
NORMALIZATION_VALUE = 7.5

standing_scan_type = ["101", "102", "103"]
laying_scan_type = ["201", "202", "203"]


def process_depthmaps(artifacts, scan_directory, result_generation):
    """Load the list of depthmaps in scan as numpy array"""
    depthmaps = []
    for artifact in artifacts:
        input_path = result_generation.get_input_path(scan_directory, artifact['file'])
        data, width, height, depth_scale, _max_confidence = load_depth(input_path)
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = preprocess(depthmap)
        depthmaps.append(depthmap)
    depthmaps = np.array(depthmaps)
    return depthmaps


def load_depth(fpath: str) -> Tuple[bytes, int, int, float, float]:
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

    with zipfile.ZipFile(fpath) as z:
        with z.open('data') as f:
            # Example for a first_line: '180x135_0.001_7_0.57045287_-0.0057296_0.0022602521_0.82130724_-0.059177425_0.0024800065_0.030834956'
            first_line = f.readline().decode().strip()

            file_header = first_line.split("_")

            # header[0] example: 180x135
            width, height = file_header[0].split("x")
            width, height = int(width), int(height)
            depth_scale = float(file_header[1])
            max_confidence = float(file_header[2])

            depth_data = f.read()
    return depth_data, width, height, depth_scale, max_confidence


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


def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")


def preprocess(depthmap):
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    return depthmap


def preprocess_image(image):
    resize_image = cv2.resize(image, (IMAGE_TARGET_WIDTH, IMAGE_TARGET_HEIGHT))
    resize_image = resize_image / 255
    return resize_image


def get_depthmaps(fpaths):
    depthmaps = []
    for fpath in fpaths:
        data, width, height, depth_scale, _ = load_depth(fpath)
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = preprocess(depthmap)
        depthmaps.append(depthmap)

    depthmaps = np.array(depthmaps)
    return depthmaps


def standing_laying_data_preprocessing(source_path, scan_type):
    img = tf.io.read_file(str(source_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) * (1. / 256)
    if scan_type is standing_scan_type:
        img = tf.image.rot90(img, k=3)
    elif scan_type is standing_scan_type:
        img = tf.image.rot90(img, k=1)
    img = tf.image.resize(img, [240, 180])
    img = tf.expand_dims(img, axis=0)
    return img


def sample_systematic_from_artifacts(artifacts: list, n_artifacts: int) -> list:
    """
    Code reference from cgm-ml
    https://github.com/Welthungerhilfe/cgm-ml/blob/main/src/common/model_utils/preprocessing_multiartifact_python.py#L89
    """
    n_artifacts_total = len(artifacts)
    n_skip = n_artifacts_total // n_artifacts  # 20 / 5 = 4
    indexes_to_select = list(
        range(n_skip // 2, n_artifacts_total, n_skip))[:n_artifacts]
    selected_artifacts = [artifacts[i] for i in indexes_to_select]
    assert len(selected_artifacts) == n_artifacts, str(artifacts)
    return selected_artifacts


def find_corresponding_image(image_order_ids, depth_id):
    """
    Code to find corresponding image for the given depthmap on the basis of order
    """
    closest_order_id = min(image_order_ids, key=lambda order: abs(order - depth_id))
    return closest_order_id
