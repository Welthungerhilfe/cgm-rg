from io import BytesIO
from zipfile import ZipFile
import requests
from typing import Tuple
import cv2
from os import getenv
import logging

# import face_recognition
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from PIL import Image
import io

from utils.constants import STANDING_SCAN_TYPE, LAYING_SCAN_TYPE


IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
NORMALIZATION_VALUE = 7.5


def efficient_process_depthmaps(artifacts, cgm_api):
    """Load the list of depthmaps in scan as numpy array"""
    depthmaps = []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth(artifact['file'], cgm_api)
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = preprocess(depthmap)
        depthmaps.append(depthmap)
    return depthmaps


def process_depthmaps(artifacts, cgm_api):
    """Load the list of depthmaps in scan as numpy array"""
    depthmaps = []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth(artifact['file'], cgm_api)
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = preprocess(depthmap)
        depthmap = eval_preprocessing(depthmap)
        depthmaps.append(np.array(depthmap))
    depthmaps = np.array(depthmaps)
    return depthmaps


def mobilenet_process_depthmaps(artifacts, cgm_api):
    depthmaps = []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth(artifact['file'], cgm_api)
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = depthmap.astype("float32")
        depthmap = np.expand_dims(depthmap, axis=2)
        depthmaps.append(depthmap)
    depthmaps = np.array(depthmaps)
    return depthmaps


def eval_preprocessing(depthmap):
    depthmap = depthmap.astype("float32")
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = tf.image.resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))
    return depthmap


def load_depth(file_id: str, cgm_api) -> Tuple[bytes, int, int, float, float]:
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

    response = cgm_api.get_files(file_id)
    zipfile = ZipFile(BytesIO(response))
    with zipfile.open('data') as f:
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


def load_depth_from_file(raw_file) -> Tuple[bytes, int, int, float, float]:
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

    zipfile = ZipFile(BytesIO(raw_file))
    with zipfile.open('data') as f:
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


# def preprocess_image(image):
#     resize_image = cv2.resize(image, (IMAGE_TARGET_WIDTH, IMAGE_TARGET_HEIGHT))
#     resize_image = resize_image / 255
#     return resize_image

def pose_input(artifacts, scan_type):
    image_bgr_li = []
    for artifact in artifacts:
        image_rgb = np.asarray(Image.open(io.BytesIO(artifact['raw_file'])))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        shape = image_bgr.shape
        if scan_type in STANDING_SCAN_TYPE:
            rotated_image = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)  # Standing
        elif scan_type in LAYING_SCAN_TYPE:
            rotated_image = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Laying
        image_bgr_li.append(rotated_image)
    return image_bgr_li, shape


def blur_input(artifacts):
    image_in = []
    for artifact in artifacts:
        image_rgb = np.asarray(Image.open(io.BytesIO(artifact['raw_file'])))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_in.append(image_bgr)

    return image_in


def blur_input_face_api(raw_file, scan_type):
    image_rgb = np.asarray(Image.open(io.BytesIO(raw_file)))

    if scan_type in STANDING_SCAN_TYPE:
        image = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    elif scan_type in LAYING_SCAN_TYPE:
        image = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def standing_laying_data_preprocessing_tf_batch(artifacts):
    images = []
    for artifact in artifacts:
        # response = cgm_api.get_files(file_id)
        # img = tf.io.decode_raw(response)
        img = tf.image.decode_jpeg(artifact['raw_file'], channels=3)
        img = tf.cast(img, tf.float32) * (1. / 256)
        img = tf.image.rot90(img, k=3)
        img = tf.image.resize(img, [240, 180])
        images.append(img)
    
    return tf.convert_to_tensor(images)
