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


IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
NORMALIZATION_VALUE = 7.5

STANDING_SCAN_TYPE = ["100", "101", "102"]
LAYING_SCAN_TYPE = ["200", "201", "202"]


def process_depthmaps(artifacts, ml_api):
    """Load the list of depthmaps in scan as numpy array"""
    depthmaps = []
    for artifact in artifacts:
        data, width, height, depth_scale, _max_confidence = load_depth(artifact['file'], ml_api)
        depthmap = prepare_depthmap(data, width, height, depth_scale)
        depthmap = preprocess(depthmap)
        depthmap = eval_preprocessing(depthmap)
        depthmaps.append(depthmap)
    depthmaps = np.array(depthmaps)
    return depthmaps


def eval_preprocessing(depthmap):
    depthmap = depthmap.astype("float32")
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = tf.image.resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))
    return depthmap


def load_depth(file_id: str, ml_api) -> Tuple[bytes, int, int, float, float]:
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

    response = ml_api.get_files(file_id)
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


def orient_img(image, scan_type):
    # The images are rotated 90 degree clockwise for standing children
    # and 90 degree anticlock wise for laying children to make children
    # head at top and toe at bottom
    if scan_type in STANDING_SCAN_TYPE:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif scan_type in LAYING_SCAN_TYPE:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def standing_laying_data_preprocessing(file_id, scan_type, ml_api):
    response = ml_api.get_files(file_id)
    rgb_image = np.asarray(Image.open(io.BytesIO(response)))
    img = orient_img(rgb_image, scan_type)

    return img


def standing_laying_data_preprocessing_tf(file_id, ml_api):
    response = ml_api.get_files(file_id)
    # img = tf.io.decode_raw(response)
    img = tf.image.decode_jpeg(response, channels=3)
    img = tf.cast(img, tf.float32) * (1. / 256)
    img = tf.image.rot90(img, k=3)
    img = tf.image.resize(img, [240, 180])
    img = tf.expand_dims(img, axis=0)
    return img


def blur_img_transformation_using_scan_version_and_scan_type(rgb_image, scan_version, scan_type):
    if scan_version in ["v0.7"]:
        # Make the image smaller, The limit of cgm-api to post an image is 500 KB.
        # Some of the images of v0.7 is greater than 500 KB
        rgb_image = cv2.resize(
            rgb_image, (0, 0), fx=1.0 / 1.3, fy=1.0 / 1.3)

    # print("scan_version is ", self.scan_version)
    image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

    # if self.scan_version in ["v0.1", "v0.2", "v0.4", "v0.5", "v0.6", "v0.7", "v0.8", "v0.9", "v1.0"]:
    # The images are provided in 90degrees turned. Here we rotate 90 degress to
    # the right.
    if scan_type in STANDING_SCAN_TYPE:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif scan_type in LAYING_SCAN_TYPE:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reorient_back(image, scan_type):
    if scan_type in STANDING_SCAN_TYPE:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif scan_type in LAYING_SCAN_TYPE:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image


def blur_face(file_id: str, scan_version, scan_type, ml_api):
    """Run face blur on given source_path
    Returns:
        bool: True if blurred otherwise False
    """
    response = ml_api.get_files(file_id)
    rgb_image = np.asarray(Image.open(io.BytesIO(response)))

    image = blur_img_transformation_using_scan_version_and_scan_type(rgb_image, scan_version, scan_type)
    image = orient_img(image, scan_type)

    height, width, channels = image.shape
    logging.info(f"{height}, {width}, {channels}")

    resized_height = 500.0
    resize_factor = height / resized_height
    # resized_width = width / resize_factor
    # resized_height, resized_width = int(resized_height), int(resized_width)

    # Scale image down for faster prediction.
    small_image = cv2.resize(
        image, (0, 0), fx=1.0 / resize_factor, fy=1.0 / resize_factor)

    # Find face locations.
    face_locations = face_recognition.face_locations(small_image, model="cnn")

    faces_detected = len(face_locations)
    logging.info("%s %s", faces_detected, "face locations found and blurred for path:")

    # Blur the image.
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was
        # scaled to 1/4 size
        top *= resize_factor
        right *= resize_factor
        bottom *= resize_factor
        left *= resize_factor
        top, right, bottom, left = int(top), int(right), int(bottom), int(left)

        # Extract the region of the image that contains the face.
        # TODO rotate -codinate

        face_image = image[top:bottom, left:right]

        # Blur the face image.
        face_image = cv2.GaussianBlur(
            face_image, ksize=(99, 99), sigmaX=30)

        # Put the blurred face region back into the frame image.
        image[top:bottom, left:right] = face_image

    image = reorient_back(image, scan_type)

    # Write image to hard drive.
    rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

    # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
    logging.info("%s %s", len(face_locations), "face locations found and blurred for path:")
    return rgb_image, True, faces_detected
