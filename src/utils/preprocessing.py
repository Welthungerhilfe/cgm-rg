import zipfile

import numpy as np
import tensorflow as tf
from skimage.transform import resize

IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
NORMALIZATION_VALUE = 7.5


def load_depth(filename):
    with zipfile.ZipFile(filename) as z:
        with z.open('data') as f:
            line = str(f.readline())[2:-3]
            header = line.split("_")

            # header[0] example: 180x135
            width, height = header[0].split("x")

            depth_scale = float(header[1])
            max_confidence = float(header[2])
            data = f.read()
            f.close()
        z.close()
    return data, width, height, depth_scale, max_confidence


def parse_depth(tx, ty, data, depth_scale, width):
    depth = data[(int(ty) * width + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * width + int(tx)) * 3 + 1]
    depth *= depth_scale
    return depth


def prepare_depthmap(data, width, height, depth_scale):
    # prepare array for output
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depth_scale, width)
    arr = np.array(output, dtype='float32')
    return arr.reshape(width, height), height, width  # TODO don't return width and height


def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")


def preprocess(depthmap):
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    return depthmap


def parse_numbers(line):
    output = []
    values = line.split(" ")
    for value in values:
        output.append(float(value))
    return output


def parse_calibration(filepath):
    # global calibration
    with open(filepath, 'r') as f:
        calibration = []
        f.readline()[:-1]
        calibration.append(parse_numbers(f.readline()))
        # print(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy,
        # cx, cy
        f.readline()[:-1]
        calibration.append(parse_numbers(f.readline()))
        # print(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy,
        # cx, cy
        f.readline()[:-1]
        calibration.append(parse_numbers(f.readline()))
        # print(str(calibration[2]) + '\n') #depth camera position relativelly
        # to color camera in meters
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration


def parse_confidence(tx, ty, data, max_confidence):
    return (data[(int(ty) * WIDTH + int(tx)) * 3 + 2]) / max_confidence

# getter


def get_width():
    return WIDTH

# getter


def get_height():
    return HEIGHT

# setter


def set_width(value):
    global WIDTH
    WIDTH = value

# setter


def set_height(value):
    global HEIGHT
    HEIGHT = value

    # parse PCD


# get valid points in depthmaps
def get_count(calibration, data, depth_scale):
    count = 0
    for x in range(2, WIDTH - 2):
        for y in range(2, HEIGHT - 2):
            depth = parse_depth(x, y, data, depth_scale, WIDTH)
            if depth:
                res = convert_2d_to_3d(calibration[1], x, y, depth)
                if res:
                    count = count + 1
    return count


def convert_2d_to_3d(intrisics, x, y, z):
    # print(intrisics)
    fx = intrisics[0] * float(WIDTH)
    fy = intrisics[1] * float(HEIGHT)
    cx = intrisics[2] * float(WIDTH)
    cy = intrisics[3] * float(HEIGHT)
    tx = (x - cx) * z / fx
    ty = (y - cy) * z / fy
    output = []
    output.append(tx)
    output.append(ty)
    output.append(z)
    return output


def get_depthmaps(paths):
    depthmaps = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = load_depth(path)
        depthmap, _, _ = prepare_depthmap(data, width, height, depthScale)
        # print(height, width)
        depthmap = preprocess(depthmap)
        # print(depthmap.shape)
        depthmaps.append(depthmap)

    depthmaps = np.array(depthmaps)

    return depthmaps


def standing_laying_data_preprocessing(source_path):
    img = tf.io.read_file(source_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) * (1. / 256)
    img = tf.image.rot90(img, k=3)
    img = tf.image.resize(img, [240, 180])
    img = tf.expand_dims(img, axis=0)
    return img


def sample_systematic_from_artifacts(artifacts: list, n_artifacts: int) -> list:
    '''
    Code reference from cgm-ml
    https://github.com/Welthungerhilfe/cgm-ml/blob/main/src/common/model_utils/preprocessing_multiartifact_python.py#L89
    '''
    n_artifacts_total = len(artifacts)
    n_skip = n_artifacts_total // n_artifacts  # 20 / 5 = 4
    indexes_to_select = list(
        range(n_skip // 2, n_artifacts_total, n_skip))[:n_artifacts]
    selected_artifacts = [artifacts[i] for i in indexes_to_select]
    assert len(selected_artifacts) == n_artifacts, str(artifacts)
    return selected_artifacts
