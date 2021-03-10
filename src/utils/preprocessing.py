import zipfile

import numpy as np
import tensorflow as tf
from pyntcloud import PyntCloud
from skimage.transform import resize

IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180


def load_depth(filename):
    with zipfile.ZipFile(filename) as z:
        with z.open('data') as f:
            line = str(f.readline())[2:-3]
            header = line.split("_")
            res = header[0].split("x")
            # print(res)
            width = int(res[0])
            height = int(res[1])
            depthScale = float(header[1])
            max_confidence = float(header[2])
            data = f.read()
            f.close()
        z.close()
    return data, width, height, depthScale, max_confidence


def parse_depth(tx, ty, data, depthScale):
    depth = data[(int(ty) * WIDTH + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * WIDTH + int(tx)) * 3 + 1]
    depth *= depthScale
    return depth


def prepare_depthmap(data, width, height, depthScale):
    # prepare array for output
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            #             output[cx][height - cy - 1][0] = parse_confidence(cx, cy)
            #             output[cx][height - cy - 1][1] = im_array[cy][cx][1] / 255.0 #test matching on RGB data
            #             output[cx][height - cy - 1][2] = 1.0 - min(parse_depth(cx, cy) / 2.0, 1.0) #depth data scaled to be visible
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depthScale)
    return (
        np.array(
            output,
            dtype='float32').reshape(
            width,
            height),
        height,
        width)


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess(depthmap):
    # print(depthmap.dtype)
    depthmap = preprocess_depthmap(depthmap)
    # depthmap = depthmap/depthmap.max()
    depthmap = depthmap / 7.5
    depthmap = resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    # depthmap = depthmap[None, :]
    return depthmap


def parse_numbers(line):
    output = []
    values = line.split(" ")
    for value in values:
        output.append(float(value))
    return output

# parse calibration file


def parse_calibration(filepath):
    # global calibration
    with open(filepath, 'r') as file:
        calibration = []
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        # print(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy,
        # cx, cy
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        # print(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy,
        # cx, cy
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        # print(str(calibration[2]) + '\n') #depth camera position relativelly
        # to color camera in meters
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration


def parse_confidence(tx, ty, data, maxConfidence):
    return (data[(int(ty) * WIDTH + int(tx)) * 3 + 2]) / maxConfidence

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
def get_count(calibration, data, depthScale):
    count = 0
    for x in range(2, WIDTH - 2):
        for y in range(2, HEIGHT - 2):
            depth = parse_depth(x, y, data, depthScale)
            if depth:
                res = convert_2d_to_3d(calibration[1], x, y, depth)
                if res:
                    count = count + 1
    return count


def get_depthmaps(paths):
    depthmaps = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = load_depth(path)
        depthmap, height, width = prepare_depthmap(
            data, WIDTH, HEIGHT, depthScale)
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
