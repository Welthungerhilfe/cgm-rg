import zipfile
import numpy as np
from pyntcloud import PyntCloud
from skimage.transform import resize

image_target_height = 240
image_target_width = 180


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
            maxConfidence = float(header[2])
            data = f.read()
            f.close()
        z.close()
    return data, width, height, depthScale, maxConfidence


def parseDepth(tx, ty, data, depthScale):
    depth = data[(int(ty) * width + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * width + int(tx)) * 3 + 1]
    depth *= depthScale
    return depth


def prepare_depthmap(data, width, height, depthScale):
    # prepare array for output
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            #             output[cx][height - cy - 1][0] = parseConfidence(cx, cy)
            #             output[cx][height - cy - 1][1] = im_array[cy][cx][1] / 255.0 #test matching on RGB data
            #             output[cx][height - cy - 1][2] = 1.0 - min(parseDepth(cx, cy) / 2.0, 1.0) #depth data scaled to be visible
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parseDepth(cx, cy, data, depthScale)
    return (
        np.array(
            output,
            dtype='float32').reshape(
            width,
            height),
        height,
        width)


# write obj
def getPCD(filename, calibration, data, maxConfidence, depthScale):
    pcd = []
    # count = str(getCount(calibration, data, depthScale))
    # print(count)
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            depth = parseDepth(x, y, data, depthScale)
            if depth:
                res = convert2Dto3D(calibration[1], x, y, depth)
                if res:
                    # file.write(str(-res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + ' ' + str(parseConfidence(x, y)) + '\n')
                    pcd.append([-res[0], res[1], res[2],
                                parseConfidence(x, y, data, maxConfidence)])

    return np.array(pcd)


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess(depthmap):
    # print(depthmap.dtype)
    depthmap = preprocess_depthmap(depthmap)
    # depthmap = depthmap/depthmap.max()
    depthmap = depthmap / 7.5
    depthmap = resize(depthmap, (image_target_height, image_target_width))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    # depthmap = depthmap[None, :]
    return depthmap


def lenovo_pcd2depth(pcd, calibration):
    try:
        points = parsePCD(pcd)
    except Exception as error:
        print(error)
        return None
    width = getWidth()
    height = getHeight()
    # print(height, width)
    output = np.zeros((width, height, 1))
    # print(calibration)
    for p in points:
        try:
            v = convert3Dto2D(calibration[1], p[0], p[1], p[2])
        except Exception as error:
            print(pcd, error)
        x = round(width - v[0] - 1)
        y = round(v[1])
        y = round(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y] = p[2]
    return output

# parse line of numbers


def parseNumbers(line):
    output = []
    values = line.split(" ")
    for value in values:
        output.append(float(value))
    return output

# parse calibration file


def parseCalibration(filepath):
    # global calibration
    with open(filepath, 'r') as file:
        calibration = []
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        # print(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy,
        # cx, cy
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        # print(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy,
        # cx, cy
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        # print(str(calibration[2]) + '\n') #depth camera position relativelly
        # to color camera in meters
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration

# convert point into 3D


def convert2Dto3D(intrisics, x, y, z):
    # print(intrisics)
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = (x - cx) * z / fx
    ty = (y - cy) * z / fy
    output = []
    output.append(tx)
    output.append(ty)
    output.append(z)
    return output

# convert point into 2D


def convert3Dto2D(intrisics, x, y, z):
    # print(intrisics)
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = x * fx / z + cx
    ty = y * fy / z + cy
    output = []
    output.append(tx)
    output.append(ty)
    output.append(z)
    return output


def parseConfidence(tx, ty, data, maxConfidence):
    return (data[(int(ty) * width + int(tx)) * 3 + 2]) / maxConfidence

# getter


def getWidth():
    return width

# getter


def getHeight():
    return height

# setter


def setWidth(value):
    global width
    width = value

# setter


def setHeight(value):
    global height
    height = value

    # parse PCD


def parsePCD(filepath):
    with open(filepath, 'r') as file:
        data = []
        while True:
            line = str(file.readline())
            if line.startswith('DATA'):
                break

        while True:
            line = str(file.readline())
            if not line:
                break
            else:
                values = parseNumbers(line)
                data.append(values)
        return data


# get valid points in depthmaps
def getCount(calibration, data, depthScale):
    count = 0
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            depth = parseDepth(x, y, data, depthScale)
            if depth:
                res = convert2Dto3D(calibration[1], x, y, depth)
                if res:
                    count = count + 1
    return count


def subsample_pointcloud(
    pointcloud,
    target_size,
    subsampling_method="random",
    dimensions=[
        0,
        1,
        2]):
    """
    Yields a subsampled pointcloud.
    These subsamplinge modes are available:
    - "random": Yields a random subset. Multiple occurrences of a single point are possible.
    - "first": Yields the first n points
    - "sequential_skip": Attempts to keep the order of the points intact, might skip some elements if the pointcloud is too big. E.g. every second point is skipped.
    Note: All methods ensure that the target_size is met. If necessary zeroes are appended.
    """

    # Check if the requested subsampling method is all right.
    possible_subsampling_methods = ["random", "first", "sequential_skip"]
    assert subsampling_method in possible_subsampling_methods, "Subsampling method {} not in {}".format(
        subsampling_method, possible_subsampling_methods)

    # Random subsampling.
    if subsampling_method == "random":
        indices = np.arange(0, pointcloud.shape[0])
        indices = np.random.choice(indices, target_size)
        result = pointcloud[indices]

    elif subsampling_method == "first":
        result = np.zeros((target_size, pointcloud.shape[1]), dtype="float32")
        result[:len(pointcloud), :] = pointcloud[:target_size]

    elif subsampling_method == "sequential_skip":
        result = np.zeros((target_size, pointcloud.shape[1]), dtype="float32")
        skip = max(1, round(len(pointcloud) / target_size))
        pointcloud_skipped = pointcloud[::skip, :]
        result = np.zeros((target_size, pointcloud.shape[1]), dtype="float32")
        result[:len(pointcloud_skipped), :] = pointcloud_skipped[:target_size]

    return result[:, dimensions]


def preprocess_pointcloud(pointcloud, subsample_size, channels):
    if subsample_size is not None:
        skip = max(1, round(len(pointcloud) / subsample_size))
        pointcloud_skipped = pointcloud[::skip, :]
        result = np.zeros(
            (subsample_size,
             pointcloud.shape[1]),
            dtype="float32")
        result[:len(pointcloud_skipped),
               :] = pointcloud_skipped[:subsample_size]
        pointcloud = result
    if channels is not None:
        pointcloud = pointcloud[:, channels]
    return pointcloud.astype("float32")


def pcd_to_depthmap(paths, calibration):
    depthmaps = []
    for path in paths:
        depthmap = lenovo_pcd2depth(path, calibration)
        if depthmap is not None:
            depthmap = preprocess(depthmap)
            depthmaps.append(depthmap)

    depthmaps = np.array(depthmaps)

    return depthmaps


def depthmap_to_pcd(paths, calibration, preprocessing_type, input_shape=[]):
    pcds = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = load_depth(path)
        pcd = getPCD(path, calibration, data, maxConfidence, depthScale)

        if pcd.shape[0] == 0:
            continue

        if preprocessing_type == 'pointnet':
            pcd = subsample_pointcloud(
                pcd,
                target_size=input_shape[0],
                subsampling_method="sequential_skip")
        elif preprocessing_type == 'gapnet':
            pcd = [preprocess_pointcloud(pcd, 1024, list(range(3)))]
        pcds.append(pcd)

    pcds = np.array(pcds)

    if preprocessing_type == 'gapnet':
        pcds = pcds.reshape((-1, 1024, 3))

    return pcds


def get_depthmaps(paths):
    depthmaps = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = load_depth(path)
        depthmap, height, width = prepare_depthmap(
            data, width, height, depthScale)
        # print(height, width)
        depthmap = preprocess(depthmap)
        # print(depthmap.shape)
        depthmaps.append(depthmap)

    depthmaps = np.array(depthmaps)

    return depthmaps


def load_pcd_as_ndarray(pcd_path):
    """
    Loads a PCD-file. Yields a numpy-array.
    """
    pointcloud = PyntCloud.from_file(pcd_path)
    values = pointcloud.points.values
    return values


def pcd_to_ndarray(pcd_paths, input_shape):
    pointclouds = []
    for pcd_path in pcd_paths:
        print(pcd_path)
        try:
            pointcloud = load_pcd_as_ndarray(pcd_path)
        except Exception as error:
            print(error)
            continue
        pointcloud = subsample_pointcloud(
            pointcloud,
            target_size=input_shape[0],
            subsampling_method="sequential_skip")
        pointclouds.append(pointcloud)
    pointclouds = np.array(pointclouds)
    # predictions = model.predict(pointclouds)
    # return predictions
    return pointclouds


def pcd_processing_gapnet(pcd_paths):
    pointclouds = []
    for pcd_path in pcd_paths:
        try:
            pointcloud = load_pcd_as_ndarray(pcd_path)
        except Exception as error:
            print(error)
            continue
        pointcloud = [preprocess_pointcloud(pointcloud, 1024, list(range(3)))]
        pointclouds.append(pointcloud)
    pointclouds = np.array(pointclouds)
    pointclouds = pointclouds.reshape((-1, 1024, 3))

    return pointclouds