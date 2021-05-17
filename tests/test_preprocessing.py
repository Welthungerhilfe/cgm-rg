from pathlib import Path
import pickle
import sys

sys.path.append('./src')
from utils import preprocessing

CWD = Path.cwd()
WIDTH = 240
HEIGHT = 180


def test_load_depth_hugh():
    # Setup
    depth_file = CWD.joinpath('tests', 'static_files', 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a', 'depth', 'bd67cd9e-69c7-11eb-984b-77ac9d2b4986')

    # Exercise
    data, width, height, depth_scale, max_confidence = preprocessing.load_depth(depth_file)

    # Verify
    assert isinstance(data, bytes)

    # Cleanup - none required


def test_load_depth_small():
    path_small_depthmap = CWD.joinpath('tests', "4ed427b5-3fd9-4f4d-8e58-19e39c7d77b6")
    depth_data, width, height, depth_scale, max_confidence = preprocessing.load_depth(path_small_depthmap)
    assert (width, height) == (int(WIDTH * 0.75), int(HEIGHT * 0.75))

    preprocessing.get_depthmaps([path_small_depthmap])


def test_load_depth_big():
    path_big_depthmap = CWD.joinpath('tests', "6b0c5a4f-d8f2-4658-b7f9-9406f1770259")
    depth_data, width, height, depth_scale, max_confidence = preprocessing.load_depth(path_big_depthmap)
    assert (width, height) == (WIDTH, HEIGHT)


def test_prepare_depthmap():
    # Setup
    width, height = int(WIDTH * 0.75), int(HEIGHT * 0.75)
    dummy_depth_data_file = CWD.joinpath('tests', 'static_files', 'dummy_depth_data.pkl')
    with open(dummy_depth_data_file, 'rb') as f:
        dummy_depth_data = pickle.load(f)

    # Exercise
    depthmap = preprocessing.prepare_depthmap(dummy_depth_data[0], width, height, dummy_depth_data[3])

    # Verify
    assert depthmap.shape == (width, height)


def test_get_depthmaps_small():
    path_small_depthmap = CWD.joinpath('tests', "4ed427b5-3fd9-4f4d-8e58-19e39c7d77b6")
    preprocessing.get_depthmaps([path_small_depthmap])


def test_get_depthmaps_big():
    path_big_depthmap = CWD.joinpath('tests', "6b0c5a4f-d8f2-4658-b7f9-9406f1770259")
    preprocessing.get_depthmaps([path_big_depthmap])


if __name__ == "__main__":
    test_load_depth_hugh()
    # test_get_depthmaps_small()
    # test_get_depthmaps_big()
    # test_prepare_depthmap()
