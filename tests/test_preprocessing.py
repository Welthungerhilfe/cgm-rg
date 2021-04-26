import pathlib
import pickle
import sys

sys.path.append('./src')
from utils import preprocessing

CWD = pathlib.Path.cwd()
WIDTH = 240
HEIGHT = 180

def test_load_depth():
    """
    Test to check load depth function
    """
    # Setup
    depth_file = CWD.joinpath('tests', 'static_files', 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a', 'depth', 'bd67cd9e-69c7-11eb-984b-77ac9d2b4986')

    # Exercise
    data, width, height, depthScale, max_confidence = preprocessing.load_depth(depth_file)

    # Verify
    assert isinstance(data, bytes)

    # Cleanup - none required


def test_prepare_depthmap():
    # Setup
    preprocessing.set_width(int(WIDTH * 0.75))
    preprocessing.set_height(int(HEIGHT * 0.75))
    dummy_depth_data_file = CWD.joinpath('tests', 'static_files', 'dummy_depth_data.pkl')
    with open(dummy_depth_data_file, 'rb') as f:
        dummy_depth_data = pickle.load(f)

    # Exercise
    result = preprocessing.prepare_depthmap(dummy_depth_data[0], dummy_depth_data[1], dummy_depth_data[2], dummy_depth_data[3])

    # Verify
    assert result[0].shape[0] == result[2]
    assert result[0].shape[1] == result[1]


def test_get_depthmaps_small():
    path_small_depthmap = CWD.joinpath('tests', "4ed427b5-3fd9-4f4d-8e58-19e39c7d77b6")
    depth_data, width, height, depth_scale, max_confidence = preprocessing.load_depth(path_small_depthmap)
    assert (width, height) == (int(WIDTH * 0.75), int(HEIGHT * 0.75))

    preprocessing.set_width(width)
    preprocessing.set_height(height)
    preprocessing.get_depthmaps([path_small_depthmap])


def test_get_depthmaps_big():
    path_big_depthmap = CWD.joinpath('tests', "6b0c5a4f-d8f2-4658-b7f9-9406f1770259")
    depth_data, width, height, depth_scale, max_confidence = preprocessing.load_depth(path_big_depthmap)
    assert (width, height) == (WIDTH, HEIGHT)

    preprocessing.set_width(width)
    preprocessing.set_height(height)
    preprocessing.get_depthmaps([path_big_depthmap])


if __name__ == "__main__":
    test_get_depthmaps_small()
    test_get_depthmaps_big()
