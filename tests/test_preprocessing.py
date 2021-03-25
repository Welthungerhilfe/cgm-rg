import sys
sys.path.append('./src')
from utils import preprocessing
import pathlib
import pickle

current_working_directory = pathlib.Path.cwd()


def test_load_depth():
    """
    Test to check load depth function
    """
    # Setup
    depth_file = current_working_directory.joinpath('tests', 'static_files', 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a', 'depth', 'bd67cd9e-69c7-11eb-984b-77ac9d2b4986')

    # Exercise
    data, width, height, depthScale, max_confidence = preprocessing.load_depth(depth_file)

    # Verify
    assert isinstance(data, bytes)

    # Cleanup - none required


def test_prepare_depthmap():
    # Setup
    preprocessing.set_width(int(240 * 0.75))
    preprocessing.set_height(int(180 * 0.75))
    dummy_depth_data_file = current_working_directory.joinpath('tests', 'static_files', 'dummy_depth_data.pkl')
    with open(dummy_depth_data_file, 'rb') as f:
        dummy_depth_data = pickle.load(f)

    # Exercise
    result = preprocessing.prepare_depthmap(dummy_depth_data[0], dummy_depth_data[1], dummy_depth_data[2], dummy_depth_data[3])

    # Verify
    assert result[0].shape[0] == result[2]
    assert result[0].shape[1] == result[1]
