import sys
import pytest
from pathlib import Path

sys.path.append('./src')
import set_up_dummy_objects

CWD = Path.cwd()


@pytest.mark.skip(reason="need to work on this test")
def test_blur_face_file_not_exists():
    """Test to check if face blur works if given path is not present"""
    pass


def test_blur_face_file_exists():
    """Test to check if face blur works if given path is present"""
    # Setup
    blurflow = set_up_dummy_objects.get_dummy_blur_flow_object()
    input_file = str(CWD.joinpath('tests', 'static_files', 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a',
                     'img', 'bd8ba746-69c7-11eb-984b-23c55a7d518b'))

    # # Set Resize factor to be used in the blur_face
    # blurflow.blur_set_resize_factor()

    # Exercise
    result = blurflow.blur_face(input_file)

    # Verify
    assert result[1]


def test_post_blur_files_successful():
    """Test to check if files are post to api successfully"""
    pass


def test_post_blur_files_unsuccessful():
    """Test to check if files are not post to api"""
    pass


def test_prepare_result_object():
    """Test to check the result object"""
    pass


def test_post_result_object_successful():
    """Test to check successful upload of result object"""
    pass


def test_post_result_object_unsuccessful():
    """Test to check unsuccessful upload of result object"""
    pass
