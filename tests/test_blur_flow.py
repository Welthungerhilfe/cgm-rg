import sys
sys.path.append('./src')
import pytest
from bunch import Bunch
import pathlib
import set_up_dummy_objects

current_working_directory = pathlib.Path.cwd()


def test_bunch_object_to_json_object():
    """
    Test to check if we get json object
    """
    # Setup
    blurflow = set_up_dummy_objects.get_dummy_blur_flow_object()
    bunch_object = Bunch(a=1, b=2)
    # Exercise
    result = blurflow.bunch_object_to_json_object(bunch_object)

    # Verify
    truth = {'a': 1, 'b': 2}
    assert result == truth
    assert isinstance(truth, dict)


def test_get_input_path():
    """
    Test to check if we get input path
    """
    # Setup
    blurflow = set_up_dummy_objects.get_dummy_blur_flow_object()
    directory = 'app/scans'
    filename = 'workflow.json'

    # Exercise
    result = blurflow.get_input_path(directory, filename)

    # Verify
    truth = 'app/scans/workflow.json'
    assert result == truth


@pytest.mark.skip(reason="need to work on this test")
def test_blur_face_file_not_exists():
    """
    Test to check if face blur works if given path is not present
    """
    pass


def test_blur_face_file_exists():
    """
    Test to check if face blur works if given path is present
    """
    # Setup
    blurflow = set_up_dummy_objects.get_dummy_blur_flow_object()
    input_file = str(current_working_directory.joinpath('tests', 'static_files', 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a', 'img', 'bd8ba746-69c7-11eb-984b-23c55a7d518b'))

    # Exercise
    result = blurflow.blur_face(input_file)

    # Verify

    assert result


def test_post_blur_files_successful():
    """
    Test to check if files are post to api successfully
    """
    pass


def test_post_blur_files_unsuccessful():
    """
    Test to check if files are not post to api
    """
    pass


def test_prepare_result_object():
    """
    Test to check the result object
    """
    pass


def test_post_result_object_successful():
    """
    Test to check successful upload of result object
    """
    pass


def test_post_result_object_unsuccessful():
    """
    Test to check unsuccessful upload of result object
    """
    pass
