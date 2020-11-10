import utils.preprocessing as preprocessing
import sys
sys.path.append("../src")


def test_that_get_height_works():
    """
    Test to check if we get correct height of depthmap.
    """
    # Setup - None
    preprocessing.setHeight(int(180 * 0.75))
    # Exercise
    result = preprocessing.getHeight()

    # Verify
    truth = 135
    assert result == truth

    # Cleanup - none required


def test_that_get_width_works():
    """
    Test to check if we get correct width of depthmap.
    """
    # Setup - None
    preprocessing.setWidth(int(240 * 0.75))
    # Exercise
    result = preprocessing.getWidth()

    # Verify
    truth = 180
    assert result == truth

    # Cleanup - none required
