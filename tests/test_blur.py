from pathlib import Path
import sys


REPO_DIR = Path(__file__).parents[1].absolute()
SRC_DIR = str(REPO_DIR / "src")
sys.path.append(SRC_DIR)

print(sys.path)
from utils.preprocessing import blur_faces_in_file  # noqa: E402

IMAGE_FNAME = "rgb_1583438117-71v1y4z0gd_1592711198959_101_74947.76209955901.jpg"


def test_blur_faces_in_file():
    source_path = str(Path(__file__).parent / IMAGE_FNAME)
    target_path = str(Path('/tmp') / f"{IMAGE_FNAME}_blurred.jpg")
    assert blur_faces_in_file(source_path, target_path)
