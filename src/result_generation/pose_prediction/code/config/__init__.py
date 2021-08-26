import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # noqa: E402
from result_generation.pose_prediction.code.config.default import _C as cfg  # noqa
from result_generation.pose_prediction.code.config.default import update_config  # noqa
from result_generation.pose_prediction.code.config.models import MODEL_EXTRAS  # noqa
