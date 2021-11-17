import numpy as np
from bunch import Bunch
from cgmml.models.HRNET.hrnet3d import (JOINT_INDEX_LEFT_ANKLE,
                                        JOINT_INDEX_LEFT_EYE,
                                        JOINT_INDEX_LEFT_HIP,
                                        JOINT_INDEX_LEFT_KNEE,
                                        JOINT_INDEX_LEFT_SHOULDER,
                                        JOINT_INDEX_NOSE,
                                        JOINT_INDEX_RIGHT_ANKLE,
                                        JOINT_INDEX_RIGHT_EYE,
                                        JOINT_INDEX_RIGHT_HIP,
                                        JOINT_INDEX_RIGHT_KNEE,
                                        JOINT_INDEX_RIGHT_SHOULDER)

SKELETON_IMPORTANT = np.array([
    [JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_RIGHT_ANKLE],
    [JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_LEFT_ANKLE],
    [JOINT_INDEX_RIGHT_HIP, JOINT_INDEX_RIGHT_KNEE],
    [JOINT_INDEX_LEFT_HIP, JOINT_INDEX_LEFT_KNEE],
    [JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_RIGHT_HIP],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_LEFT_HIP],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_RIGHT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_RIGHT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_RIGHT_EYE],
    [JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_EYE],
])

DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG_TRAIN = Bunch(dict(  # Hyperparameters

    SCAN_TYPES_TO_USE=['200', '201'],  # '202', '101', '102' is empty,

    # bone features
    SKELETON=SKELETON_IMPORTANT,
    USE_3DVECTOR_FEATURES=True,  # needed
    USE_MULTIBONE_SUM=True,  # needed in RF

    # joint features
    USE_CONFIDENCE_FEATURES=False,  # not needed in RF
    USE_VALIDITY_FEATURES=False,  # not needed in RF
))
