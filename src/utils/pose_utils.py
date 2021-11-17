from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from bunch import Bunch
from cgmml.models.HRNET.hrnet3d import (JOINT_INDEX_LEFT_ANKLE,
                                        JOINT_INDEX_LEFT_HIP,
                                        JOINT_INDEX_LEFT_KNEE,
                                        JOINT_INDEX_LEFT_SHOULDER,
                                        JOINT_INDEX_RIGHT_ANKLE,
                                        JOINT_INDEX_RIGHT_HIP,
                                        JOINT_INDEX_RIGHT_KNEE,
                                        JOINT_INDEX_RIGHT_SHOULDER)

MULTIBONE_STRUCTURES = [
    [JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_RIGHT_HIP, JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_RIGHT_ANKLE],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_LEFT_HIP, JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_LEFT_ANKLE],
]

Joint = namedtuple('Joint', ['x', 'y', 'z', 'confidence', 'distance_in_px'])


def get_vec_of_joints(joint1, joint2):
    p1 = np.array([joint1.x, joint1.y, joint1.z])
    p2 = np.array([joint2.x, joint2.y, joint2.z])
    vec = p1 - p2
    return vec


def get_features_from_fpath(obj_file_path: Path, config_train: Bunch) -> Dict[str, float]:
    joints = get_joints_from_fpath(obj_file_path)
    child_features = {}

    for joint_index1, joint_index2 in config_train.SKELETON:
        # norms of skeleton vectors (11 or 19 features)
        vec = get_vec_of_joints(joints[joint_index1], joints[joint_index2])
        child_features[f'dist{joint_index1}to{joint_index2}'] = np.linalg.norm(vec)

        # 3D vectors (11*3=33 or 19*3 features)
        if config_train.USE_3DVECTOR_FEATURES:
            for dim, vec_component in zip(['x', 'y', 'z'], vec):
                child_features[f'distvec_{dim}{joint_index1}to{joint_index2}'] = vec_component

    # sum of bones' lengths
    if config_train.USE_MULTIBONE_SUM:
        for structure in MULTIBONE_STRUCTURES:
            sum_ = 0
            for joint_index1, joint_index2 in zip(structure[:-1], structure[1:]):
                vec = get_vec_of_joints(joints[joint_index1], joints[joint_index2])
                sum_ += np.linalg.norm(vec)
            structure_name = '-'.join(map(str, structure))
            child_features[f'multibone{structure_name}'] = sum_

    # confidence and validity (17*2=34 features)
    for joint_index, joint in enumerate(joints):
        if config_train.USE_CONFIDENCE_FEATURES:
            child_features[f'joint{joint_index}confidence'] = joint.confidence
        if config_train.USE_VALIDITY_FEATURES:
            child_features[f'joint{joint_index}validity'] = joint.distance_in_px

    # Check that the features are not NaN
    values = np.array(list(child_features.values()))
    assert np.all(~np.isnan(values)), (obj_file_path, child_features, joints)

    return child_features


def get_artifact_id_from_path(path: Path) -> str:
    return path.with_suffix('').name


def get_path_from_artifact_id(artifact_id: str, dir_path: Path) -> Path:
    return Path(dir_path) / f'{artifact_id}.obj'


def parse_line_in_obj_file(line: str) -> Joint:
    line_elements = line.strip().split(' ')
    assert len(line_elements) == 8, line_elements
    joint = Joint(x=float(line_elements[1]),
                  y=float(line_elements[2]),
                  z=float(line_elements[3]),
                  confidence=float(line_elements[-2]),
                  distance_in_px=float(line_elements[-1]))
    return joint


def get_joints_from_fpath(obj_file_path: Path) -> List[Joint]:
    lines = open(obj_file_path).readlines()
    lines = [line for line in lines if line[0] == 'v']  # keep only lines starting with 'v'
    joints = [parse_line_in_obj_file(line) for line in lines]
    assert len(joints) == 17
    return joints


def plot_history(history: tf.keras.callbacks.History):
    mae, val_mae = history.history["mae"], history.history["val_mae"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs = range(1, len(mae) + 1)
    plt.plot(epochs, mae, "ro", label="Training MAE")
    plt.plot(epochs, val_mae, "b", label="Validation MAE")
    plt.title("Training and validation MAE")
    plt.grid(True)
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, 5))  # 5cm

    plt.figure()
    plt.plot(epochs, loss, "ro", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.grid(True)
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 75))  # 5cm
    plt.show()


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare grouped dataframe for evaluation
    Args:
        df: Needs to contain at least artifact_ids as index, as well as 'height', 'predicted' and 'scantype' columns.
    Returns:
        grouped dataframe
    """
    df.loc[:, 'qrcode'] = df.index
    df_grouped = df.groupby(['qrcode', 'scantype']).mean()
    df_grouped.loc[:, 'error'] = df_grouped['height'] - df_grouped['predicted']
    return df_grouped
