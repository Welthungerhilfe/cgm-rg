from os import getenv

# workflow details

PLAINCNN_HEIGHT_WORKFLOW_NAME = "q4-depthmap-plaincnn-height-469k"
PLAINCNN_HEIGHT_WORKFLOW_VERSION = "2.0.7"
MEAN_PLAINCNN_HEIGHT_WORKFLOW_NAME = "q4-depthmap-plaincnn-height-469k-mean"
MEAN_PLAINCNN_HEIGHT_WORKFLOW_VERSION = "2.0.7"

EFFICIENT_HEIGHT_WORKFLOW_NAME = "efficient_former_model"
EFFICIENT_HEIGHT_WORKFLOW_VERSION = "1.0.0"
MEAN_EFFICIENT_HEIGHT_WORKFLOW_NAME = "efficient_former_model_mean"
MEAN_EFFICIENT_HEIGHT_WORKFLOW_VERSION = "1.0.0"

PLAINCNN_WEIGHT_WORKFLOW_NAME = "q4-depthmap-plaincnn-weight-264k"
PLAINCNN_WEIGHT_WORKFLOW_VERSION = "1.0.1"
MEAN_PLAINCNN_WEIGHT_WORKFLOW_NAME = "q4-depthmap-plaincnn-weight-264k-mean"
MEAN_PLAINCNN_WEIGHT_WORKFLOW_VERSION = "1.0.1"

STANDING_LAYING_WORKFLOW_NAME = "standing_laying"
STANDING_LAYING_WORKFLOW_VERSION = "1.0.4"

POSE_WORKFLOW_NAME = "pose_prediction"
POSE_WORKFLOW_VERSION = "1.0.4"
POSE_VISUALIZE_WORKFLOW_NAME = "pose_prediction_visualization"
POSE_VISUALIZE_WORKFLOW_VERSION = "1.0.4"

BLUR_WORKFLOW_NAME = "face_recognition"
BLUR_WORKFLOW_VERSION = "2.1.0"
FACE_DETECTION_WORKFLOW_NAME = "face_detection"
FACE_DETECTION_WORKFLOW_VERSION = "2.1.0"

DEPTH_IMG_WORKFLOW_NAME = "depthmap-image"
DEPTH_IMG_WORKFLOW_VERSION = "1.0.1"

APP_POSE_VISUALIZE_WORKFLOW_NAME = "mlkit_pose_visualisation"
APP_POSE_VISUALIZE_WORKFLOW_VERSION = "0.1.0"
APP_POSE_WORKFLOW_NAME = "app_pose_predicition"
APP_POSE_WORKFLOW_VERSION = "1.0"

EFFICIENT_POSE_WORKFLOW_NAME = "efficient_pose_prediction"
EFFICIENT_POSE_WORKFLOW_VERSION = "1.0.0"
EFFICIENT_POSE_VISUALIZE_WORKFLOW_NAME = "efficient_pose_prediction_visualization"
EFFICIENT_POSE_VISUALIZE_WORKFLOW_VERSION = "1.0.0"

STANDING_SCAN_TYPE = ["100", "101", "102"]
LAYING_SCAN_TYPE = ["200", "201", "202"]

# # inference endpoints service names

# HEIGHT_PLAINCNN_SERVICE_NAME = ""
# WEIGHT_PLAINCNN_SERVICE_NAME = ""
# POSE_SERVICE_NAME = ""
# FACE_BLUR_SERVICE_NAME = ""
# STANDING_LAYING_SERVICE_NAME = ""

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [
        8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [
                  0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17


MLKIT_KEYPOINT_INDEXES = {
    "1": "nose",
    "2": "left_eye_inner",
    "3": "left_eye",
    "4": "left_eye_outer",
    "5": "right_eye_inner",
    "6": "right_eye",
    "7": "right_eye_outer",
    "8": "left_ear",
    "9": "right_ear",
    "10": "left_mouth",
    "11": "right_mouth",
    "12": "left_shoulder",
    "13": "right_shoulder",
    "14": "left_elbow",
    "15": "right_elbow",
    "16": "left_wrist",
    "17": "right_wrist",
    "18": "left_hip",
    "19": "right_hip",
    "20": "left_knee",
    "21": "right_knee",
    "22": "left_ankle",
    "23": "right_ankle",
    "24": "left_pinky",
    "25": "right_pinky",
    "26": "left_index",
    "27": "right_index",
    "28": "left_thumb",
    "29": "right_thumb",
    "30": "left_heel",
    "31": "right_heel",
    "32": "left_foot_index",
    "33": "right_foot_index",
}


MLKIT_SKELETON = [
    [15, 17],
    [13, 19],
    [13, 15],
    [19, 21],
    [21, 23],
    [17, 29],
    [17, 25],
    [17, 27],
    [27, 25],
    [23, 31],
    [31, 33],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 8],
    [1, 5],
    [5, 6],
    [6, 7],
    [7, 9],
    [10, 11],
    [12, 13],
    [18, 19],
    [12, 14],
    [14, 16],
    [12, 18],
    [18, 20],
    [20, 22],
    [16, 28],
    [16, 24],
    [16, 26],
    [26, 24],
    [22, 30],
    [30, 32],
]

MLKIT_NUM_KPTS = 33


MLKIT_BODY_JOINTS = [
    # Right body
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_wrist", "right_thumb"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_index"),
    ("right_index", "right_pinky"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    # Face
    ("nose", "left_eye_inner"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("left_eye_outer", "left_ear"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("right_eye_outer", "right_ear"),
    ("left_mouth", "right_mouth"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    # Left body
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_wrist", "left_thumb"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_index"),
    ("left_index", "left_pinky"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),

]


# MlkitColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#               [0, 255, 85], [0, 255, 170], [0, 255, 255], [
#                   0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# All white colors
MlkitColors = [[255, 255, 255]] * 33
