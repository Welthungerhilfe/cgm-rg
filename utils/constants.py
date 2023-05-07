from os import getenv

# workflow details

PLAINCNN_HEIGHT_WORKFLOW_NAME = "q4-depthmap-plaincnn-height"
PLAINCNN_HEIGHT_WORKFLOW_VERSION = "2.0.5"
MEAN_PLAINCNN_HEIGHT_WORKFLOW_NAME = "q4-depthmap-plaincnn-height-mean"
MEAN_PLAINCNN_HEIGHT_WORKFLOW_VERSION = "2.0.5"

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
BLUR_WORKFLOW_VERSION = "1.0.3"
FACE_DETECTION_WORKFLOW_NAME = "face_detection"
FACE_DETECTION_WORKFLOW_VERSION = "1.0.3"

DEPTH_IMG_WORKFLOW_NAME = "depthmap-image"
DEPTH_IMG_WORKFLOW_VERSION = "1.0.1"

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
