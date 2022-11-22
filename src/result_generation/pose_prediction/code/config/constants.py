COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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
    "1"  : "nose",
    "2"  : "lefy_eye_inner",
    "3"  : "lefy_eye",
    "4"  : "left_eye_outer",
    "5"  : "right_eye_inner",
    "6"  : "right_eye",
    "7"  : "right_eye_outer",
    "8"  : "left_ear",
    "9"  : "right_ear",
    "10" : "left_mouth",
    "11" : "right_mouth",
    "12" : "left_shoulder",
    "13" : "right_shoulder",
    "14" : "left_elbow",
    "15" : "right_elbow",
    "16" : "left_wrist",
    "17" : "right_wrist",
    "18" : "left_hip",
    "19" : "right_hip",
    "20" : "left_knee",
    "21" : "right_knee",
    "22" : "left_ankle",
    "23" : "right_ankle",
    "24" : "left_pinky",
    "25" : "right_pinky",
    "26" : "left_index",
    "27" : "right_index",
    "28" : "left_thumb",
    "29" : "right_thumb",
    "30" : "left_heel",
    "31" : "right_heel",
    "32" : "left_foot_index",
    "33" : "right_foot_index",
}



MLKIT_SKELETON = [
[15 ,   17],
[13 ,   19],
[13 ,   15],
[19 ,   21],
[21 ,   23],
[17 ,   29],
[17 ,   25],
[17 ,   27],
[27 ,   25],
[23 ,   31],
[31 ,   33],
[1  ,  2],
[2  ,  3],
[3  ,  4],
[4  ,  8],
[1  ,  5],
[5  ,  6],
[6  ,  7],
[7  ,  9],
[10 ,   11],
[12 ,   13],
[18 ,   19],
[12 ,   14],
[14 ,   16],
[12 ,   18],
[18 ,   20],
[20 ,   22],
[16 ,   28],
[16 ,   24],
[16 ,   26],
[26 ,   24],
[22 ,   30],
[30 ,   32],
]

MLKIT_NUM_KPTS = 33


MLKIT_BODY_JOINTS = [
    # Right body
    ("right_shoulder" , "right_elbow"),
    ("right_elbow" , "right_wrist"),
    ("right_shoulder" , "right_hip"),
    ("right_hip" , "right_knee"),
    ("right_knee" , "right_ankle"),
    ("right_wrist" , "right_thumb"),
    ("right_wrist" , "right_pinky"),
    ("right_wrist" , "right_index"),
    ("right_index" , "right_pinky"),
    ("right_ankle" , "right_heel"),
    ("right_heel" , "right_foot_index"),
    #Face
    ("nose" , "lefy_eye_inner"),
    ("lefy_eye_inner" , "lefy_eye"),
    ("lefy_eye" , "left_eye_outer"),
    ("left_eye_outer" , "left_ear"),
    ("nose" , "right_eye_inner"),
    ("right_eye_inner" , "right_eye"),
    ("right_eye" , "right_eye_outer"),
    ("right_eye_outer" , "right_ear"),
    ("left_mouth" , "right_mouth"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    #Left body
    ("left_shoulder" , "left_elbow"),
    ("left_elbow" , "left_wrist"),
    ("left_shoulder" , "left_hip"),
    ("left_hip" , "left_knee"),
    ("left_knee" , "left_ankle"),
    ("left_wrist" , "left_thumb"),
    ("left_wrist" , "left_pinky"),
    ("left_wrist" , "left_index"),
    ("left_index" , "left_pinky"),
    ("left_ankle" , "left_heel"),
    ("left_heel" , "left_foot_index"),

]


# MlkitColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#               [0, 255, 85], [0, 255, 170], [0, 255, 255], [
#                   0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# All white colors
MlkitColors = [[255, 255, 255]] * 33

