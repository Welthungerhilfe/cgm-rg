import numpy as np
import mediapipe as mp
import cv2


mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False)


def estimate_distance_from_segmentation(rgb, depth, max_patch=15, threshold=0.3):
    h_depth, w_depth = depth.shape[:2]
    h_rgb, w_rgb = rgb.shape[:2]

    # --- Resize RGB to match depth size ---
    if (h_rgb, w_rgb) != (h_depth, w_depth):
        rgb = cv2.resize(rgb, (w_depth, h_depth), interpolation=cv2.INTER_LINEAR)

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
        result = segmentor.process(rgb)
        mask = result.segmentation_mask > threshold  # Binary mask

    if not np.any(mask):
        print("No person found in segmentation.")
        return None, (None, None), mask

    # Get coordinates of True pixels
    coords = np.argwhere(mask)
    cy, cx = coords.mean(axis=0).astype(int)

    h, w = depth.shape[:2]
    distance = 0
    patch_size = 3

    while patch_size <= max_patch:
        x1 = max(cx - patch_size, 0)
        x2 = min(cx + patch_size + 1, w)
        y1 = max(cy - patch_size, 0)
        y2 = min(cy + patch_size + 1, h)

        patch = depth[y1:y2, x1:x2]
        patch_nonzero = patch[patch > 0]

        if patch_nonzero.size > 0:
            distance = np.mean(patch_nonzero)
            break

        patch_size += 2

    if distance == 0:
        # fallback: mean depth in entire segmentation mask
        mask_depth = depth[mask]
        nonzero_depth = mask_depth[mask_depth > 0]
        if nonzero_depth.size > 0:
            distance = np.mean(nonzero_depth)
        else:
            distance = None

    return distance, (cx, cy), mask


def estimate_distance_from_pose(rgb, depth, max_patch=15):
    dh, dw = depth.shape[:2]
    h_rgb, w_rgb = rgb.shape[:2]

    # Resize RGB to match depth resolution
    if (h_rgb, w_rgb) != (dh, dw):
        rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)

    results = pose_estimator.process(rgb)

    if not results.pose_landmarks:
        print("No person detected.")
        return None, (None, None)

    landmarks = results.pose_landmarks.landmark
    h, w = rgb.shape[:2]

    # Get required landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Convert to pixel coordinates
    nose_xy = np.array([nose.x * w, nose.y * h])
    mid_hip_xy = np.array([(left_hip.x + right_hip.x) / 2 * w,
                           (left_hip.y + right_hip.y) / 2 * h])

    # Compute 3/4 point from nose toward mid-hip
    center_xy = (nose_xy + 0.75 * (mid_hip_xy - nose_xy)).astype(int)
    cx, cy = center_xy

    # Extract distance from depth map around center point
    patch_size = 3
    distance = 0

    while patch_size <= max_patch:
        x1 = max(cx - patch_size, 0)
        x2 = min(cx + patch_size + 1, w)
        y1 = max(cy - patch_size, 0)
        y2 = min(cy + patch_size + 1, h)

        patch = depth[y1:y2, x1:x2]
        patch_nonzero = patch[patch > 0]
        if patch_nonzero.size > 0:
            distance = np.mean(patch_nonzero)
            break
        patch_size += 2

    if distance == 0:
        distance = None

    return distance, (cx, cy)
