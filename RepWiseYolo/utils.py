# utils.py - Adapted for YOLO Keypoint Structure

import numpy as np
import cv2
import math

# MediaPipe imports removed

# --- YOLO Keypoint Mapping (using the 17 standard COCO keypoints) ---
# Map the semantic name used in your exercise logic to the COCO keypoint index (0-16).
YOLO_KEYPOINT_MAP = {
    "NOSE": 0, "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6, "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8, "LEFT_WRIST": 9, "RIGHT_WRIST": 10, "LEFT_HIP": 11,
    "RIGHT_HIP": 12, "LEFT_KNEE": 13, "RIGHT_KNEE": 14, "LEFT_ANKLE": 15,
    "RIGHT_ANKLE": 16,
    "LEFT_EYE": 1, "RIGHT_EYE": 2, "LEFT_EAR": 3, "RIGHT_EAR": 4,  # Added for visibility check robustness
}

# --- OpenCV Font ---
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Color Constants ---
GOOD_COLOR = (0, 255, 0)  # Green
BAD_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
OUTLINE_COLOR = (0, 0, 0)  # Black


# --- Helper Functions ---

def calculate_angle(a, b, c):
    """
    Calculates the angle between three 3D/2D points.
    a, b, c: Tuples or lists of (x, y, z) or (x, y) coordinates.
    The angle is calculated at point 'b'.
    """
    a = np.array(a[:3])  # First point (take max 3 elements)
    b = np.array(b[:3])  # Mid point (vertex)
    c = np.array(c[:3])  # End point

    # Calculate vectors
    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cosine_angle = dot_product / (mag_ba * mag_bc)

    # Clip values to prevent arccos errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_landmark_coords(landmarks, part_name, image_width, image_height):
    """
    Retrieves the pixel coordinates (x, y) of a specific landmark from YOLO output.
    landmarks: The keypoints array for the primary person, e.g., [[x1, y1, conf1], [x2, y2, conf2], ...]
    """
    if part_name not in YOLO_KEYPOINT_MAP:
        # In a real app, this should throw an error, but here we return a safe default
        return (0, 0)

    index = YOLO_KEYPOINT_MAP[part_name]

    # Check if the index is valid for the landmarks array
    if index >= len(landmarks):
        return (0, 0)

    lm = landmarks[index]

    # YOLO keypoints are usually raw pixel coordinates or normalized.
    # We assume they are raw pixel coords (X, Y) here, which matches how they were used in the old code.
    return (int(round(lm[0])), int(round(lm[1])))


def get_landmark_3d(landmarks, part_name):
    """
    Retrieves the proportional coordinates (x, y, z=0) of a specific landmark from YOLO output.
    We use proportional coordinates (raw pixel values) and set Z=0 to maintain angle calculation compatibility.
    """
    if part_name not in YOLO_KEYPOINT_MAP:
        return [0, 0, 0]

    index = YOLO_KEYPOINT_MAP[part_name]

    if index >= len(landmarks):
        return [0, 0, 0]

    lm = landmarks[index]

    # Use the raw pixel coordinates as proportional values for angle calculation, and set z to 0.
    return [lm[0], lm[1], 0]


# --- Skeleton Drawing Function (NEW for YOLO) ---
def draw_yolo_skeleton(image, landmarks, color=(100, 100, 100), thickness=2, circle_radius=2):
    """
    Draws the generic skeleton on the image from the YOLO keypoints array.
    This replaces mp_drawing.draw_landmarks for the base skeleton.
    """

    # Define connections based on COCO keypoint indices
    connections = [
        (YOLO_KEYPOINT_MAP["LEFT_SHOULDER"], YOLO_KEYPOINT_MAP["LEFT_ELBOW"]),
        (YOLO_KEYPOINT_MAP["LEFT_ELBOW"], YOLO_KEYPOINT_MAP["LEFT_WRIST"]),
        (YOLO_KEYPOINT_MAP["RIGHT_SHOULDER"], YOLO_KEYPOINT_MAP["RIGHT_ELBOW"]),
        (YOLO_KEYPOINT_MAP["RIGHT_ELBOW"], YOLO_KEYPOINT_MAP["RIGHT_WRIST"]),

        (YOLO_KEYPOINT_MAP["LEFT_SHOULDER"], YOLO_KEYPOINT_MAP["RIGHT_SHOULDER"]),
        (YOLO_KEYPOINT_MAP["LEFT_SHOULDER"], YOLO_KEYPOINT_MAP["LEFT_HIP"]),
        (YOLO_KEYPOINT_MAP["RIGHT_SHOULDER"], YOLO_KEYPOINT_MAP["RIGHT_HIP"]),
        (YOLO_KEYPOINT_MAP["LEFT_HIP"], YOLO_KEYPOINT_MAP["RIGHT_HIP"]),

        (YOLO_KEYPOINT_MAP["LEFT_HIP"], YOLO_KEYPOINT_MAP["LEFT_KNEE"]),
        (YOLO_KEYPOINT_MAP["LEFT_KNEE"], YOLO_KEYPOINT_MAP["LEFT_ANKLE"]),
        (YOLO_KEYPOINT_MAP["RIGHT_HIP"], YOLO_KEYPOINT_MAP["RIGHT_KNEE"]),
        (YOLO_KEYPOINT_MAP["RIGHT_KNEE"], YOLO_KEYPOINT_MAP["RIGHT_ANKLE"]),
    ]

    keypoint_coords = {}
    for name, index in YOLO_KEYPOINT_MAP.items():
        if index < len(landmarks):
            # Keypoint is [x, y, confidence]
            if landmarks[index][2] > 0.4:  # Only draw if confidence is reasonable
                keypoint_coords[index] = (int(landmarks[index][0]), int(landmarks[index][1]))

    # Draw lines (bones)
    for p1_idx, p2_idx in connections:
        if p1_idx in keypoint_coords and p2_idx in keypoint_coords:
            p1 = keypoint_coords[p1_idx]
            p2 = keypoint_coords[p2_idx]
            cv2.line(image, p1, p2, color, thickness)

    # Draw circles (joints)
    for index, (x, y) in keypoint_coords.items():
        cv2.circle(image, (x, y), circle_radius, color, -1)