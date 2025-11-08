import mediapipe as mp
import numpy as np

# --- MediaPipe Initialization ---
# We initialize mp_pose here so all exercise files can import and use it
mp_pose = mp.solutions.pose

# --- Color Constants ---
GOOD_COLOR = (0, 255, 0)  # Green
BAD_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
OUTLINE_COLOR = (0, 0, 0)  # Black


# --- Helper Functions ---

def calculate_angle(a, b, c):
    """
    Calculates the angle between three 3D points.
    a, b, c: Tuples or lists of (x, y, z) coordinates.
    The angle is calculated at point 'b'.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate dot product
    dot_product = np.dot(ba, bc)

    # Calculate magnitudes
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    # Calculate cosine of the angle
    # Add a small epsilon to avoid division by zero
    cosine_angle = dot_product / (mag_ba * mag_bc + 1e-6)

    # Clip values to prevent arccos errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_landmark_coords(landmarks, part_name, image_width, image_height):
    """
    Retrieves the pixel coordinates (x, y) of a specific landmark.
    """
    lm = landmarks[mp_pose.PoseLandmark[part_name].value]
    return (int(lm.x * image_width), int(lm.y * image_height))


def get_landmark_3d(landmarks, part_name):
    """
    Retrieves the 3D coordinates (x, y) of a specific landmark.
    """
    lm = landmarks[mp_pose.PoseLandmark[part_name].value]
    return [lm.x, lm.y, lm.z]