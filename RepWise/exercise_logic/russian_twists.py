from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR

# Simple state variables to track the range of motion (rotation)
ROTATION_LEFT_THRESHOLD = -0.15  # X-coordinate distance relative to hip center (negative is left)
ROTATION_RIGHT_THRESHOLD = 0.15  # X-coordinate distance relative to hip center (positive is right)
BACK_FLAT_THRESHOLD = 120 # Angle between knee, hip, and shoulder (upright torso check)


def process_russian_twist(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for the Bodyweight Russian Twist.
    Checks torso rotation (left/right) using shoulder X-coordinates relative to hip.
    Also checks for a flat back (upright torso).
    """
    # Get 3D coordinates (using right side landmarks for rotation check)
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")

    # Torso angle check (e.g., knee-hip-shoulder angle for leaning back)
    # Using hip angle (knee-hip-shoulder) to check if the user is leaning back correctly
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    back_angle = calculate_angle(left_knee_3d, left_hip_3d, left_shoulder_3d)

    # Relative X-position of the right wrist to the hip (proxy for rotation)
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    rotation_value = right_wrist_3d[0] - left_hip_3d[0]

    # Get 2D coordinates for drawing
    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    center_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)

    # --- Form Correction ---
    back_line_color = GOOD_COLOR
    if back_angle < BACK_FLAT_THRESHOLD or back_angle > 180: # Check for the correct seated lean-back angle
        feedback_text = "Maintain a flat back and lean back slightly."
        back_line_color = BAD_COLOR
    else:
        feedback_text = "Rotate left!"
        back_line_color = GOOD_COLOR


    # --- Rep Counting (State Machine) ---
    # State: left, center, right

    # 1. At Left side (Contraction)
    if rotation_value < ROTATION_LEFT_THRESHOLD:
        if exercise_state == "right":
            exercise_state = "left"
            feedback_text = "Twist to the right!"

    # 2. At Right side (Contraction)
    elif rotation_value > ROTATION_RIGHT_THRESHOLD:
        if exercise_state == "left":
            exercise_state = "right"
            rep_counter += 1
            feedback_text = "Rep Complete! Twist back to the left."

    # 3. Center (Starting Position)
    elif ROTATION_LEFT_THRESHOLD <= rotation_value <= ROTATION_RIGHT_THRESHOLD:
        if exercise_state == "up": # Use "up" as initial state before first rotation
            feedback_text = "Twist left to begin!"
        elif exercise_state == "left":
            feedback_text = "Keep twisting right!"
        elif exercise_state == "right":
            feedback_text = "Keep twisting left!"

    # --- Draw Visual Cues ---
    # Draw line across shoulders to visualize rotation
    cv2.line(image, left_shoulder_2d, right_shoulder_2d, back_line_color, 4)
    # Draw dot on the center of the hip
    cv2.circle(image, center_hip_2d, 10, back_line_color, -1)

    # Display rotation value
    cv2.putText(image, f'Rotation: {rotation_value:.2f}', (center_hip_2d[0] + 15, center_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Back Angle: {int(back_angle)}', (center_hip_2d[0] + 15, center_hip_2d[1] + 25),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text