from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR

# Simple history to track hip height for jump detection
hip_height_history = []
MAX_HISTORY_LEN = 5

def process_jump_squat(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Jump Squat.
    Checks knee depth, back straightness, and uses vertical hip movement for jump detection.
    """

    global hip_height_history

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")

    # Get 2D coordinates
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d) # Depth check
    back_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d) # Back straightness

    # Track hip height (y-coord) for jump detection (lower y is higher up on screen)
    current_hip_y = left_hip_2d[1]
    hip_height_history.append(current_hip_y)
    if len(hip_height_history) > MAX_HISTORY_LEN:
        hip_height_history.pop(0)

    # --- Define Thresholds ---
    KNEE_DEPTH_THRESHOLD = 100  # Squat depth achieved (e.g., parallel)
    KNEE_JUMP_THRESHOLD = 165   # Full knee extension in the air
    BACK_STRAIGHT_THRESHOLD = 80 # Minimum angle for a straight back

    # Jump detection criteria (hip moves upwards significantly and rapidly)
    IS_JUMPING = False
    if len(hip_height_history) == MAX_HISTORY_LEN:
        # Check if hip is moving upwards (y-coord decreasing) quickly
        # This simple check confirms the hip is higher than a few frames ago
        if current_hip_y < min(hip_height_history[:-2]) and knee_angle > KNEE_JUMP_THRESHOLD:
            IS_JUMPING = True

    # --- Form Correction Cues & UI Coloring ---
    back_line_color = GOOD_COLOR
    knee_line_color = GOOD_COLOR

    # 1. Check Back Form
    if back_angle < BACK_STRAIGHT_THRESHOLD:
        feedback_text = "Keep your back straight!"
        back_line_color = BAD_COLOR

    # 2. Count Reps (State Machine)

    # LANDED/STANDING UP: Ready to start squat (reset state)
    if knee_angle > KNEE_JUMP_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete! Absorb and sink."

    # SQUATTING PHASE: Going down
    elif knee_angle < KNEE_DEPTH_THRESHOLD and back_angle > BACK_STRAIGHT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Drive up explosively!"

    # JUMP PHASE: In the air
    elif IS_JUMPING and exercise_state == "down":
        knee_line_color = GOOD_COLOR
        feedback_text = "EXPLODE!"

    # In between, not at depth
    elif exercise_state == "up" and knee_angle > KNEE_DEPTH_THRESHOLD and knee_angle < KNEE_JUMP_THRESHOLD:
        if "back" not in feedback_text:
            feedback_text = "SQUAT deeper!"
        knee_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Draw skeleton lines (hip to knee, knee to ankle)
    cv2.line(image, left_hip_2d, left_knee_2d, knee_line_color, 4)
    cv2.line(image, left_knee_2d, get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height), knee_line_color, 4)
    cv2.circle(image, left_knee_2d, 10, knee_line_color, -1)
    cv2.circle(image, left_hip_2d, 10, back_line_color, -1)

    # Display angles
    cv2.putText(image, f'Knee: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text