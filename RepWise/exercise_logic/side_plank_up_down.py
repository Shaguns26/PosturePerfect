from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_side_plank_up_down(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for the Side Plank Up Down (Hip Dips).
    Checks vertical hip position relative to the shoulder for range of motion.
    Assumes side view, and user is on the left elbow/side.
    """

    # Get 3D coordinates (using left side as the support side)
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)

    # Vertical position check
    # Hip Y-coordinate relative to the Shoulder Y-coordinate
    # Lower Y means higher up on the screen
    hip_vertical_diff = left_hip_2d[1] - left_shoulder_2d[1]

    # Angle check for straight body line (shoulder-hip-ankle) - Should be close to 180 (straight)
    body_line_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_ankle_3d)

    # --- Define Thresholds ---
    HIP_TOP_THRESHOLD = 0  # Hip is level with shoulder (max height)
    HIP_BOTTOM_THRESHOLD = 150  # Hip has dipped down (low point)
    BODY_STRAIGHT_THRESHOLD = 160 # For form correction

    # --- Form Correction ---
    line_color = GOOD_COLOR

    if body_line_angle < BODY_STRAIGHT_THRESHOLD:
        feedback_text = "Keep a straight body line! Squeeze glutes."
        line_color = BAD_COLOR
    else:
        line_color = GOOD_COLOR


    # --- Rep Counting (State Machine) ---
    # State: up (top of movement), down (bottom of movement/dip)

    # 1. At bottom (hip dipped)
    if hip_vertical_diff > HIP_BOTTOM_THRESHOLD and body_line_angle > BODY_STRAIGHT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Lift hips to the ceiling!"

    # 2. At top (hips raised)
    elif hip_vertical_diff < HIP_TOP_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete! Dip down slowly."

    # 3. In between, not high or low enough
    elif exercise_state == "up" and HIP_TOP_THRESHOLD < hip_vertical_diff < HIP_BOTTOM_THRESHOLD:
        feedback_text = "Lower hips for depth."
        line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Draw body line
    cv2.line(image, left_shoulder_2d, left_hip_2d, line_color, 4)
    cv2.line(image, left_hip_2d, left_ankle_2d, line_color, 4)

    # Draw circles
    cv2.circle(image, left_hip_2d, 10, line_color, -1)
    cv2.circle(image, left_shoulder_2d, 10, line_color, -1)

    # Display angle and diff
    cv2.putText(image, f'H-S Diff: {hip_vertical_diff:.0f}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Body Angle: {int(body_line_angle)}', (left_shoulder_2d[0] + 15, left_shoulder_2d[1] + 25),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text