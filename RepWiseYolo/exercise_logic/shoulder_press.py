from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, \
    TEXT_COLOR


def process_shoulder_press(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Shoulder Press (Seated or Standing).
    Checks for back lean and rep range.
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")  # For back angle

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    shoulder_angle = calculate_angle(left_elbow_3d, left_shoulder_3d, left_hip_3d)  # Measures overhead
    back_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)  # Checks for lean

    # --- Define Thresholds ---
    SHOULDER_OVERHEAD_THRESHOLD = 160  # Top of press
    SHOULDER_RACK_THRESHOLD = 100  # Bottom (racked)
    BACK_STRAIGHT_THRESHOLD = 150  # Min angle for straight back (prevent lean)

    # --- Form Correction & UI Coloring ---
    arm_line_color = GOOD_COLOR
    back_line_color = GOOD_COLOR

    # 1. Check for Back Lean
    if back_angle < BACK_STRAIGHT_THRESHOLD:
        feedback_text = "Don't lean back! Keep core tight."
        back_line_color = BAD_COLOR
    else:
        feedback_text = "Good posture!"
        back_line_color = GOOD_COLOR

    # 2. Count Reps (State Machine)

    # At bottom (racked)
    if shoulder_angle < SHOULDER_RACK_THRESHOLD and back_angle > BACK_STRAIGHT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Press overhead!"

    # At top (overhead)
    elif shoulder_angle > SHOULDER_OVERHEAD_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"

    # At top, waiting
    elif exercise_state == "up" and shoulder_angle > SHOULDER_OVERHEAD_THRESHOLD:
        if "lean" not in feedback_text:
            feedback_text = "Lower to shoulders."

    # --- Draw Visual Cues ---
    # Arm line
    cv2.line(image, left_shoulder_2d, left_elbow_2d, arm_line_color, 4)
    cv2.line(image, left_elbow_2d, left_wrist_2d, arm_line_color, 4)

    # Back line (for lean)
    cv2.line(image, left_shoulder_2d, left_hip_2d, back_line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, back_line_color, 4)

    # Draw circles
    cv2.circle(image, left_elbow_2d, 10, arm_line_color, -1)
    cv2.circle(image, left_hip_2d, 10, back_line_color, -1)

    # Display angles
    cv2.putText(image, f'Shoulder: {int(shoulder_angle)}', (left_shoulder_2d[0] + 15, left_shoulder_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Back: {int(back_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text