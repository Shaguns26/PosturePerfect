from utils import get_landmark_3d, get_landmark_coords, calculate_angle, mp_pose, GOOD_COLOR, BAD_COLOR, cv2, FONT, \
    TEXT_COLOR


def process_chest_press(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Chest Press (Dumbbell or Barbell).
    Assumes a side view, checks for elbow flare and rep range.
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    shoulder_angle = calculate_angle(left_elbow_3d, left_shoulder_3d, left_hip_3d)  # Checks elbow flare

    # --- Define Thresholds ---
    ELBOW_BENT_THRESHOLD = 90  # Bottom of the press
    ELBOW_STRAIGHT_THRESHOLD = 160  # Top (lockout)
    SHOULDER_FLARE_THRESHOLD = 90  # Max angle for tucked elbows (prevents injury)

    # --- Form Correction & UI Coloring ---
    elbow_line_color = GOOD_COLOR
    shoulder_line_color = GOOD_COLOR

    # 1. Check for Elbow Flare
    if shoulder_angle > SHOULDER_FLARE_THRESHOLD:
        feedback_text = "Tuck your elbows!"
        shoulder_line_color = BAD_COLOR
    else:
        feedback_text = "Good elbow position!"
        shoulder_line_color = GOOD_COLOR

    # 2. Count Reps (State Machine)

    # At bottom of press
    if elbow_angle < ELBOW_BENT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Press up!"

    # At top (lockout)
    elif elbow_angle > ELBOW_STRAIGHT_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"

    # At top, waiting
    elif exercise_state == "up" and elbow_angle > ELBOW_STRAIGHT_THRESHOLD:
        if "Tuck" not in feedback_text:
            feedback_text = "Lower with control."

    # --- Draw Visual Cues ---
    # Arm line
    cv2.line(image, left_shoulder_2d, left_elbow_2d, elbow_line_color, 4)
    cv2.line(image, left_elbow_2d, left_wrist_2d, elbow_line_color, 4)

    # Shoulder line (for flare)
    cv2.line(image, left_elbow_2d, left_shoulder_2d, shoulder_line_color, 4)
    cv2.line(image, left_shoulder_2d, left_hip_2d, shoulder_line_color, 4)

    # Draw circles
    cv2.circle(image, left_elbow_2d, 10, elbow_line_color, -1)
    cv2.circle(image, left_shoulder_2d, 10, shoulder_line_color, -1)

    # Display angles
    cv2.putText(image, f'Elbow: {int(elbow_angle)}', (left_elbow_2d[0] + 15, left_elbow_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Shoulder: {int(shoulder_angle)}', (left_shoulder_2d[0] + 15, left_shoulder_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text