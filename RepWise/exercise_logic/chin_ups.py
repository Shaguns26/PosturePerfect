from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_chin_ups(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Chin Up.
    Checks elbow angle for rep range (chin above the bar).
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)

    # Use nose/ear height relative to wrist to check chin over bar
    left_ear_2d_y = get_landmark_coords(landmarks, "LEFT_EAR", frame_width, frame_height)[1]
    left_wrist_2d_y = left_wrist_2d[1]

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)

    # --- Define Thresholds ---
    ELBOW_TOP_THRESHOLD = 90  # Max bend at the top of the chin up
    ELBOW_HANG_THRESHOLD = 160  # Fully extended arms (bottom/dead hang)
    CHIN_OVER_BAR_HEIGHT_DIFF = -20  # Ear Y-coord must be significantly HIGHER (smaller Y) than the wrist Y-coord

    # --- Form Correction & UI Coloring ---
    arm_line_color = GOOD_COLOR

    # 1. Check Chin Height (The primary goal of a chin-up)
    is_chin_up = (left_ear_2d_y < left_wrist_2d_y + CHIN_OVER_BAR_HEIGHT_DIFF) and elbow_angle < ELBOW_TOP_THRESHOLD

    # 2. Count Reps (State Machine)

    # At top (chin up)
    if is_chin_up:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Good pull! Lower down slowly."

    # At bottom (dead hang)
    elif elbow_angle > ELBOW_HANG_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete! Pull up."

    # At bottom, waiting
    elif exercise_state == "down" and elbow_angle > ELBOW_HANG_THRESHOLD:
        feedback_text = "Pull up!"

    # In between (not high enough)
    elif exercise_state == "down" and elbow_angle > ELBOW_TOP_THRESHOLD:
        feedback_text = "Pull higher! Get your chin over the bar."
        arm_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Arm line
    cv2.line(image, left_shoulder_2d, left_elbow_2d, arm_line_color, 4)
    cv2.line(image, left_elbow_2d, left_wrist_2d, arm_line_color, 4)

    # Draw circles
    cv2.circle(image, left_elbow_2d, 10, arm_line_color, -1)
    cv2.circle(image, left_shoulder_2d, 10, arm_line_color, -1)

    # Display angles
    cv2.putText(image, f'Elbow: {int(elbow_angle)}', (left_elbow_2d[0] + 15, left_elbow_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    # Display chin height reference
    if is_chin_up:
        # Note: You need a valid 2D coordinate for left_ear_2d_y to use it for text placement.
        # Using shoulder for placement here, but conceptually you'd use the ear's position.
        cv2.putText(image, "CHIN UP!", (left_shoulder_2d[0] - 50, left_shoulder_2d[1] - 30), FONT, 0.7, GOOD_COLOR, 2,
                    cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text