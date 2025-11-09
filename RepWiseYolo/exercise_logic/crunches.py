from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_crunches(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for Crunches.
    Checks the head-shoulder-hip angle for torso curl/lift.
    """

    # Get 3D coordinates
    left_ear_3d = get_landmark_3d(landmarks, "LEFT_EAR")
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)

    # Calculate angle (Angle at shoulder to measure how much the torso is curling)
    # A smaller angle indicates a tighter curl/crunch.
    curl_angle = calculate_angle(left_ear_3d, left_shoulder_3d, left_hip_3d)

    # --- Define Thresholds ---
    CRUNCH_PEAK_THRESHOLD = 160  # Maximum curl/lift (smaller number means more curl)
    CRUNCH_FLOOR_THRESHOLD = 175  # Torso fully lowered (straight line)

    # --- Form Correction Cues & UI Coloring ---
    torso_line_color = GOOD_COLOR

    # 1. Count Reps (State Machine)

    # At peak contraction/curl
    if curl_angle < CRUNCH_PEAK_THRESHOLD:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Squeeze! Lower slowly."
            torso_line_color = GOOD_COLOR

    # At floor (repetition complete)
    elif curl_angle > CRUNCH_FLOOR_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete! Curl up."
        torso_line_color = GOOD_COLOR

    # In between, not high enough
    elif exercise_state == "down" and curl_angle > CRUNCH_PEAK_THRESHOLD:
        feedback_text = "Curl higher! Lift your shoulders."
        torso_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Torso line
    cv2.line(image, left_shoulder_2d, left_hip_2d, torso_line_color, 4)
    cv2.circle(image, left_shoulder_2d, 10, torso_line_color, -1)
    cv2.circle(image, left_hip_2d, 10, torso_line_color, -1)

    # Display angles
    cv2.putText(image, f'Curl: {int(curl_angle)}', (left_shoulder_2d[0] + 15, left_shoulder_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text