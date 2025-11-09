from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, mp_pose, GOOD_COLOR, BAD_COLOR, cv2, FONT, \
    TEXT_COLOR


def process_pushup(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a pushup.
    Calculates angles, provides feedback, counts reps, and draws cues.
    """

    # Get 3D coordinates for angle calculations
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")

    # Get 2D pixel coordinates for drawing
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    back_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)  # Simplified back angle

    # --- Form Correction Cues & UI Coloring ---
    elbow_line_color = GOOD_COLOR
    back_line_color = GOOD_COLOR
    hip_circle_color = GOOD_COLOR

    # Back straightness
    if back_angle < 160:  # Threshold for straight back
        feedback_text = "Keep your back straight!"
        back_line_color = BAD_COLOR
        hip_circle_color = BAD_COLOR
    else:
        feedback_text = "Good back form!"
        back_line_color = GOOD_COLOR
        hip_circle_color = GOOD_COLOR

    # Elbow depth (for rep counting)
    if elbow_angle < 90 and back_angle > 160:  # Deep enough and back is straight
        exercise_state = "down"
        elbow_line_color = GOOD_COLOR
        feedback_text = "Lower!"

    elif elbow_angle > 160 and exercise_state == "down":  # Back up, rep complete
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"
        elbow_line_color = GOOD_COLOR

    elif elbow_angle > 160 and exercise_state == "up":  # Staying up, ready for next rep
        feedback_text = "Ready to lower!"
        elbow_line_color = GOOD_COLOR
    else:
        elbow_line_color = BAD_COLOR  # Indicate elbows aren't fully locked or deep enough
        if "back" not in feedback_text:  # Don't overwrite critical back feedback
            feedback_text = "Push up or lower!"

    # --- Draw Visual Cues ---
    # Elbow circle
    cv2.circle(image, left_elbow_2d, 10, elbow_line_color, -1)

    # Back lines
    cv2.line(image, left_shoulder_2d, left_hip_2d, back_line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, back_line_color, 4)

    # Hip circle
    cv2.circle(image, left_hip_2d, 10, hip_circle_color, -1)

    # Highlight bad back
    if back_line_color == BAD_COLOR:
        cv2.circle(image, left_hip_2d, 15, BAD_COLOR, -1)  # Larger red circle on hip

    # Display angles
    cv2.putText(image, f'Elbow: {int(elbow_angle)}', (left_elbow_2d[0] + 15, left_elbow_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Back: {int(back_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text