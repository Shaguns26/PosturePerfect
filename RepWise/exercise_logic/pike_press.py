from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_pike_press(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for the Bodyweight Pike Press.
    Checks elbow angle for depth and hip angle for maintaining the pike position.
    Assumes a side view.
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d) # Press depth
    pike_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d) # Maintains the pike shape (hips high)

    # --- Define Thresholds ---
    ELBOW_PRESS_THRESHOLD = 90  # Max bend at the bottom of the press
    ELBOW_LOCKOUT_THRESHOLD = 160  # Fully extended arms at the top
    PIKE_SHAPE_THRESHOLD = 70  # Min angle to ensure hips are elevated (pike)

    # --- Form Correction Cues & UI Coloring ---
    arm_line_color = GOOD_COLOR
    pike_line_color = GOOD_COLOR

    # 1. Check Pike Shape (Form priority)
    if pike_angle > PIKE_SHAPE_THRESHOLD or pike_angle < 45: # Angle too large means hips dropped (plank)
        feedback_text = "Hips higher! Maintain a tight pike shape."
        pike_line_color = BAD_COLOR
    else:
        feedback_text = "Good pike shape!"
        pike_line_color = GOOD_COLOR


    # 2. Count Reps (State Machine)

    # At bottom (Press depth reached)
    if elbow_angle < ELBOW_PRESS_THRESHOLD and pike_angle < PIKE_SHAPE_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Drive up through your hands!"

    # At top (Lockout)
    elif elbow_angle > ELBOW_LOCKOUT_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete! Lower head slowly."

    # Standing, waiting (holding lockout)
    elif exercise_state == "up" and elbow_angle > ELBOW_LOCKOUT_THRESHOLD:
        if "Hips higher" not in feedback_text:
            feedback_text = "Lower to the floor."

    # --- Draw Visual Cues ---
    # Draw arm line
    cv2.line(image, left_shoulder_2d, left_elbow_2d, arm_line_color, 4)
    cv2.line(image, left_elbow_2d, get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height), arm_line_color, 4)

    # Draw pike line (hip to shoulder)
    cv2.line(image, left_hip_2d, left_shoulder_2d, pike_line_color, 4)

    # Draw circles
    cv2.circle(image, left_elbow_2d, 10, arm_line_color, -1)
    cv2.circle(image, left_hip_2d, 10, pike_line_color, -1)

    # Display angles
    cv2.putText(image, f'Elbow: {int(elbow_angle)}', (left_elbow_2d[0] + 15, left_elbow_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Pike: {int(pike_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text