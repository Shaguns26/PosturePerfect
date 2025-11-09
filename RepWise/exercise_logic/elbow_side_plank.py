from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_elbow_side_plank(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for the Elbow Side Plank (Static Hold).
    Focuses on form correction (straight body line) and provides feedback.
    Rep counter is not used for this static hold.
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

    # Angle check for straight body line (shoulder-hip-ankle) - Should be close to 180 (straight)
    body_line_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_ankle_3d)

    # Vertical offset of the hip relative to the shoulder (check for hip sag)
    hip_vertical_diff = left_hip_2d[1] - left_shoulder_2d[1] # Lower Y is higher up on screen

    # --- Define Thresholds ---
    BODY_STRAIGHT_THRESHOLD = 170 # Angle should be near 180
    HIP_SAG_THRESHOLD = 50 # Max vertical sag (in pixels)

    # --- Form Correction ---
    line_color = GOOD_COLOR

    # 1. Check for straight body line
    if body_line_angle < BODY_STRAIGHT_THRESHOLD:
        feedback_text = "Straighten your body line! Push hips forward."
        line_color = BAD_COLOR
    # 2. Check for hip sag (hip being too low)
    elif hip_vertical_diff > HIP_SAG_THRESHOLD:
        feedback_text = "Lift your hips up! Don't let them sag."
        line_color = BAD_COLOR
    else:
        feedback_text = "Perfect plank form! Hold strong."
        line_color = GOOD_COLOR

    # --- Rep Counting (Static Hold) ---
    # Since this is a static hold, we keep the rep counter unchanged.
    # The state machine remains in the initial state or a simplified "holding" state.
    if exercise_state == "up":
        pass
    else:
        exercise_state = "up"


    # --- Draw Visual Cues ---
    # Draw body line
    cv2.line(image, left_shoulder_2d, left_hip_2d, line_color, 4)
    cv2.line(image, left_hip_2d, left_ankle_2d, line_color, 4)

    # Draw circles
    cv2.circle(image, left_hip_2d, 10, line_color, -1)
    cv2.circle(image, left_shoulder_2d, 10, line_color, -1)

    # Display angle and diff
    cv2.putText(image, f'Hold: {int(body_line_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Hip Sag: {int(hip_vertical_diff)}', (left_shoulder_2d[0] + 15, left_shoulder_2d[1] + 25),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text