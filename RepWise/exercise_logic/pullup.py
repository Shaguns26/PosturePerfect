from utils import get_landmark_3d, get_landmark_coords, calculate_angle, mp_pose, GOOD_COLOR, BAD_COLOR


def process_pull_up(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Pull Up.
    Checks elbow angle for rep range.
    """

    drawing_specs = {}

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)

    # --- Define Thresholds ---
    ELBOW_TOP_THRESHOLD = 90  # Top of the pull-up
    ELBOW_HANG_THRESHOLD = 160  # Bottom (dead hang)

    # --- Form Correction & UI Coloring ---
    arm_line_color = GOOD_COLOR

    # 1. Count Reps (State Machine)

    # At top of pull
    if elbow_angle < ELBOW_TOP_THRESHOLD:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Good pull! Lower down."

    # At bottom (dead hang)
    elif elbow_angle > ELBOW_HANG_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete!"

    # At bottom, waiting
    elif exercise_state == "down" and elbow_angle > ELBOW_HANG_THRESHOLD:
        feedback_text = "Pull up!"

    # In between (not high enough)
    elif exercise_state == "down" and elbow_angle > ELBOW_TOP_THRESHOLD:
        feedback_text = "Pull higher!"
        arm_line_color = BAD_COLOR

    # Populate drawing_specs
    drawing_specs = {
        "arm_line_color": arm_line_color,
        "left_shoulder_2d": left_shoulder_2d,
        "left_elbow_2d": left_elbow_2d,
        "left_wrist_2d": left_wrist_2d,
        "elbow_angle": int(elbow_angle)
    }

    return rep_counter, exercise_state, feedback_text, drawing_specs