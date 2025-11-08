from utils import get_landmark_3d, get_landmark_coords, calculate_angle, mp_pose, GOOD_COLOR, BAD_COLOR


def process_deadlift(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Deadlift.
    Checks for hip hinge vs. squat and back straightness.
    """

    drawing_specs = {}

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)

    # Calculate angles
    hip_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)  # Measures hip hinge
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)  # Measures knee bend

    # --- Define Thresholds ---
    HIP_HINGE_THRESHOLD = 90  # Hips hinged over
    HIP_STRAIGHT_THRESHOLD = 160  # Standing up
    KNEE_BEND_THRESHOLD = 130  # Max knee bend for a good hinge (not a squat)
    KNEE_STRAIGHT_THRESHOLD = 160  # Standing up

    # --- Form Correction & UI Coloring ---
    hip_line_color = GOOD_COLOR
    knee_line_color = GOOD_COLOR

    # 1. Check for Squatting (Bad Form)
    if hip_angle < HIP_HINGE_THRESHOLD and knee_angle < KNEE_BEND_THRESHOLD:
        feedback_text = "Don't squat! Hinge at your hips."
        hip_line_color = BAD_COLOR
        knee_line_color = BAD_COLOR
    else:
        feedback_text = "Good hinge!"
        hip_line_color = GOOD_COLOR
        knee_line_color = GOOD_COLOR

    # 2. Count Reps (State Machine)

    # At bottom of hinge with good form
    if hip_angle < HIP_HINGE_THRESHOLD and knee_angle > KNEE_BEND_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Good position! Drive up."

    # Standing up (lockout)
    elif hip_angle > HIP_STRAIGHT_THRESHOLD and knee_angle > KNEE_STRAIGHT_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete! Lockout."

    # Standing, waiting
    elif exercise_state == "up" and hip_angle > HIP_STRAIGHT_THRESHOLD:
        if "squat" not in feedback_text:  # Don't overwrite bad form cue
            feedback_text = "Hinge at your hips to lower."

    # Populate drawing_specs
    drawing_specs = {
        "hip_line_color": hip_line_color,
        "knee_line_color": knee_line_color,
        "left_shoulder_2d": left_shoulder_2d,
        "left_hip_2d": left_hip_2d,
        "left_knee_2d": left_knee_2d,
        "left_ankle_2d": left_ankle_2d,
        "hip_angle": int(hip_angle),
        "knee_angle": int(knee_angle)
    }

    return rep_counter, exercise_state, feedback_text, drawing_specs