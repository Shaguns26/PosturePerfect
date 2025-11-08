from utils import get_landmark_3d, get_landmark_coords, calculate_angle, mp_pose, GOOD_COLOR, BAD_COLOR


def process_air_squat(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for an Air Squat (Bodyweight Squat).
    This logic is identical to the barbell squat for form checking.
    Calculates angles for depth and back form, counts reps, and provides feedback.
    """

    drawing_specs = {}

    # Get 3D coordinates for angle calculations
    # Using left side, assuming side-on view is best for squats
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")

    # Get 2D coordinates for drawing
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)

    # Calculate angles
    # Knee angle (Hip-Knee-Ankle) for depth
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)
    # Back angle (Shoulder-Hip-Knee) for back form
    back_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)

    # --- Define Thresholds ---
    KNEE_DEPTH_THRESHOLD = 90  # Hips below knees (or parallel)
    KNEE_STRAIGHT_THRESHOLD = 160  # Standing up
    BACK_STRAIGHT_THRESHOLD = 80  # Minimum angle for a straight back (prevent rounding)

    # --- Form Correction Cues & UI Coloring ---
    back_line_color = GOOD_COLOR
    knee_line_color = GOOD_COLOR

    # 1. Check Back Form (Highest Priority)
    if back_angle < BACK_STRAIGHT_THRESHOLD:
        feedback_text = "Chest up! Keep your back straight."
        back_line_color = BAD_COLOR
    else:
        feedback_text = "Good back form!"
        back_line_color = GOOD_COLOR

    # 2. Check Depth & Count Reps (State Machine)

    # At depth and back is straight
    if knee_angle < KNEE_DEPTH_THRESHOLD and back_angle > BACK_STRAIGHT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Good depth! Drive up."

    # Standing up from a squat
    elif knee_angle > KNEE_STRAIGHT_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"

    # Standing, waiting to squat
    elif exercise_state == "up" and knee_angle > KNEE_STRAIGHT_THRESHOLD:
        if "back" not in feedback_text:  # Don't overwrite back feedback
            feedback_text = "Lower into your squat."

    # In between, not at depth
    elif exercise_state == "up" and knee_angle < KNEE_STRAIGHT_THRESHOLD:
        if "back" not in feedback_text:
            feedback_text = "Lower... hit parallel!"
        knee_line_color = BAD_COLOR  # Indicate not deep enough

    # Populate drawing_specs
    drawing_specs = {
        "back_line_color": back_line_color,
        "knee_line_color": knee_line_color,
        "left_shoulder_2d": left_shoulder_2d,
        "left_hip_2d": left_hip_2d,
        "left_knee_2d": left_knee_2d,
        "left_ankle_2d": left_ankle_2d,
        "back_angle": int(back_angle),
        "knee_angle": int(knee_angle)
    }

    return rep_counter, exercise_state, feedback_text, drawing_specs