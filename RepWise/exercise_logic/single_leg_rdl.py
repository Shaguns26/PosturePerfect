from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_single_leg_rdl(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Single Legged Romanian Deadlift (RDL).
    Checks the hip hinge depth and standing leg knee stability.
    Assumes side view, and the LEFT leg is the grounded (standing) leg.
    """

    # Get 3D coordinates (Standing/Grounded Leg)
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    # 1. Hinge Angle (Shoulder-Hip-Knee) - Torso/Leg angle. Smaller angle means more hinged.
    hinge_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)

    # 2. Knee Stability (Hip-Knee-Ankle) - Should maintain slight bend (not locked, not squatted)
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)


    # --- Define Thresholds ---
    KNEE_MAX_BEND = 150 # Prevents squatting on standing leg
    KNEE_MIN_BEND = 175 # Prevents locking the standing knee
    HINGE_BOTTOM_THRESHOLD = 110  # Max depth reached (torso low, near parallel)
    HINGE_TOP_THRESHOLD = 170  # Standing up (lockout)

    # --- Form Correction Cues & UI Coloring ---
    hinge_line_color = GOOD_COLOR
    knee_line_color = GOOD_COLOR

    # 1. Check Knee Stability (Standing leg)
    if knee_angle < KNEE_MAX_BEND:
        feedback_text = "Don't squat! Maintain slight bend in standing knee."
        knee_line_color = BAD_COLOR
    elif knee_angle > KNEE_MIN_BEND:
        feedback_text = "Unlock your knee. Maintain slight bend."
        knee_line_color = BAD_COLOR
    else:
        feedback_text = "Good knee stability."


    # 2. Count Reps (State Machine)

    # At bottom (Max hinge)
    if hinge_angle < HINGE_BOTTOM_THRESHOLD and KNEE_MAX_BEND < knee_angle < KNEE_MIN_BEND:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Good stretch! Drive up using glutes."

    # Standing up (lockout)
    elif hinge_angle > HINGE_TOP_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete! Hinge slowly."

    # Standing, waiting
    elif exercise_state == "up" and hinge_angle > HINGE_TOP_THRESHOLD:
        if "Don't squat" not in feedback_text and "Unlock your knee" not in feedback_text:
            feedback_text = "Hinge forward at the hips."

    # --- Draw Visual Cues ---
    # Draw body lines
    cv2.line(image, left_shoulder_2d, left_hip_2d, hinge_line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, knee_line_color, 4)
    cv2.line(image, left_knee_2d, get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height), knee_line_color, 4)


    # Draw circles on joints
    cv2.circle(image, left_hip_2d, 10, hinge_line_color, -1)
    cv2.circle(image, left_knee_2d, 10, knee_line_color, -1)

    # Display angles
    cv2.putText(image, f'Hinge: {int(hinge_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Knee: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text