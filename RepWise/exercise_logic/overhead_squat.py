from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_overhead_squat(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for the Pole Overhead Squat.
    Checks knee depth, back straightness, and arm lockout/verticality.
    Assumes side view.
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d) # Squat depth
    back_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d) # Torso lean/back straightness
    arm_lockout_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d) # Arm straightness

    # --- Define Thresholds ---
    KNEE_DEPTH_THRESHOLD = 90  # Hips below parallel
    KNEE_STRAIGHT_THRESHOLD = 160  # Standing up
    BACK_STRAIGHT_THRESHOLD = 80  # Min angle for a straight back
    ARM_LOCKOUT_THRESHOLD = 165  # Arms must be straight (near 180)

    # --- Form Correction Cues & UI Coloring ---
    back_line_color = GOOD_COLOR
    knee_line_color = GOOD_COLOR
    arm_line_color = GOOD_COLOR

    # 1. Check Arm Lockout (Highest priority for OH Squat)
    if arm_lockout_angle < ARM_LOCKOUT_THRESHOLD:
        feedback_text = "Lock your elbows! Keep arms straight."
        arm_line_color = BAD_COLOR

    # 2. Check Back Form
    elif back_angle < BACK_STRAIGHT_THRESHOLD:
        feedback_text = "Chest up! Keep your back straight."
        back_line_color = BAD_COLOR

    else:
        feedback_text = "Ready to squat."

    # 3. Check Depth & Count Reps (State Machine)

    # At depth, with good form
    if knee_angle < KNEE_DEPTH_THRESHOLD and back_angle > BACK_STRAIGHT_THRESHOLD and arm_lockout_angle > ARM_LOCKOUT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Good depth! Drive up."

    # Standing up from a squat
    elif knee_angle > KNEE_STRAIGHT_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"

    # In between, not at depth
    elif exercise_state == "up" and knee_angle < KNEE_STRAIGHT_THRESHOLD and knee_angle > KNEE_DEPTH_THRESHOLD:
        if "Lock your elbows" not in feedback_text and "straight" not in feedback_text:
            feedback_text = "Lower deeper, keeping the pole overhead!"
        knee_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Draw body lines
    cv2.line(image, left_shoulder_2d, left_hip_2d, back_line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, knee_line_color, 4)
    cv2.line(image, left_knee_2d, get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height), knee_line_color, 4)

    # Draw arm lines
    cv2.line(image, left_shoulder_2d, get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height), arm_line_color, 4)
    cv2.line(image, get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height), get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height), arm_line_color, 4)

    # Draw circles
    cv2.circle(image, left_hip_2d, 10, back_line_color, -1)
    cv2.circle(image, left_knee_2d, 10, knee_line_color, -1)
    cv2.circle(image, left_shoulder_2d, 10, arm_line_color, -1)

    # Display angles
    cv2.putText(image, f'Knee: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Arm Lock: {int(arm_lockout_angle)}', (left_shoulder_2d[0] + 15, left_shoulder_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text