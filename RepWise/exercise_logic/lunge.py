from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_lunge(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Forward Lunge.
    Checks knee depth and torso uprightness.
    """

    # Using right side for the front leg (assumes side-on view, right leg leads)
    front_knee_3d = get_landmark_3d(landmarks, "RIGHT_KNEE")
    front_hip_3d = get_landmark_3d(landmarks, "RIGHT_HIP")
    front_ankle_3d = get_landmark_3d(landmarks, "RIGHT_ANKLE")

    rear_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    rear_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    rear_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")

    # Get 2D coordinates for drawing
    front_knee_2d = get_landmark_coords(landmarks, "RIGHT_KNEE", frame_width, frame_height)
    front_ankle_2d = get_landmark_coords(landmarks, "RIGHT_ANKLE", frame_width, frame_height)
    rear_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)  # For torso drawing

    # Calculate angles
    front_knee_angle = calculate_angle(front_hip_3d, front_knee_3d, front_ankle_3d)  # Front knee depth
    torso_angle = calculate_angle(rear_shoulder_3d, rear_hip_3d, rear_knee_3d)  # Torso straightness

    # --- Define Thresholds ---
    KNEE_DEPTH_THRESHOLD = 95  # Front knee angle at the bottom (near 90 degrees)
    KNEE_STRAIGHT_THRESHOLD = 160  # Standing up
    TORSO_UPRIGHT_THRESHOLD = 150  # Maintain upright torso

    # --- Form Correction Cues & UI Coloring ---
    front_knee_line_color = GOOD_COLOR
    torso_line_color = GOOD_COLOR

    # 1. Check Torso Uprightness
    if torso_angle < TORSO_UPRIGHT_THRESHOLD:
        feedback_text = "Torso straight! Don't lean forward."
        torso_line_color = BAD_COLOR
    else:
        feedback_text = "Good torso position."
        torso_line_color = GOOD_COLOR

    # 2. Check Depth & Count Reps (State Machine)

    # At depth and torso is straight
    if front_knee_angle < KNEE_DEPTH_THRESHOLD and torso_angle > TORSO_UPRIGHT_THRESHOLD:
        if exercise_state == "up":
            exercise_state = "down"
            feedback_text = "Good depth! Drive up."

    # Standing up
    elif front_knee_angle > KNEE_STRAIGHT_THRESHOLD and exercise_state == "down":
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete! Switch legs."

    # Standing, waiting
    elif exercise_state == "up" and front_knee_angle > KNEE_STRAIGHT_THRESHOLD:
        if "Torso" not in feedback_text:
            feedback_text = "Step forward and lower."

    # In between, not at depth
    elif exercise_state == "up" and front_knee_angle < KNEE_STRAIGHT_THRESHOLD:
        if "Torso" not in feedback_text:
            feedback_text = "Lower the back knee further."
        front_knee_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Front Knee line
    cv2.line(image, front_knee_2d, front_ankle_2d, front_knee_line_color, 4)
    cv2.circle(image, front_knee_2d, 10, front_knee_line_color, -1)

    # Torso line (Hip to Knee)
    cv2.line(image, rear_hip_2d, front_knee_2d, torso_line_color,
             4)  # Connects rear hip to front knee for a torso line reference
    cv2.circle(image, rear_hip_2d, 10, torso_line_color, -1)

    # Display angles
    cv2.putText(image, f'Front Knee: {int(front_knee_angle)}', (front_knee_2d[0] + 15, front_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Torso: {int(torso_angle)}', (rear_hip_2d[0] + 15, rear_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text