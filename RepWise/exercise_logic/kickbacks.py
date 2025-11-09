from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_kickbacks(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for Glute Kickbacks (on all fours).
    Checks the kickback height (hip extension) on the moving leg (assumes LEFT).
    Requires side view.
    """

    # Get 3D coordinates (using LEFT leg for movement)
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")

    # Get 2D coordinates
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)


    # Calculate angles
    # 1. Kickback Angle (Shoulder-Hip-Knee) - Angle opens as leg raises behind
    kickback_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)

    # 2. Knee Angle (Hip-Knee-Ankle) - Should be maintained near 90 degrees for bent-knee variation
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)

    # --- Define Thresholds ---
    KICK_MAX_THRESHOLD = 170 # Max extension (angle opens up)
    KICK_START_THRESHOLD = 90  # Starting position (knee under hip, angle is small/near 90)
    KNEE_MIN_BEND = 70 # Minimum acceptable bent knee angle
    KNEE_MAX_BEND = 140 # Maximum acceptable bent knee angle

    # --- Form Correction Cues & UI Coloring ---
    leg_line_color = GOOD_COLOR

    # 1. Check Knee Form (Keep knee bent)
    if knee_angle < KNEE_MIN_BEND or knee_angle > KNEE_MAX_BEND:
        feedback_text = "Maintain a controlled knee bend."
        leg_line_color = BAD_COLOR
    else:
        feedback_text = "Good position."

    # 2. Count Reps (State Machine)

    # At top (max kick)
    if kickback_angle > KICK_MAX_THRESHOLD and KNEE_MIN_BEND < knee_angle < KNEE_MAX_BEND:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Squeeze and hold! Lower slowly."

    # At bottom (starting position)
    elif kickback_angle < KICK_START_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete! Kick back."

    # In between, not high enough
    elif exercise_state == "down" and kickback_angle > KICK_START_THRESHOLD:
        if "Maintain" not in feedback_text:
            feedback_text = "Kick higher and squeeze glutes."
        leg_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Draw leg lines
    cv2.line(image, left_hip_2d, left_knee_2d, leg_line_color, 4)
    cv2.line(image, left_knee_2d, left_ankle_2d, leg_line_color, 4)

    # Draw circles on joints
    cv2.circle(image, left_hip_2d, 10, leg_line_color, -1)
    cv2.circle(image, left_knee_2d, 10, leg_line_color, -1)

    # Display angles
    cv2.putText(image, f'Kick Angle: {int(kickback_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Knee Angle: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text