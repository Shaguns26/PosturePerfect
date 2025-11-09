from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_glute_bridge(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Glute Bridge.
    Checks the hip extension angle (shoulder-hip-knee line).
    Assumes a side view.
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angle: Hip extension (Angle at Hip, should be near 180 at top)
    extension_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)

    # --- Define Thresholds ---
    HIP_TOP_THRESHOLD = 165  # Straight line from shoulder to knee (max extension)
    HIP_BOTTOM_THRESHOLD = 110  # Hips resting on the floor or near start

    # --- Form Correction Cues & UI Coloring ---
    line_color = GOOD_COLOR

    # 1. Check for Over-extension (Hyperextension is bad, but generally < 180 is fine)
    if extension_angle > HIP_TOP_THRESHOLD:
        feedback_text = "Squeeze glutes! Don't arch your lower back."
        line_color = GOOD_COLOR # Don't flag as bad unless extreme

    # 2. Count Reps (State Machine)

    # At top (max extension)
    if extension_angle > HIP_TOP_THRESHOLD:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Good squeeze! Lower with control."

    # At bottom
    elif extension_angle < HIP_BOTTOM_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete! Drive hips up."

    # In between, not high enough
    elif exercise_state == "down" and extension_angle > HIP_BOTTOM_THRESHOLD:
        feedback_text = "Push your hips higher!"
        line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Draw body line (Shoulder-Hip-Knee)
    cv2.line(image, left_shoulder_2d, left_hip_2d, line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, line_color, 4)

    # Draw circles on joints
    cv2.circle(image, left_hip_2d, 10, line_color, -1)
    cv2.circle(image, left_shoulder_2d, 10, line_color, -1)
    cv2.circle(image, left_knee_2d, 10, line_color, -1)

    # Display angles
    cv2.putText(image, f'Hip Ext: {int(extension_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text