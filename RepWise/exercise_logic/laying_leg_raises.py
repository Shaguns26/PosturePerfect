from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_laying_leg_raises(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for Laying Leg Raises.
    Checks the hip-knee-ankle angle (for straight legs) and shoulder-hip-knee angle (for lift height).
    """

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")

    # Get 2D coordinates
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)

    # Calculate angles
    # 1. Leg Straightness (Angle at knee)
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)

    # 2. Leg Lift Height (Angle at hip, relative to torso/floor)
    # The shoulder-hip-knee angle measures how far the leg is from the torso line (straight line = 180)
    lift_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)

    # --- Define Thresholds ---
    KNEE_STRAIGHT_THRESHOLD = 170  # Min angle for straight legs (max 180)
    LIFT_PEAK_THRESHOLD = 90  # Legs raised close to 90 degrees (smaller angle = higher lift)
    LOWER_FLOOR_THRESHOLD = 170  # Legs lowered close to floor (max 180)

    # --- Form Correction Cues & UI Coloring ---
    knee_line_color = GOOD_COLOR
    leg_line_color = GOOD_COLOR

    # 1. Check Leg Straightness (Form priority)
    if knee_angle < KNEE_STRAIGHT_THRESHOLD:
        feedback_text = "Straighten your legs! Don't bend your knees."
        knee_line_color = BAD_COLOR
    else:
        feedback_text = "Good leg straightness."
        knee_line_color = GOOD_COLOR

    # 2. Count Reps (State Machine)
    # Ensure form is good before counting rep phases

    # At peak lift
    if lift_angle < LIFT_PEAK_THRESHOLD and knee_angle > KNEE_STRAIGHT_THRESHOLD:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Pause! Lower slowly with control."

    # At bottom (repetition complete)
    elif lift_angle > LOWER_FLOOR_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete! Raise again."

    # In between, not low/high enough
    elif exercise_state == "down" and lift_angle < LOWER_FLOOR_THRESHOLD and lift_angle > LIFT_PEAK_THRESHOLD:
        if "Straighten" not in feedback_text:
            feedback_text = "Raise higher or lower slower!"
        leg_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Draw leg lines
    cv2.line(image, left_hip_2d, left_knee_2d, leg_line_color, 4)
    cv2.line(image, left_knee_2d, left_ankle_2d, knee_line_color, 4)

    # Draw circles
    cv2.circle(image, left_knee_2d, 10, knee_line_color, -1)
    cv2.circle(image, left_hip_2d, 10, leg_line_color, -1)

    # Display angles
    cv2.putText(image, f'Lift: {int(lift_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Knee: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text
