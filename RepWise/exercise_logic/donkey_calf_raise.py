from ..utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_donkey_calf_raise(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a Donkey Calf Raise.
    Checks ankle angle for height and hip angle for hinge position.
    """

    # Get 3D coordinates
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")
    left_foot_index_3d = get_landmark_3d(landmarks, "LEFT_FOOT_INDEX")

    # Get 2D coordinates
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)
    left_foot_index_2d = get_landmark_coords(landmarks, "LEFT_FOOT_INDEX", frame_width, frame_height)

    # Calculate angles
    ankle_angle = calculate_angle(left_knee_3d, left_ankle_3d, left_foot_index_3d) # Knee-Ankle-Foot_Index
    hip_angle = calculate_angle(left_ankle_3d, left_hip_3d, left_knee_3d) # Ankle-Hip-Knee (Checks for hinge)

    # --- Define Thresholds ---
    ANKLE_CONTRACTION_THRESHOLD = 90  # Max dorsiflexion/bottom stretch (lower angle = toes down)
    ANKLE_PEAK_THRESHOLD = 150  # Max plantarflexion/top contraction (higher angle = toes up)
    HIP_HINGE_THRESHOLD = 110 # Max angle to be hinged forward

    # --- Form Correction Cues & UI Coloring ---
    ankle_line_color = GOOD_COLOR
    hip_line_color = GOOD_COLOR

    # 1. Check Hinge Position
    if hip_angle > HIP_HINGE_THRESHOLD:
        feedback_text = "Hinge forward! Keep your torso low."
        hip_line_color = BAD_COLOR
    else:
        feedback_text = "Good hinge position."
        hip_line_color = GOOD_COLOR

    # 2. Count Reps (State Machine)
    # At top (contraction)
    if ankle_angle > ANKLE_PEAK_THRESHOLD and hip_angle < HIP_HINGE_THRESHOLD:
        if exercise_state == "down":
            exercise_state = "up"
            feedback_text = "Squeeze! Lower slowly."

    # At bottom (stretch)
    elif ankle_angle < ANKLE_CONTRACTION_THRESHOLD and exercise_state == "up":
        exercise_state = "down"
        rep_counter += 1
        feedback_text = "Rep Complete! Drive up."

    # In between, not high enough
    elif exercise_state == "down" and ankle_angle < ANKLE_PEAK_THRESHOLD:
        if "Hinge" not in feedback_text:
            feedback_text = "Push up onto your toes!"
        ankle_line_color = BAD_COLOR

    # --- Draw Visual Cues ---
    # Ankle line
    cv2.line(image, left_ankle_2d, left_foot_index_2d, ankle_line_color, 4)
    cv2.line(image, left_knee_2d, left_ankle_2d, hip_line_color, 4)

    # Draw circles
    cv2.circle(image, left_ankle_2d, 10, ankle_line_color, -1)

    # Display angles
    cv2.putText(image, f'Ankle: {int(ankle_angle)}', (left_ankle_2d[0] + 15, left_ankle_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text