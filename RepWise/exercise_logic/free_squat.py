from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_air_squat(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for Air Squats (Free Squats).
    Checks knee depth and back angle.
    Assumes angled side view.
    """
    speech_text = ""

    # Get 3D coordinates (using LEFT side for angled view)
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_ankle_3d = get_landmark_3d(landmarks, "LEFT_ANKLE")
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")

    # Get 2D coordinates for drawing (using LEFT side)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)

    # Calculate angles
    # 1. Knee Angle (Hip-Knee-Ankle): Used for depth (90 degrees is parallel)
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)

    # 2. Torso Angle (Shoulder-Hip-Knee): Used for back/torso lean (should stay relatively open)
    torso_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)

    # --- Define Thresholds ---
    KNEE_PARALLEL_THRESHOLD = 95  # Angle for achieving depth (near 90 degrees)
    KNEE_TOP_THRESHOLD = 165  # Angle for standing up/lockout (near 180 degrees)
    TORSO_LEAN_MAX = 100  # Max torso angle (prevents excessive forward lean)

    # State tracking and form validation
    is_upright_torso = torso_angle > TORSO_LEAN_MAX

    # --- Form Correction Cues & UI Coloring ---
    knee_line_color = GOOD_COLOR
    hip_line_color = GOOD_COLOR
    current_feedback = ""

    # 1. Check Torso Lean (Back Safety) - Priority check
    if not is_upright_torso:
        current_feedback = "Too much forward lean! Chest up."
        speech_text = "Lift chest."
        hip_line_color = BAD_COLOR

    # 2. Rep Counting (State Machine)

    # State 1: UP (Ready to start or Rep Complete)
    if exercise_state == "up":
        if knee_angle > KNEE_TOP_THRESHOLD:
            # Fully standing, ready to start
            if current_feedback == "":
                current_feedback = "Ready! Squat down to begin rep."
            if rep_counter == 0 and speech_text == "":
                speech_text = "Squat down to start."

            # TRANSITION: UP -> DOWN (Start squatting)
            if knee_angle < KNEE_TOP_THRESHOLD - 5 and is_upright_torso:
                exercise_state = "down"
                current_feedback = "Hips back and down. Don't let knees cave in."
                speech_text = "Squat."

        else:
            # Not fully locked out
            current_feedback = "Stand up fully (Knee angle: " + str(int(knee_angle)) + ")"
            knee_line_color = BAD_COLOR

    # State 2: DOWN (Rep in progress - focusing on achieving depth)
    elif exercise_state == "down":
        if knee_angle < KNEE_PARALLEL_THRESHOLD:
            # REACHED DEPTH: Now transition to RECOVERING state
            exercise_state = "recovering"
            if current_feedback == "":
                current_feedback = "Good depth! Drive up through your heels."
                if speech_text == "":
                    speech_text = "Drive up."
        elif knee_angle > KNEE_PARALLEL_THRESHOLD:
            # Not low enough
            if current_feedback == "":
                current_feedback = "Squat deeper to hit parallel."
                if speech_text == "":
                    speech_text = "Deeper."
                knee_line_color = BAD_COLOR

    # State 3: RECOVERING (Coming up from the bottom)
    elif exercise_state == "recovering":
        # Check for full lockout (Rep completion)
        if knee_angle > KNEE_TOP_THRESHOLD and is_upright_torso:
            # TRANSITION: RECOVERING -> UP (Rep Count)
            exercise_state = "up"
            rep_counter += 1
            current_feedback = "Rep Complete! Reset and squat again."
            speech_text = "Rep complete."
        else:
            # Still coming up or stopped short
            if current_feedback == "":
                current_feedback = "Keep pushing up to lockout."
                knee_line_color = BAD_COLOR

    # Apply form cue if necessary, otherwise use the state feedback
    feedback_text = current_feedback if current_feedback else feedback_text

    # --- Draw Visual Cues ---
    # Draw body lines (Hip -> Knee -> Ankle for Squat)
    cv2.line(image, left_hip_2d, left_knee_2d, knee_line_color, 4)
    cv2.line(image, left_knee_2d, left_ankle_2d, knee_line_color, 4)
    # Draw Torso line (Shoulder -> Hip)
    cv2.line(image, left_shoulder_2d, left_hip_2d, hip_line_color, 4)

    # Draw circles on joints
    cv2.circle(image, left_hip_2d, 10, hip_line_color, -1)
    cv2.circle(image, left_knee_2d, 10, knee_line_color, -1)

    # Display angles
    cv2.putText(image, f'Knee: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Torso: {int(torso_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text, speech_text