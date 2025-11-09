from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_good_mornings(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for Good Mornings.
    Checks the hip hinge depth and knee stability.
    Assumes side view.
    """
    # Initialize speech text for this frame
    speech_text = ""

    # Get 3D coordinates
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

    # 2. Knee Stability (Hip-Knee-Ankle) - Should be maintained near 175 (slight bend)
    knee_angle = calculate_angle(left_hip_3d, left_knee_3d, left_ankle_3d)

    # --- Define Thresholds ---
    KNEE_BEND_MIN_THRESHOLD = 160
    KNEE_BEND_MAX_THRESHOLD = 178
    # UPDATED: Hinge Angle < 70 degrees for bottom (Requested)
    HINGE_BOTTOM_THRESHOLD = 70
    HINGE_TOP_THRESHOLD = 165
    HINGE_START_THRESHOLD = 158

    # State tracking and form validation
    is_good_knee = KNEE_BEND_MIN_THRESHOLD <= knee_angle <= KNEE_BEND_MAX_THRESHOLD

    # --- Form Correction Cues & UI Coloring ---
    hinge_line_color = GOOD_COLOR
    knee_line_color = GOOD_COLOR

    current_feedback = ""

    # 1. Check Knee Stability (Priority check)
    if knee_angle < KNEE_BEND_MIN_THRESHOLD:
        current_feedback = "Knee bend too deep! Maintain slight bend."
        speech_text = "Less knee bend."
        knee_line_color = BAD_COLOR
    elif knee_angle > KNEE_BEND_MAX_THRESHOLD:
        current_feedback = "Knee locked! Maintain a slight, soft bend."
        speech_text = "Unlock your knee."
        knee_line_color = BAD_COLOR

    # 2. Rep Counting (State Machine)

    # State 1: UP (Ready to start or Rep Complete)
    if exercise_state == "up":
        if hinge_angle > HINGE_TOP_THRESHOLD:
            # Fully standing, ready to start
            if current_feedback == "":
                current_feedback = "Ready! Hinge forward to begin rep."
            if rep_counter == 0 and speech_text == "":
                speech_text = "Hinge forward to start."

            # TRANSITION: UP -> DOWN (Start Hinging)
            if hinge_angle < HINGE_START_THRESHOLD and is_good_knee:
                exercise_state = "down"
                current_feedback = "Lower your chest, maintain a flat back."
                speech_text = "Lower."

        else:
            # FIX: User is bent over (hinge_angle < HINGE_TOP_THRESHOLD) but state is "up"
            if hinge_angle < HINGE_START_THRESHOLD and is_good_knee:
                # If we are already bent past the starting point, immediately transition to "down"
                exercise_state = "down"
                current_feedback = "Continue lowering to hit depth."
                speech_text = "Lower."
            else:
                # User is bent, but not low enough to start the rep, or has bad knee form
                current_feedback = "Stand tall (Hip angle: " + str(int(hinge_angle)) + ")"
                hinge_line_color = BAD_COLOR

    # State 2: DOWN (Rep in progress - focusing on achieving depth)
    elif exercise_state == "down":
        if hinge_angle < HINGE_BOTTOM_THRESHOLD:
            # REACHED DEPTH: Now transition to RECOVERING state
            exercise_state = "recovering"
            if current_feedback == "":
                current_feedback = "Good depth! Drive up slowly using glutes."
                if speech_text == "":
                    speech_text = "Drive up."
        elif hinge_angle > HINGE_BOTTOM_THRESHOLD:
            # Not low enough
            if current_feedback == "":
                current_feedback = "Lower your chest further. Find the stretch."
                if speech_text == "":
                    speech_text = "Bend down."
                hinge_line_color = BAD_COLOR

    # State 3: RECOVERING (Coming up from the bottom)
    elif exercise_state == "recovering":
        # Check for full lockout (Rep completion)
        if hinge_angle > HINGE_TOP_THRESHOLD and is_good_knee:
            # TRANSITION: RECOVERING -> UP (Rep Count)
            exercise_state = "up"
            rep_counter += 1
            current_feedback = "Rep Complete! Hinge forward for the next one."
            speech_text = "Rep complete."
        elif hinge_angle < HINGE_BOTTOM_THRESHOLD:
            # User bounced or went lower again (remain in recovering)
            if current_feedback == "":
                current_feedback = "Drive up! Find the lockout."
        else:
            # Still coming up
            if current_feedback == "":
                current_feedback = "Drive up slowly using glutes."

    # Apply form cue if necessary, otherwise use the state feedback
    feedback_text = current_feedback if current_feedback else feedback_text

    # --- Draw Visual Cues ---
    # Draw body lines
    cv2.line(image, left_shoulder_2d, left_hip_2d, hinge_line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, hinge_line_color, 4)
    cv2.line(image, left_knee_2d, get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height),
             knee_line_color, 4)

    # Draw circles on joints
    cv2.circle(image, left_hip_2d, 10, hinge_line_color, -1)
    cv2.circle(image, left_knee_2d, 10, knee_line_color, -1)

    # Display angles
    cv2.putText(image, f'Hinge: {int(hinge_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, f'Knee: {int(knee_angle)}', (left_knee_2d[0] + 15, left_knee_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text, speech_text