from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR


def process_shoulder_press(image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for Shoulder Press (no weights).
    Checks the elbow extension and arm vertical position.
    Works from front or side view.

    IMPORTANT: This function expects 'landmarks' to be the YOLO keypoints array:
    [[x1, y1, conf1], [x2, y2, conf2], ...] with 17 keypoints (COCO format)

    It uses utils.py wrapper functions (get_landmark_3d, get_landmark_coords)
    which internally map to YOLO_KEYPOINT_MAP indices.
    """
    # Initialize speech text for this frame
    speech_text = ""

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")

    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    right_hip_3d = get_landmark_3d(landmarks, "RIGHT_HIP")

    # Get 2D coordinates for drawing
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)

    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    right_elbow_2d = get_landmark_coords(landmarks, "RIGHT_ELBOW", frame_width, frame_height)
    right_wrist_2d = get_landmark_coords(landmarks, "RIGHT_WRIST", frame_width, frame_height)

    # Calculate angles
    # 1. Elbow Angle (Shoulder-Elbow-Wrist) - Should be ~180° when extended, <130° when lowered
    left_elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_elbow_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)

    # Average both arms
    elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

    # 2. Arm Vertical Position - Check if wrists are above shoulders (for proper press height)
    arm_raised = False
    if left_shoulder_3d and left_wrist_3d:
        # Y coordinate: lower value = higher position in image
        left_raised = left_wrist_3d[1] < left_shoulder_3d[1]
        arm_raised = left_raised

    if right_shoulder_3d and right_wrist_3d:
        right_raised = right_wrist_3d[1] < right_shoulder_3d[1]
        arm_raised = arm_raised or right_raised

    # --- Define Thresholds ---
    ELBOW_EXTENDED_THRESHOLD = 140  # Arms extended overhead
    ELBOW_LOWERED_THRESHOLD = 130  # Arms lowered to shoulder level
    ELBOW_START_THRESHOLD = 145  # Starting position threshold

    # State tracking and form validation
    is_extended = elbow_angle > ELBOW_EXTENDED_THRESHOLD and arm_raised
    is_lowered = elbow_angle < ELBOW_LOWERED_THRESHOLD

    # --- Form Correction Cues & UI Coloring ---
    left_arm_color = GOOD_COLOR
    right_arm_color = GOOD_COLOR

    current_feedback = ""

    # 1. Check Arm Extension Form (Priority check)
    if elbow_angle < 80 and not is_lowered:
        current_feedback = "Extend your arms more!"
        speech_text = "Extend arms."
        left_arm_color = BAD_COLOR
        right_arm_color = BAD_COLOR
    elif is_extended and not arm_raised:
        current_feedback = "Press your arms higher overhead!"
        speech_text = "Press higher."
        left_arm_color = BAD_COLOR
        right_arm_color = BAD_COLOR

    # 2. Rep Counting (State Machine)

    # State 1: UP (Arms extended overhead - Rep Complete)
    if exercise_state == "up":
        if is_extended:
            # Fully extended overhead, ready for next rep
            if current_feedback == "":
                current_feedback = "Ready! Lower your arms to begin rep."
            if rep_counter == 0 and speech_text == "":
                speech_text = "Lower to start."

            # TRANSITION: UP -> DOWN (Start Lowering)
            if elbow_angle < ELBOW_START_THRESHOLD:
                exercise_state = "down"
                current_feedback = "Lower your arms to shoulder level."
                speech_text = "Lower."

        else:
            # FIX: User has arms lowered but state is "up"
            if is_lowered:
                # If we are already lowered, immediately transition to "down"
                exercise_state = "down"
                current_feedback = "Continue lowering, then press up."
                speech_text = "Lower."
            else:
                # User is in between positions
                current_feedback = "Press arms up overhead (Elbow: " + str(int(elbow_angle)) + ")"
                left_arm_color = BAD_COLOR
                right_arm_color = BAD_COLOR

    # State 2: DOWN (Arms lowered - preparing to press up)
    elif exercise_state == "down":
        if is_lowered:
            # REACHED BOTTOM: Now transition to RECOVERING state
            exercise_state = "recovering"
            if current_feedback == "":
                current_feedback = "Good! Now press up overhead."
                if speech_text == "":
                    speech_text = "Press up."
        elif not is_lowered:
            # Not low enough
            if current_feedback == "":
                current_feedback = "Lower your arms more to shoulder level."
                if speech_text == "":
                    speech_text = "Lower."
                left_arm_color = BAD_COLOR
                right_arm_color = BAD_COLOR

    # State 3: RECOVERING (Pressing up from bottom)
    elif exercise_state == "recovering":
        # Check for full extension (Rep completion)
        if is_extended:
            # TRANSITION: RECOVERING -> UP (Rep Count)
            exercise_state = "up"
            rep_counter += 1
            current_feedback = "Rep Complete! Lower for the next one."
            speech_text = "Rep complete."
        elif is_lowered:
            # User went back down again (remain in recovering)
            if current_feedback == "":
                current_feedback = "Press up! Full extension overhead!"
        else:
            # Still pressing up
            if current_feedback == "":
                current_feedback = "Keep pressing up! Lock out overhead!"

    # Apply form cue if necessary, otherwise use the state feedback
    feedback_text = current_feedback if current_feedback else feedback_text

    # --- Draw Visual Cues ---
    # Draw left arm
    if left_shoulder_2d and left_elbow_2d:
        cv2.line(image, left_shoulder_2d, left_elbow_2d, left_arm_color, 4)
    if left_elbow_2d and left_wrist_2d:
        cv2.line(image, left_elbow_2d, left_wrist_2d, left_arm_color, 4)

    # Draw right arm
    if right_shoulder_2d and right_elbow_2d:
        cv2.line(image, right_shoulder_2d, right_elbow_2d, right_arm_color, 4)
    if right_elbow_2d and right_wrist_2d:
        cv2.line(image, right_elbow_2d, right_wrist_2d, right_arm_color, 4)

    # Draw circles on joints
    if left_elbow_2d:
        cv2.circle(image, left_elbow_2d, 10, left_arm_color, -1)
    if right_elbow_2d:
        cv2.circle(image, right_elbow_2d, 10, right_arm_color, -1)

    if left_wrist_2d:
        cv2.circle(image, left_wrist_2d, 10, left_arm_color, -1)
    if right_wrist_2d:
        cv2.circle(image, right_wrist_2d, 10, right_arm_color, -1)

    # Display angles
    if left_elbow_2d:
        cv2.putText(image, f'L Elbow: {int(left_elbow_angle)}', (left_elbow_2d[0] + 15, left_elbow_2d[1]),
                    FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    if right_elbow_2d:
        cv2.putText(image, f'R Elbow: {int(right_elbow_angle)}', (right_elbow_2d[0] + 15, right_elbow_2d[1]),
                    FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return rep_counter, exercise_state, feedback_text, speech_text