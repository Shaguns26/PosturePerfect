import time
from utils import get_landmark_3d, get_landmark_coords, calculate_angle, GOOD_COLOR, BAD_COLOR, cv2, FONT, TEXT_COLOR

# Define a constant for the initial/stopped state time
PLANK_STOPPED = 0.0


def process_plank(image, landmarks, frame_width, frame_height, total_held_duration_base, plank_start_time,
                  feedback_text):
    """
    Processes the logic for the Plank hold.
    total_held_duration_base: Total time held BEFORE the current segment started (if running) or total accumulated time (if paused).
    plank_start_time: The timestamp (float) when the current holding segment began. PLANK_STOPPED if paused.

    Returns: (new_total_held_duration_base, new_plank_start_time, feedback_text, speech_text)
    """
    speech_text = ""
    current_time = time.time()

    # --- Check Pose Detectability ---
    try:
        # Check Left Hip (11) and Left Ankle (15) confidence
        hip_conf = landmarks[11][2]
        ankle_conf = landmarks[15][2]
        is_form_detectable = hip_conf > 0.5 and ankle_conf > 0.5
    except IndexError:
        is_form_detectable = False

    # --- Get Coordinates and Angles ---
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")

    # 2D coordinates for drawing
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_ankle_2d = get_landmark_coords(landmarks, "LEFT_ANKLE", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    hip_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, get_landmark_3d(landmarks, "LEFT_WRIST"))
    is_elbow_plank = elbow_angle < 130

    STRAIGHT_BACK_MIN = 170
    STRAIGHT_BACK_MAX = 185
    hip_line_color = GOOD_COLOR
    elbow_line_color = GOOD_COLOR
    current_feedback = ""
    is_good_form = True

    # --- Form Check ---
    if is_form_detectable:
        if hip_angle < STRAIGHT_BACK_MIN:
            current_feedback = "Hips too low! Engage your core and glutes."
            speech_text = "Hips up!"
            hip_line_color = BAD_COLOR
            is_good_form = False
        elif hip_angle > STRAIGHT_BACK_MAX:
            current_feedback = "Hips too high! Flatten your back and look down."
            speech_text = "Hips down."
            hip_line_color = BAD_COLOR
            is_good_form = False

        if is_elbow_plank and not (70 < elbow_angle < 120):
            current_feedback += (" Check elbow position." if current_feedback else "Elbow position needs adjustment.")
            elbow_line_color = BAD_COLOR
            is_good_form = False

        if not current_feedback:
            current_feedback = "Perfect form! Keep holding strong."

    # --- Timer State Machine ---
    new_plank_start_time = plank_start_time
    new_total_held_duration_base = total_held_duration_base

    # Calculate current segment time (0 if paused)
    time_since_segment_start = current_time - plank_start_time if plank_start_time > PLANK_STOPPED else 0.0

    # Calculate total duration for reporting
    duration_to_report = total_held_duration_base + time_since_segment_start

    # State 1: PAUSED / RESET (plank_start_time == PLANK_STOPPED)
    if plank_start_time == PLANK_STOPPED:
        if is_form_detectable and is_good_form:
            # TRANSITION: PAUSED -> RUNNING
            new_plank_start_time = current_time  # Record the start timestamp
            current_feedback = f"RUNNING: Timer resumed! Maintain this form."
            speech_text = "Timer running."
        else:
            # Stays PAUSED. Report accumulated time.
            current_feedback = f"PAUSED. Total held: {format_duration(duration_to_report)}"
            if not is_form_detectable:
                current_feedback = "POSE LOST: " + current_feedback
            elif not is_good_form:
                current_feedback = f"FORM BREAK: " + current_feedback

    # State 2: RUNNING (plank_start_time > PLANK_STOPPED)
    else:
        if is_form_detectable and is_good_form:
            # Keep RUNNING. Only duration_to_report updates (handled by main.py passing the live time).
            current_feedback = f"HOLDING: {format_duration(duration_to_report)} / {current_feedback}"
        else:
            # TRANSITION: RUNNING -> PAUSED

            # The new base duration is the total time held up to this pause moment.
            new_total_held_duration_base = duration_to_report

            # Pause the timer by setting start time to 0.0
            new_plank_start_time = PLANK_STOPPED

            fail_reason = "Pose lost." if not is_form_detectable else "Form failed."
            current_feedback = f"PAUSED! {fail_reason}. Total: {format_duration(duration_to_report)}"
            speech_text = "Stop. Form break."

    # --- Draw Visual Cues (Drawing logic remains the same) ---
    cv2.line(image, left_shoulder_2d, left_hip_2d, hip_line_color, 4)
    cv2.line(image, left_hip_2d, left_knee_2d, hip_line_color, 4)

    if is_elbow_plank:
        cv2.line(image, left_shoulder_2d, left_elbow_2d, elbow_line_color, 4)
        cv2.circle(image, left_elbow_2d, 10, elbow_line_color, -1)

    cv2.circle(image, left_shoulder_2d, 10, hip_line_color, -1)
    cv2.circle(image, left_hip_2d, 10, hip_line_color, -1)
    cv2.circle(image, left_ankle_2d, 10, GOOD_COLOR, -1)

    cv2.putText(image, f'Hip Angle: {int(hip_angle)}', (left_hip_2d[0] + 15, left_hip_2d[1]),
                FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    # If running, total_held_duration_base is passed back unchanged.
    # If paused, total_held_duration_base is updated to the accumulated time.
    return new_total_held_duration_base, new_plank_start_time, current_feedback, speech_text


def format_duration(seconds):
    """Formats seconds into mm:ss.ms string, including milliseconds."""
    minutes = int(seconds // 60)
    # Use f-string formatting to control seconds and milliseconds precision (3 decimal places)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:06.3f}"