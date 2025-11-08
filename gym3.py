import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(a, b, c):
    """
    Calculates the angle between three 3D points.
    a, b, c: Tuples or lists of (x, y, z) coordinates.
    The angle is calculated at point 'b'.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate dot product
    dot_product = np.dot(ba, bc)

    # Calculate magnitudes
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    # Calculate cosine of the angle
    # Add a small epsilon to avoid division by zero
    cosine_angle = dot_product / (mag_ba * mag_bc + 1e-6)

    # Clip values to prevent arccos errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_landmark_coords(landmarks, part_name, image_width, image_height):
    """
    Retrieves the pixel coordinates (x, y) of a specific landmark.
    """
    lm = landmarks[mp_pose.PoseLandmark[part_name].value]
    return (int(lm.x * image_width), int(lm.y * image_height))


def get_landmark_3d(landmarks, part_name):
    """
    Retrieves the 3D coordinates (x, y, z) of a specific landmark.
    """
    lm = landmarks[mp_pose.PoseLandmark[part_name].value]
    return [lm.x, lm.y, lm.z]


# --- Exercise Processing Functions ---

def process_pushup(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes pushup with precise angle specifications.
    Top: Elbow 170°–185° (≥165°), Body line 175°–180° (flag <165°)
    Bottom: Elbow 80°–100° (≤100°), Shoulder abduction 40°–60°
    """
    drawing_specs = {}

    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")

    # Get 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    body_line_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)
    
    # Calculate shoulder abduction (upper arm vs torso)
    # Vector from shoulder to elbow vs vertical reference
    shoulder_to_elbow = np.array(left_elbow_3d) - np.array(left_shoulder_3d)
    vertical_ref = np.array([0, -1, 0])  # Downward vertical
    shoulder_abduction = np.degrees(np.arccos(np.clip(
        np.dot(shoulder_to_elbow, vertical_ref) / np.linalg.norm(shoulder_to_elbow), -1, 1)))

    # Form checking colors
    elbow_color = GOOD_COLOR
    body_color = GOOD_COLOR
    shoulder_color = GOOD_COLOR

    # Body line check (175°-180°, flag <165°)
    if body_line_angle < 165:
        feedback_text = "Keep body straight! Hips sagging/piking"
        body_color = BAD_COLOR
    elif body_line_angle >= 175:
        body_color = GOOD_COLOR
    else:
        body_color = (255, 165, 0)  # Orange for marginal

    # Shoulder abduction check (40°-60°)
    if shoulder_abduction < 40:
        feedback_text = "Elbows too tucked to body!"
        shoulder_color = BAD_COLOR
    elif shoulder_abduction > 60:
        feedback_text = "Elbows too flared out!"
        shoulder_color = BAD_COLOR
    else:
        shoulder_color = GOOD_COLOR

    # Rep counting logic
    if elbow_angle <= 100 and body_line_angle >= 170:  # Valid bottom position
        exercise_state = "down"
        if 80 <= elbow_angle <= 100:
            elbow_color = GOOD_COLOR
            feedback_text = "Good depth! Push up!"
        else:
            elbow_color = (255, 165, 0)  # Orange
            feedback_text = "Push up!"
            
    elif elbow_angle >= 165 and exercise_state == "down":  # Return to top
        if elbow_angle >= 170 and body_line_angle >= 175:  # Perfect lockout
            exercise_state = "up"
            rep_counter += 1
            feedback_text = "Perfect rep! Ready for next"
            elbow_color = GOOD_COLOR
        elif elbow_angle >= 165:  # Acceptable lockout
            exercise_state = "up"
            rep_counter += 1
            feedback_text = "Rep complete!"
            elbow_color = GOOD_COLOR
            
    elif elbow_angle >= 165 and exercise_state == "up":  # Holding top
        if elbow_angle >= 170:
            feedback_text = "Perfect position! Ready to lower"
            elbow_color = GOOD_COLOR
        else:
            feedback_text = "Lock elbows fully!"
            elbow_color = (255, 165, 0)
    else:
        elbow_color = BAD_COLOR
        if body_line_angle < 165:
            pass  # Keep body line feedback
        else:
            feedback_text = "Full range of motion needed!"

    drawing_specs = {
        "elbow_line_color": elbow_color,
        "back_line_color": body_color,
        "hip_circle_color": body_color,
        "shoulder_color": shoulder_color,
        "left_elbow_2d": left_elbow_2d,
        "left_shoulder_2d": left_shoulder_2d,
        "left_hip_2d": left_hip_2d,
        "left_knee_2d": left_knee_2d
    }

    return rep_counter, exercise_state, feedback_text, drawing_specs


def process_incline_press(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes incline press with precise specifications.
    Bottom: Elbow 80°–100°, Shoulder abduction 30°–60° (flag >70°)
    Top: Elbow 165°–180° (≥165°), Side-to-side diff <15°
    """
    drawing_specs = {}
    
    # Get 3D coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    
    # Get 2D coordinates
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    right_elbow_2d = get_landmark_coords(landmarks, "RIGHT_ELBOW", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    
    # Calculate arm angles
    left_arm_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_arm_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)
    avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Calculate shoulder abduction for both arms
    # Vector from shoulder to elbow
    left_shoulder_to_elbow = np.array(left_elbow_3d) - np.array(left_shoulder_3d)
    right_shoulder_to_elbow = np.array(right_elbow_3d) - np.array(right_shoulder_3d)
    
    # Reference vector (torso direction) - approximate as vertical
    torso_ref = np.array([0, -1, 0])
    
    left_abduction = np.degrees(np.arccos(np.clip(
        np.dot(left_shoulder_to_elbow, torso_ref) / np.linalg.norm(left_shoulder_to_elbow), -1, 1)))
    right_abduction = np.degrees(np.arccos(np.clip(
        np.dot(right_shoulder_to_elbow, torso_ref) / np.linalg.norm(right_shoulder_to_elbow), -1, 1)))
    
    # Form checking
    elbow_color = GOOD_COLOR
    symmetry_color = GOOD_COLOR
    
    # Check shoulder abduction (30°-60°, flag >70°)
    if left_abduction > 70 or right_abduction > 70:
        feedback_text = "Elbows too flared! Tuck them in"
        elbow_color = BAD_COLOR
    elif left_abduction < 30 or right_abduction < 30:
        feedback_text = "Elbows too tucked! Slight flare needed"
        elbow_color = BAD_COLOR
    elif 30 <= left_abduction <= 60 and 30 <= right_abduction <= 60:
        elbow_color = GOOD_COLOR
    
    # Check symmetry (side-to-side diff <15°)
    angle_difference = abs(left_arm_angle - right_arm_angle)
    if angle_difference >= 15:
        feedback_text = "Keep arms symmetric!"
        symmetry_color = BAD_COLOR
        elbow_color = BAD_COLOR
    
    # Rep counting logic
    if avg_arm_angle <= 100:  # Bottom position
        if 80 <= avg_arm_angle <= 100:
            exercise_state = "down"
            feedback_text = "Good depth! Press up!"
            if elbow_color == GOOD_COLOR:  # Only if form is good
                elbow_color = GOOD_COLOR
        else:
            feedback_text = "Lower to proper depth!"
            elbow_color = (255, 165, 0)  # Orange
            
    elif avg_arm_angle >= 165 and exercise_state == "down":  # Top position
        if (angle_difference < 15 and 
            30 <= left_abduction <= 60 and 30 <= right_abduction <= 60):
            exercise_state = "up"
            rep_counter += 1
            if avg_arm_angle >= 170:
                feedback_text = "Perfect rep! Lower slowly"
            else:
                feedback_text = "Rep complete! Lower slowly"
            elbow_color = GOOD_COLOR
        else:
            feedback_text = "Fix form before counting rep!"
            elbow_color = BAD_COLOR
            
    elif avg_arm_angle >= 165 and exercise_state == "up":  # Holding top
        feedback_text = "Ready to lower"
        elbow_color = GOOD_COLOR
    else:
        if elbow_color == GOOD_COLOR:  # Don't overwrite form feedback
            feedback_text = "Full range of motion!"
            elbow_color = (255, 165, 0)
    
    drawing_specs = {
        "left_elbow_2d": left_elbow_2d,
        "right_elbow_2d": right_elbow_2d,
        "left_shoulder_2d": left_shoulder_2d,
        "right_shoulder_2d": right_shoulder_2d,
        "elbow_color": elbow_color,
        "symmetry_color": symmetry_color
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


def process_decline_fly(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes decline fly cables with precise specifications.
    Elbow angle: 150°–165° (flag <130° too pressy, >175° hyper-straight)
    Open: wrist distance >0.40, shoulder abduction 40°–60°
    Closed: wrist distance <0.18–0.20
    """
    drawing_specs = {}
    
    # Get coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    
    # 2D coordinates
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    right_wrist_2d = get_landmark_coords(landmarks, "RIGHT_WRIST", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    
    # Calculate arm angles
    left_arm_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_arm_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)
    avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Calculate normalized wrist distance
    wrist_distance = np.sqrt((left_wrist_3d[0] - right_wrist_3d[0])**2 + 
                            (left_wrist_3d[1] - right_wrist_3d[1])**2)
    
    # Calculate shoulder horizontal abduction
    # Vector from shoulder to wrist projected on horizontal plane
    left_shoulder_to_wrist = np.array(left_wrist_3d) - np.array(left_shoulder_3d)
    right_shoulder_to_wrist = np.array(right_wrist_3d) - np.array(right_shoulder_3d)
    
    # Project to horizontal plane (ignore Y component)
    left_horizontal = np.array([left_shoulder_to_wrist[0], 0, left_shoulder_to_wrist[2]])
    right_horizontal = np.array([right_shoulder_to_wrist[0], 0, right_shoulder_to_wrist[2]])
    
    # Reference forward direction
    forward_ref = np.array([0, 0, -1])
    
    left_abduction = np.degrees(np.arccos(np.clip(
        np.dot(left_horizontal, forward_ref) / (np.linalg.norm(left_horizontal) + 1e-6), -1, 1)))
    right_abduction = np.degrees(np.arccos(np.clip(
        np.dot(right_horizontal, forward_ref) / (np.linalg.norm(right_horizontal) + 1e-6), -1, 1)))
    
    # Form checking
    arm_color = GOOD_COLOR
    
    # Check elbow angles (150°–165°)
    if left_arm_angle < 130 or right_arm_angle < 130:
        feedback_text = "Too much elbow bend! More fly, less press"
        arm_color = BAD_COLOR
    elif left_arm_angle > 175 or right_arm_angle > 175:
        feedback_text = "Arms too straight! Slight bend needed"
        arm_color = BAD_COLOR
    elif not (145 <= avg_arm_angle <= 170):
        feedback_text = "Maintain soft elbow bend!"
        arm_color = (255, 165, 0)  # Orange
    
    # Rep counting logic
    if wrist_distance > 0.40:  # Open position
        if (145 <= avg_arm_angle <= 170 and 
            40 <= left_abduction <= 60 and 40 <= right_abduction <= 60):
            exercise_state = "open"
            feedback_text = "Feel the stretch! Bring arms together"
            arm_color = GOOD_COLOR
        else:
            exercise_state = "open"
            if not (40 <= left_abduction <= 60 and 40 <= right_abduction <= 60):
                feedback_text = "Good stretch depth! Fix arm position"
            # Keep existing arm angle feedback
            
    elif wrist_distance < 0.20 and exercise_state == "open":  # Closed position (rep complete)
        if 150 <= avg_arm_angle <= 165:
            exercise_state = "closed"
            rep_counter += 1
            feedback_text = "Perfect rep! Open wide again"
            arm_color = GOOD_COLOR
        else:
            feedback_text = "Fix elbow angle for valid rep!"
            arm_color = BAD_COLOR
            
    elif wrist_distance < 0.20:  # Staying closed
        if 150 <= avg_arm_angle <= 165:
            feedback_text = "Good squeeze! Open arms wide"
            arm_color = GOOD_COLOR
        else:
            # Keep elbow angle feedback
            pass
    else:  # Mid-range
        if arm_color == GOOD_COLOR:  # Don't overwrite form feedback
            feedback_text = "Full range of motion!"
            arm_color = (255, 165, 0)
    
    drawing_specs = {
        "left_wrist_2d": left_wrist_2d,
        "right_wrist_2d": right_wrist_2d,
        "left_shoulder_2d": left_shoulder_2d,
        "right_shoulder_2d": right_shoulder_2d,
        "arm_color": arm_color
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


def process_chest_fly_machine(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes chest fly machine exercise.
    Similar to decline fly but seated position.
    """
    # Use similar logic to decline fly but adjust for seated position
    return process_decline_fly(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text)


def process_tricep_overhead(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes tricep overhead extension with precise specifications.
    Bottom: Elbow 40°–70°, Top: Elbow 170°–180° (≥165°)
    Upper arm: 0°–15° forward from vertical, flag elbow flare >30° from midline
    """
    drawing_specs = {}
    
    # Get coordinates (focusing on one arm for overhead extension)
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    
    # 2D coordinates
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    right_elbow_2d = get_landmark_coords(landmarks, "RIGHT_ELBOW", frame_width, frame_height)
    right_wrist_2d = get_landmark_coords(landmarks, "RIGHT_WRIST", frame_width, frame_height)
    
    # Calculate elbow angles for both arms
    left_elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_elbow_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    # Calculate upper arm position relative to vertical
    left_upper_arm = np.array(left_elbow_3d) - np.array(left_shoulder_3d)
    right_upper_arm = np.array(right_elbow_3d) - np.array(right_shoulder_3d)
    
    # Vertical reference (upward)
    vertical_ref = np.array([0, 1, 0])
    
    # Angle from vertical (should be 0°-15° forward)
    left_vertical_angle = np.degrees(np.arccos(np.clip(
        np.dot(left_upper_arm, vertical_ref) / np.linalg.norm(left_upper_arm), -1, 1)))
    right_vertical_angle = np.degrees(np.arccos(np.clip(
        np.dot(right_upper_arm, vertical_ref) / np.linalg.norm(right_upper_arm), -1, 1)))
    
    # Check elbow flare (distance from midline)
    shoulder_midpoint = [(left_shoulder_3d[0] + right_shoulder_3d[0])/2, 
                        (left_shoulder_3d[1] + right_shoulder_3d[1])/2,
                        (left_shoulder_3d[2] + right_shoulder_3d[2])/2]
    
    left_elbow_flare = abs(left_elbow_3d[0] - shoulder_midpoint[0])
    right_elbow_flare = abs(right_elbow_3d[0] - shoulder_midpoint[0])
    
    # Form checking
    elbow_color = GOOD_COLOR
    
    # Check upper arm position (0°-15° from vertical)
    if left_vertical_angle > 25 or right_vertical_angle > 25:
        feedback_text = "Keep upper arms more vertical! Elbows near ears"
        elbow_color = BAD_COLOR
    elif left_vertical_angle > 20 or right_vertical_angle > 20:
        feedback_text = "Upper arms drifting forward!"
        elbow_color = (255, 165, 0)  # Orange
    
    # Check elbow flare (>30° from midline is flagged)
    if left_elbow_flare > 0.3 or right_elbow_flare > 0.3:  # Normalized distance
        feedback_text = "Elbows flaring out too wide!"
        elbow_color = BAD_COLOR
    
    # Rep counting logic
    if avg_elbow_angle <= 80:  # Bottom position (stretch)
        if 40 <= avg_elbow_angle <= 70:
            exercise_state = "bent"
            feedback_text = "Good stretch! Extend upward!"
            if elbow_color == GOOD_COLOR:  # Only if form is good
                elbow_color = GOOD_COLOR
        elif avg_elbow_angle < 40:
            feedback_text = "Don't go too deep! Risk of injury"
            elbow_color = BAD_COLOR
        else:
            exercise_state = "bent"
            feedback_text = "Lower for full stretch!"
            elbow_color = (255, 165, 0)
            
    elif avg_elbow_angle >= 165 and exercise_state == "bent":  # Top position (lockout)
        if (left_vertical_angle <= 20 and right_vertical_angle <= 20 and
            left_elbow_flare <= 0.25 and right_elbow_flare <= 0.25):
            exercise_state = "extended"
            rep_counter += 1
            if avg_elbow_angle >= 170:
                feedback_text = "Perfect lockout! Lower slowly"
            else:
                feedback_text = "Rep complete! Lower slowly"
            elbow_color = GOOD_COLOR
        else:
            feedback_text = "Fix upper arm position for valid rep!"
            elbow_color = BAD_COLOR
            
    elif avg_elbow_angle >= 165 and exercise_state == "extended":  # Holding top
        feedback_text = "Ready to lower for stretch"
        elbow_color = GOOD_COLOR
    else:
        if elbow_color == GOOD_COLOR:  # Don't overwrite form feedback
            feedback_text = "Full extension needed!"
            elbow_color = (255, 165, 0)
    
    drawing_specs = {
        "left_shoulder_2d": left_shoulder_2d,
        "left_elbow_2d": left_elbow_2d,
        "left_wrist_2d": left_wrist_2d,
        "right_shoulder_2d": right_shoulder_2d,
        "right_elbow_2d": right_elbow_2d,
        "right_wrist_2d": right_wrist_2d,
        "elbow_color": elbow_color
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


def process_tricep_cable_pulldown(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes tricep cable pulldown with precise specifications.
    Top: Elbow 90°–110°, Bottom: Elbow 170°–180° (≥165°)
    Elbows pinned close to body, torso lean 0°–15°
    """
    drawing_specs = {}
    
    # Get coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    right_hip_3d = get_landmark_3d(landmarks, "RIGHT_HIP")
    
    # 2D coordinates
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    right_elbow_2d = get_landmark_coords(landmarks, "RIGHT_ELBOW", frame_width, frame_height)
    
    # Calculate elbow angles
    left_elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_elbow_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    # Calculate torso lean (shoulder to hip angle from vertical)
    torso_center_top = [(left_shoulder_3d[0] + right_shoulder_3d[0])/2,
                       (left_shoulder_3d[1] + right_shoulder_3d[1])/2,
                       (left_shoulder_3d[2] + right_shoulder_3d[2])/2]
    torso_center_bottom = [(left_hip_3d[0] + right_hip_3d[0])/2,
                          (left_hip_3d[1] + right_hip_3d[1])/2,
                          (left_hip_3d[2] + right_hip_3d[2])/2]
    
    torso_vector = np.array(torso_center_top) - np.array(torso_center_bottom)
    vertical_ref = np.array([0, 1, 0])  # Upward vertical
    
    torso_lean_angle = np.degrees(np.arccos(np.clip(
        np.dot(torso_vector, vertical_ref) / np.linalg.norm(torso_vector), -1, 1)))
    
    # Check elbow position relative to torso (should stay close)
    torso_width = np.linalg.norm(np.array(left_shoulder_3d) - np.array(right_shoulder_3d))
    left_elbow_distance = np.linalg.norm(np.array(left_elbow_3d) - np.array(left_shoulder_3d))
    right_elbow_distance = np.linalg.norm(np.array(right_elbow_3d) - np.array(right_shoulder_3d))
    
    # Normalized elbow separation (how far elbows drift from ribs)
    elbow_separation = max(left_elbow_distance, right_elbow_distance) / torso_width
    
    # Form checking
    elbow_color = GOOD_COLOR
    
    # Check torso lean (0°-15° acceptable, slight forward lean)
    if torso_lean_angle > 20:
        feedback_text = "Don't lean too far forward!"
        elbow_color = BAD_COLOR
    elif torso_lean_angle > 15:
        feedback_text = "Slight forward lean is okay"
        elbow_color = (255, 165, 0)  # Orange
    
    # Check elbow position (should stay pinned to ribs)
    if elbow_separation > 0.20:  # Elbows drifting too far
        feedback_text = "Pin elbows closer to body!"
        elbow_color = BAD_COLOR
    elif elbow_separation > 0.15:
        feedback_text = "Keep elbows closer to ribs!"
        elbow_color = (255, 165, 0)
    
    # Rep counting logic
    if 90 <= avg_elbow_angle <= 120:  # Top position (bent)
        exercise_state = "up"
        if 90 <= avg_elbow_angle <= 110:
            feedback_text = "Good starting position! Push down fully!"
            if elbow_color == GOOD_COLOR:  # Only if form is good
                elbow_color = GOOD_COLOR
        else:
            feedback_text = "Start with elbows at 90°!"
            elbow_color = (255, 165, 0)
            
    elif avg_elbow_angle >= 165 and exercise_state == "up":  # Bottom position (extended)
        if (elbow_separation <= 0.15 and torso_lean_angle <= 20):
            exercise_state = "down"
            rep_counter += 1
            if avg_elbow_angle >= 170:
                feedback_text = "Perfect lockout! Control back up"
            else:
                feedback_text = "Rep complete! Control back up"
            elbow_color = GOOD_COLOR
        else:
            feedback_text = "Fix form before counting rep!"
            elbow_color = BAD_COLOR
            
    elif avg_elbow_angle >= 165 and exercise_state == "down":  # Holding bottom
        feedback_text = "Ready to raise for next rep"
        elbow_color = GOOD_COLOR
    else:
        if elbow_color == GOOD_COLOR:  # Don't overwrite form feedback
            feedback_text = "Full extension needed!"
            elbow_color = (255, 165, 0)
    
    drawing_specs = {
        "left_elbow_2d": left_elbow_2d,
        "right_elbow_2d": right_elbow_2d,
        "elbow_color": elbow_color
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


def process_shoulder_press(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes shoulder press with precise specifications.
    Bottom: Elbow 80°–100°, humerus 10°–30° forward (not T-pose)
    Top: Elbow 170°–180° (≥165°), shoulder flexion 170°–185°
    """
    drawing_specs = {}
    
    # Get coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    
    # 2D coordinates
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    right_elbow_2d = get_landmark_coords(landmarks, "RIGHT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    right_wrist_2d = get_landmark_coords(landmarks, "RIGHT_WRIST", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    
    # Calculate elbow angles
    left_arm_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_arm_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)
    avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Calculate humerus position (shoulder to elbow vector)
    left_humerus = np.array(left_elbow_3d) - np.array(left_shoulder_3d)
    right_humerus = np.array(right_elbow_3d) - np.array(right_shoulder_3d)
    
    # Calculate humerus angle from lateral (side) position
    # Lateral reference (pure side abduction)
    lateral_ref = np.array([1, 0, 0])  # Sideways
    
    left_humerus_angle = np.degrees(np.arccos(np.clip(
        np.dot(left_humerus, lateral_ref) / np.linalg.norm(left_humerus), -1, 1)))
    right_humerus_angle = np.degrees(np.arccos(np.clip(
        np.dot(right_humerus, lateral_ref) / np.linalg.norm(right_humerus), -1, 1)))
    
    # Calculate shoulder flexion (how high arms are raised)
    # Vector from shoulder to wrist
    left_shoulder_to_wrist = np.array(left_wrist_3d) - np.array(left_shoulder_3d)
    right_shoulder_to_wrist = np.array(right_wrist_3d) - np.array(right_shoulder_3d)
    
    # Vertical upward reference
    vertical_up = np.array([0, 1, 0])
    
    left_shoulder_flexion = np.degrees(np.arccos(np.clip(
        np.dot(left_shoulder_to_wrist, vertical_up) / np.linalg.norm(left_shoulder_to_wrist), -1, 1)))
    right_shoulder_flexion = np.degrees(np.arccos(np.clip(
        np.dot(right_shoulder_to_wrist, vertical_up) / np.linalg.norm(right_shoulder_to_wrist), -1, 1)))
    
    avg_shoulder_flexion = (left_shoulder_flexion + right_shoulder_flexion) / 2
    
    # Form checking
    elbow_color = GOOD_COLOR
    
    # Check if arms are in T-pose position (bad form)
    if left_humerus_angle < 20 or right_humerus_angle < 20:  # Too lateral
        feedback_text = "Bring elbows forward! Not T-pose position"
        elbow_color = BAD_COLOR
    
    # Check if elbows are too far forward
    elif left_humerus_angle > 60 or right_humerus_angle > 60:
        feedback_text = "Don't bring elbows too far forward!"
        elbow_color = BAD_COLOR
    
    # Check wrist over elbow alignment at bottom
    if avg_arm_angle >= 80 and avg_arm_angle <= 100:
        # Check if wrists are approximately over elbows
        left_wrist_elbow_dist = abs(left_wrist_3d[0] - left_elbow_3d[0])
        right_wrist_elbow_dist = abs(right_wrist_3d[0] - right_elbow_3d[0])
        
        if left_wrist_elbow_dist > 0.1 or right_wrist_elbow_dist > 0.1:
            feedback_text = "Stack wrists over elbows!"
            elbow_color = (255, 165, 0)  # Orange
    
    # Rep counting logic
    if 80 <= avg_arm_angle <= 110:  # Bottom position
        if (30 <= left_humerus_angle <= 50 and 30 <= right_humerus_angle <= 50):
            exercise_state = "down"
            if 80 <= avg_arm_angle <= 100:
                feedback_text = "Good starting position! Press overhead!"
                if elbow_color == GOOD_COLOR:
                    elbow_color = GOOD_COLOR
            else:
                feedback_text = "Lower to proper starting position!"
                elbow_color = (255, 165, 0)
        else:
            exercise_state = "down"
            # Keep humerus position feedback
            
    elif avg_arm_angle >= 165 and exercise_state == "down":  # Top position
        if (avg_shoulder_flexion <= 15 and  # Arms overhead (170°-185° from vertical)
            30 <= left_humerus_angle <= 50 and 30 <= right_humerus_angle <= 50):
            exercise_state = "up"
            rep_counter += 1
            if avg_arm_angle >= 170:
                feedback_text = "Perfect lockout! Lower to shoulders"
            else:
                feedback_text = "Rep complete! Lower to shoulders"
            elbow_color = GOOD_COLOR
        else:
            feedback_text = "Fix arm position for valid rep!"
            elbow_color = BAD_COLOR
            
    elif avg_arm_angle >= 165 and exercise_state == "up":  # Holding top
        if avg_shoulder_flexion <= 15:
            feedback_text = "Good overhead position! Ready to lower"
            elbow_color = GOOD_COLOR
        else:
            feedback_text = "Press more overhead!"
            elbow_color = (255, 165, 0)
    else:
        if elbow_color == GOOD_COLOR:  # Don't overwrite form feedback
            feedback_text = "Full range of motion!"
            elbow_color = (255, 165, 0)
    
    drawing_specs = {
        "left_elbow_2d": left_elbow_2d,
        "right_elbow_2d": right_elbow_2d,
        "left_wrist_2d": left_wrist_2d,
        "right_wrist_2d": right_wrist_2d,
        "left_shoulder_2d": left_shoulder_2d,
        "right_shoulder_2d": right_shoulder_2d,
        "elbow_color": elbow_color
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


def process_lateral_raise(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes lateral raise with precise specifications.
    Elbow: 150°–170° (flag >175° too straight, <140° too bent)
    Top: Humerus abduction 80°–100° (flag >110° too high)
    Bottom: Abduction 0°–20° (or 10°-30° for lean-away variation)
    """
    drawing_specs = {}
    
    # Get coordinates
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    right_shoulder_3d = get_landmark_3d(landmarks, "RIGHT_SHOULDER")
    right_elbow_3d = get_landmark_3d(landmarks, "RIGHT_ELBOW")
    right_wrist_3d = get_landmark_3d(landmarks, "RIGHT_WRIST")
    
    # 2D coordinates
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    right_wrist_2d = get_landmark_coords(landmarks, "RIGHT_WRIST", frame_width, frame_height)
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    right_shoulder_2d = get_landmark_coords(landmarks, "RIGHT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    right_elbow_2d = get_landmark_coords(landmarks, "RIGHT_ELBOW", frame_width, frame_height)
    
    # Calculate arm angles (elbow bend)
    left_arm_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    right_arm_angle = calculate_angle(right_shoulder_3d, right_elbow_3d, right_wrist_3d)
    avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Calculate humerus abduction (shoulder to elbow vector relative to torso)
    left_humerus = np.array(left_elbow_3d) - np.array(left_shoulder_3d)
    right_humerus = np.array(right_elbow_3d) - np.array(right_shoulder_3d)
    
    # Downward reference (arms at sides)
    down_ref = np.array([0, -1, 0])
    
    # Calculate abduction angles
    left_abduction = np.degrees(np.arccos(np.clip(
        np.dot(left_humerus, down_ref) / np.linalg.norm(left_humerus), -1, 1)))
    right_abduction = np.degrees(np.arccos(np.clip(
        np.dot(right_humerus, down_ref) / np.linalg.norm(right_humerus), -1, 1)))
    
    avg_abduction = (left_abduction + right_abduction) / 2
    
    # Form checking
    arm_color = GOOD_COLOR
    
    # Check elbow angles (150°-170°)
    if left_arm_angle > 175 or right_arm_angle > 175:
        feedback_text = "Too straight! Slight bend in elbows!"
        arm_color = BAD_COLOR
    elif left_arm_angle < 140 or right_arm_angle < 140:
        feedback_text = "Too much bend! Straighten arms more!"
        arm_color = BAD_COLOR
    elif not (150 <= avg_arm_angle <= 170):
        feedback_text = "Adjust elbow bend!"
        arm_color = (255, 165, 0)  # Orange
    
    # Check for going too high
    if avg_abduction > 110:
        feedback_text = "Don't raise above shoulder height!"
        arm_color = BAD_COLOR
    elif avg_abduction > 100:
        feedback_text = "Getting too high! Stop at shoulders"
        arm_color = (255, 165, 0)
    
    # Rep counting logic
    if avg_abduction <= 30:  # Bottom position (arms down or lean-away position)
        if avg_abduction <= 20:  # Standard bottom position
            exercise_state = "down"
            if 150 <= avg_arm_angle <= 170:
                feedback_text = "Good starting position! Raise to shoulders"
                if arm_color == GOOD_COLOR:
                    arm_color = GOOD_COLOR
            else:
                feedback_text = "Fix elbow bend before starting!"
                arm_color = BAD_COLOR
        elif 10 <= avg_abduction <= 30:  # Lean-away variation
            exercise_state = "down"
            feedback_text = "Good tension! Raise to shoulder height"
            if arm_color == GOOD_COLOR:
                arm_color = GOOD_COLOR
        else:
            exercise_state = "down"
            feedback_text = "Lower arms more or use lean-away style"
            arm_color = (255, 165, 0)
            
    elif 80 <= avg_abduction <= 100 and exercise_state == "down":  # Top position (shoulder height)
        if 150 <= avg_arm_angle <= 170:
            exercise_state = "up"
            rep_counter += 1
            if 85 <= avg_abduction <= 95:
                feedback_text = "Perfect shoulder height! Lower slowly"
            else:
                feedback_text = "Rep complete! Lower slowly"
            arm_color = GOOD_COLOR
        else:
            feedback_text = "Fix elbow bend for valid rep!"
            arm_color = BAD_COLOR
            
    elif 80 <= avg_abduction <= 100 and exercise_state == "up":  # Holding top
        if 150 <= avg_arm_angle <= 170:
            feedback_text = "Good hold at shoulder height!"
            arm_color = GOOD_COLOR
        else:
            # Keep elbow angle feedback
            pass
    else:
        if arm_color == GOOD_COLOR:  # Don't overwrite form feedback
            if avg_abduction < 80:
                feedback_text = "Raise to shoulder height!"
            else:
                feedback_text = "Lower from current position!"
            arm_color = (255, 165, 0)
    
    drawing_specs = {
        "left_wrist_2d": left_wrist_2d,
        "right_wrist_2d": right_wrist_2d,
        "left_shoulder_2d": left_shoulder_2d,
        "right_shoulder_2d": right_shoulder_2d,
        "left_elbow_2d": left_elbow_2d,
        "right_elbow_2d": right_elbow_2d,
        "arm_color": arm_color
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


# --- Legacy Functions (keeping for compatibility) ---

def process_barbell_squat(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Barbell Squat logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_deadlift(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Deadlift logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_chest_press(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Chest Press logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_pull_up(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Pull Up logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


# --- Main Application Logic ---

# Global state variables
rep_counter = 0
exercise_state = "up"  # Can be "up" or "down"
feedback_text = ""
current_exercise = "pushup"  # Default exercise
drawing_specs = {}  # Dictionary to hold drawing info

# Colors for drawing
GOOD_COLOR = (0, 255, 0)  # Green
BAD_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
OUTLINE_COLOR = (0, 0, 0)  # Black

# Exercise selection function
def change_exercise():
    """Change exercise based on keyboard input"""
    global current_exercise, rep_counter, exercise_state
    rep_counter = 0
    exercise_state = "up"
    
    exercises = [
        "pushup", "incline_press", "decline_fly", "chest_fly_machine",
        "tricep_overhead", "tricep_cable_pulldown", "shoulder_press", "lateral_raise"
    ]
    
    print("\nAvailable exercises:")
    for i, ex in enumerate(exercises):
        print(f"{i+1}. {ex.replace('_', ' ').title()}")
    
    return exercises

# Webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Available exercises (press number keys to switch):")
print("1. Pushup  2. Incline Press  3. Decline Fly  4. Chest Fly Machine")
print("5. Tricep Overhead  6. Tricep Cable Pulldown  7. Shoulder Press  8. Lateral Raise")
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_height, frame_width, _ = frame.shape

    # Recolor image to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Reset drawing specs
    drawing_specs = {}

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # --- Exercise-specific Logic Switch ---

        if current_exercise == "pushup":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_pushup(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "incline_press":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_incline_press(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "decline_fly":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_decline_fly(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "chest_fly_machine":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_chest_fly_machine(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "tricep_overhead":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_tricep_overhead(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "tricep_cable_pulldown":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_tricep_cable_pulldown(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "shoulder_press":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_shoulder_press(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "lateral_raise":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_lateral_raise(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        else:
            feedback_text = "No exercise selected."

        # --- Draw Visual Cues on the Body ---

        if drawing_specs:
            if current_exercise == "pushup":
                specs = drawing_specs
                # Elbow circle
                cv2.circle(image, specs["left_elbow_2d"], 10, specs["elbow_line_color"], -1)
                # Back lines
                cv2.line(image, specs["left_shoulder_2d"], specs["left_hip_2d"], specs["back_line_color"], 4)
                cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["back_line_color"], 4)
                # Hip circle
                cv2.circle(image, specs["left_hip_2d"], 10, specs["hip_circle_color"], -1)
                # Shoulder position indicator
                if "shoulder_color" in specs:
                    cv2.circle(image, specs["left_shoulder_2d"], 8, specs["shoulder_color"], -1)

            elif current_exercise in ["incline_press", "tricep_cable_pulldown"]:
                # Draw elbow indicators for pressing movements
                if "left_elbow_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["left_elbow_2d"], 10, drawing_specs["elbow_color"], -1)
                if "right_elbow_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["right_elbow_2d"], 10, drawing_specs["elbow_color"], -1)

            elif current_exercise == "shoulder_press":
                # Draw comprehensive arm visualization for shoulder press
                if "left_elbow_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["left_elbow_2d"], 10, drawing_specs["elbow_color"], -1)
                if "right_elbow_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["right_elbow_2d"], 10, drawing_specs["elbow_color"], -1)
                # Draw arm segments
                if "left_shoulder_2d" in drawing_specs and "left_elbow_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["left_shoulder_2d"], drawing_specs["left_elbow_2d"], drawing_specs["elbow_color"], 3)
                if "right_shoulder_2d" in drawing_specs and "right_elbow_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["right_shoulder_2d"], drawing_specs["right_elbow_2d"], drawing_specs["elbow_color"], 3)
                if "left_elbow_2d" in drawing_specs and "left_wrist_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["left_elbow_2d"], drawing_specs["left_wrist_2d"], drawing_specs["elbow_color"], 3)
                if "right_elbow_2d" in drawing_specs and "right_wrist_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["right_elbow_2d"], drawing_specs["right_wrist_2d"], drawing_specs["elbow_color"], 3)

            elif current_exercise in ["decline_fly", "chest_fly_machine"]:
                # Draw wrist indicators and arm lines for fly movements
                if "left_wrist_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["left_wrist_2d"], 10, drawing_specs["arm_color"], -1)
                if "right_wrist_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["right_wrist_2d"], 10, drawing_specs["arm_color"], -1)
                # Draw lines between shoulders and wrists
                if "left_shoulder_2d" in drawing_specs and "left_wrist_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["left_shoulder_2d"], drawing_specs["left_wrist_2d"], drawing_specs["arm_color"], 3)
                if "right_shoulder_2d" in drawing_specs and "right_wrist_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["right_shoulder_2d"], drawing_specs["right_wrist_2d"], drawing_specs["arm_color"], 3)

            elif current_exercise == "tricep_overhead":
                # Draw both arms for tricep extension
                if all(key in drawing_specs for key in ["left_shoulder_2d", "left_elbow_2d", "left_wrist_2d"]):
                    cv2.line(image, drawing_specs["left_shoulder_2d"], drawing_specs["left_elbow_2d"], drawing_specs["elbow_color"], 3)
                    cv2.line(image, drawing_specs["left_elbow_2d"], drawing_specs["left_wrist_2d"], drawing_specs["elbow_color"], 3)
                    cv2.circle(image, drawing_specs["left_elbow_2d"], 10, drawing_specs["elbow_color"], -1)
                if all(key in drawing_specs for key in ["right_shoulder_2d", "right_elbow_2d", "right_wrist_2d"]):
                    cv2.line(image, drawing_specs["right_shoulder_2d"], drawing_specs["right_elbow_2d"], drawing_specs["elbow_color"], 3)
                    cv2.line(image, drawing_specs["right_elbow_2d"], drawing_specs["right_wrist_2d"], drawing_specs["elbow_color"], 3)
                    cv2.circle(image, drawing_specs["right_elbow_2d"], 10, drawing_specs["elbow_color"], -1)

            elif current_exercise == "lateral_raise":
                # Draw comprehensive arm visualization for lateral raise
                if "left_wrist_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["left_wrist_2d"], 10, drawing_specs["arm_color"], -1)
                if "right_wrist_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["right_wrist_2d"], 10, drawing_specs["arm_color"], -1)
                if "left_elbow_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["left_elbow_2d"], 8, drawing_specs["arm_color"], -1)
                if "right_elbow_2d" in drawing_specs:
                    cv2.circle(image, drawing_specs["right_elbow_2d"], 8, drawing_specs["arm_color"], -1)
                # Draw arm segments
                if "left_shoulder_2d" in drawing_specs and "left_elbow_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["left_shoulder_2d"], drawing_specs["left_elbow_2d"], drawing_specs["arm_color"], 3)
                if "left_elbow_2d" in drawing_specs and "left_wrist_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["left_elbow_2d"], drawing_specs["left_wrist_2d"], drawing_specs["arm_color"], 3)
                if "right_shoulder_2d" in drawing_specs and "right_elbow_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["right_shoulder_2d"], drawing_specs["right_elbow_2d"], drawing_specs["arm_color"], 3)
                if "right_elbow_2d" in drawing_specs and "right_wrist_2d" in drawing_specs:
                    cv2.line(image, drawing_specs["right_elbow_2d"], drawing_specs["right_wrist_2d"], drawing_specs["arm_color"], 3)

        # --- Display Exercise Info and Controls ---
        
        # Create overlay for exercise info
        overlay = image.copy()
        alpha = 0.6

        # Exercise name and controls box
        cv2.rectangle(overlay, (0, 0), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(image, f'EXERCISE: {current_exercise.replace("_", " ").upper()}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'REPS: ' + str(rep_counter), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'STATE: ' + exercise_state.upper(), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'Keys: 1-8 to switch exercise', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Main Feedback Text (larger, center bottom)
        text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = frame_height - 30

        # Feedback text background
        cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(image, feedback_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    except Exception as e:
        # print(f"Error: {e}") # Uncomment for debugging
        cv2.putText(image, "Adjust camera or position", (50, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, BAD_COLOR, 2, cv2.LINE_AA)
        pass

    # Render pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=2, circle_radius=2))

    # Display the image
    cv2.imshow('Enhanced AI Gym Coach', image)

    # Handle keyboard input
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_exercise = "pushup"
        rep_counter = 0
        exercise_state = "up"
    elif key == ord('2'):
        current_exercise = "incline_press"
        rep_counter = 0
        exercise_state = "up"
    elif key == ord('3'):
        current_exercise = "decline_fly"
        rep_counter = 0
        exercise_state = "open"
    elif key == ord('4'):
        current_exercise = "chest_fly_machine"
        rep_counter = 0
        exercise_state = "open"
    elif key == ord('5'):
        current_exercise = "tricep_overhead"
        rep_counter = 0
        exercise_state = "extended"
    elif key == ord('6'):
        current_exercise = "tricep_cable_pulldown"
        rep_counter = 0
        exercise_state = "up"
    elif key == ord('7'):
        current_exercise = "shoulder_press"
        rep_counter = 0
        exercise_state = "up"
    elif key == ord('8'):
        current_exercise = "lateral_raise"
        rep_counter = 0
        exercise_state = "down"

cap.release()
cv2.destroyAllWindows()
pose.close()