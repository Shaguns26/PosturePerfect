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
    Processes the logic for a pushup.
    Calculates angles, provides feedback, counts reps, and returns drawing specs.
    """

    # Initialize drawing specs
    drawing_specs = {}

    # Get 3D coordinates for angle calculations
    left_shoulder_3d = get_landmark_3d(landmarks, "LEFT_SHOULDER")
    left_elbow_3d = get_landmark_3d(landmarks, "LEFT_ELBOW")
    left_wrist_3d = get_landmark_3d(landmarks, "LEFT_WRIST")
    left_hip_3d = get_landmark_3d(landmarks, "LEFT_HIP")
    left_knee_3d = get_landmark_3d(landmarks, "LEFT_KNEE")

    # Get 2D pixel coordinates for drawing
    left_shoulder_2d = get_landmark_coords(landmarks, "LEFT_SHOULDER", frame_width, frame_height)
    left_elbow_2d = get_landmark_coords(landmarks, "LEFT_ELBOW", frame_width, frame_height)
    left_wrist_2d = get_landmark_coords(landmarks, "LEFT_WRIST", frame_width, frame_height)
    left_hip_2d = get_landmark_coords(landmarks, "LEFT_HIP", frame_width, frame_height)
    left_knee_2d = get_landmark_coords(landmarks, "LEFT_KNEE", frame_width, frame_height)

    # Calculate angles
    elbow_angle = calculate_angle(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
    back_angle = calculate_angle(left_shoulder_3d, left_hip_3d, left_knee_3d)  # Simplified back angle

    # --- Form Correction Cues & UI Coloring ---
    elbow_line_color = GOOD_COLOR
    back_line_color = GOOD_COLOR
    hip_circle_color = GOOD_COLOR

    # Back straightness
    if back_angle < 160:  # Threshold for straight back
        feedback_text = "Keep your back straight!"
        back_line_color = BAD_COLOR
        hip_circle_color = BAD_COLOR
    else:
        feedback_text = "Good back form!"
        back_line_color = GOOD_COLOR
        hip_circle_color = GOOD_COLOR

    # Elbow depth (for rep counting)
    if elbow_angle < 90 and back_angle > 160:  # Deep enough and back is straight
        exercise_state = "down"
        elbow_line_color = GOOD_COLOR
        feedback_text = "Lower!"

    elif elbow_angle > 160 and exercise_state == "down":  # Back up, rep complete
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"
        elbow_line_color = GOOD_COLOR

    elif elbow_angle > 160 and exercise_state == "up":  # Staying up, ready for next rep
        feedback_text = "Ready to lower!"
        elbow_line_color = GOOD_COLOR
    else:
        elbow_line_color = BAD_COLOR  # Indicate elbows aren't fully locked or deep enough
        if "back" not in feedback_text:  # Don't overwrite critical back feedback
            feedback_text = "Push up or lower!"

    # Populate drawing_specs for the main loop to draw
    drawing_specs = {
        "elbow_line_color": elbow_line_color,
        "back_line_color": back_line_color,
        "hip_circle_color": hip_circle_color,
        "left_elbow_2d": left_elbow_2d,
        "left_shoulder_2d": left_shoulder_2d,
        "left_hip_2d": left_hip_2d,
        "left_knee_2d": left_knee_2d
    }

    return rep_counter, exercise_state, feedback_text, drawing_specs


# --- Dummy Functions for Other Exercises ---

def process_barbell_squat(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Barbell Squat logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_deadlift(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Deadlift logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_chest_press(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Chest Press logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_shoulder_press(landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Shoulder Press logic not implemented."
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

# Webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

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
        elif current_exercise == "barbell_squat":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_barbell_squat(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "deadlift":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_deadlift(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "chest_press":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_chest_press(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "shoulder_press":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_shoulder_press(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "pull_up":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_pull_up(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        else:
            feedback_text = "No exercise selected."

        # --- Draw Visual Cues on the Body ---

        # Draw cues only if the exercise processor provided them
        if current_exercise == "pushup" and drawing_specs:
            specs = drawing_specs  # for readability

            # Elbow circle
            cv2.circle(image, specs["left_elbow_2d"], 10, specs["elbow_line_color"], -1)

            # Back lines
            cv2.line(image, specs["left_shoulder_2d"], specs["left_hip_2d"], specs["back_line_color"], 4)
            cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["back_line_color"], 4)

            # Hip circle
            cv2.circle(image, specs["left_hip_2d"], 10, specs["hip_circle_color"], -1)

            # Highlight bad back
            if specs["back_line_color"] == BAD_COLOR:
                cv2.circle(image, specs["left_hip_2d"], 15, BAD_COLOR, -1)  # Larger red circle on hip

        # --- Display Reps and General Feedback (GUI) ---

        # Create a semi-transparent background for text
        overlay = image.copy()
        alpha = 0.6  # Transparency factor.

        # Reps and State box
        cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)  # Black box
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(image, 'REPS: ' + str(rep_counter), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'STATE: ' + exercise_state.upper(), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

        # Main Feedback Text (larger, center bottom)
        text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = frame_height - 30  # A bit above bottom

        # Feedback text background
        cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(image, feedback_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    except Exception as e:
        # print(f"Error: {e}") # Uncomment for debugging
        cv2.putText(image, "Adjust camera or position", (50, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, BAD_COLOR, 2, cv2.LINE_AA)
        pass  # Pass if no landmarks are detected

    # Render ALL detections (the full skeleton from MediaPipe, slightly dimmed)
    # This provides the base skeleton, and we draw our custom cues on top.
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=2),
                              # Dimmed color
                              mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=2, circle_radius=2)
                              )

    # Display the image
    cv2.imshow('AI Gym Coach', image)

    # Exit logic
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()