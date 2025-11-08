import cv2
import numpy as np
import pyttsx3
import time
from threading import Thread
from ultralytics import YOLO
from collections import deque

# Initialize YOLOv8-Pose model
print("Loading YOLOv8-Pose model...")
model = YOLO('yolov8n-pose.pt')  # Will auto-download on first run
print("Model loaded successfully!")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Voice feedback state management
last_announcement_time = 0
last_announcement_text = ""
ANNOUNCEMENT_COOLDOWN = 3.0  # Seconds between announcements

# Smoothing buffers for angles (reduces jitter)
elbow_angle_buffer = deque(maxlen=5)
back_angle_buffer = deque(maxlen=5)


def speak_async(text):
    """Speak text in a separate thread to avoid blocking the main loop."""
    def speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    
    thread = Thread(target=speak)
    thread.daemon = True
    thread.start()


def announce_feedback(feedback_text, force=False):
    """
    Announces feedback if the text has changed OR enough time has passed.
    force=True will announce immediately regardless of cooldown (for important events).
    """
    global last_announcement_time, last_announcement_text
    
    current_time = time.time()
    time_since_last = current_time - last_announcement_time
    
    # Announce if text changed (immediate) OR cooldown passed OR forced
    if force or feedback_text != last_announcement_text or time_since_last >= ANNOUNCEMENT_COOLDOWN:
        speak_async(feedback_text)
        last_announcement_time = current_time
        last_announcement_text = feedback_text


def calculate_angle(a, b, c):
    """
    Calculates the angle between three 2D points.
    a, b, c: Tuples or lists of (x, y) coordinates.
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


def smooth_angle(angle, buffer):
    """Smooth angle using moving average to reduce jitter."""
    buffer.append(angle)
    return np.mean(buffer)


def get_keypoint(keypoints, index):
    """
    Extract x, y coordinates from YOLOv8 keypoints.
    Returns (x, y) or None if confidence is too low.
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    
    kpt = keypoints[0]  # First person detected
    if index >= len(kpt.data[0]):
        return None
    
    x, y, conf = kpt.data[0][index]
    
    # Return None if confidence is too low
    if conf < 0.5:
        return None
    
    return (int(x), int(y))


# --- Exercise Processing Functions ---

def process_pushup(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Processes the logic for a pushup using YOLOv8 keypoints.
    
    YOLOv8-Pose Keypoint indices (COCO format):
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
    5: Left Shoulder, 6: Right Shoulder
    7: Left Elbow, 8: Right Elbow
    9: Left Wrist, 10: Right Wrist
    11: Left Hip, 12: Right Hip
    13: Left Knee, 14: Right Knee
    15: Left Ankle, 16: Right Ankle
    """
    
    # Initialize drawing specs
    drawing_specs = {}
    
    # Get keypoints (using left side for consistency)
    left_shoulder = get_keypoint(keypoints, 5)
    left_elbow = get_keypoint(keypoints, 7)
    left_wrist = get_keypoint(keypoints, 9)
    left_hip = get_keypoint(keypoints, 11)
    left_knee = get_keypoint(keypoints, 13)
    
    # Check if all required keypoints are detected
    if None in [left_shoulder, left_elbow, left_wrist, left_hip, left_knee]:
        feedback_text = "Adjust camera - can't see full body"
        return rep_counter, exercise_state, feedback_text, {}
    
    # Calculate angles with smoothing
    elbow_angle_raw = calculate_angle(left_shoulder, left_elbow, left_wrist)
    back_angle_raw = calculate_angle(left_shoulder, left_hip, left_knee)
    
    elbow_angle = smooth_angle(elbow_angle_raw, elbow_angle_buffer)
    back_angle = smooth_angle(back_angle_raw, back_angle_buffer)
    
    # --- Form Correction Cues & UI Coloring ---
    elbow_line_color = GOOD_COLOR
    back_line_color = GOOD_COLOR
    hip_circle_color = GOOD_COLOR
    
    # Back straightness check
    if back_angle < 160:  # Threshold for straight back
        feedback_text = "Keep your back straight!"
        back_line_color = BAD_COLOR
        hip_circle_color = BAD_COLOR
        announce_feedback(feedback_text)
    else:
        feedback_text = "Good back form!"
        back_line_color = GOOD_COLOR
        hip_circle_color = GOOD_COLOR
        announce_feedback(feedback_text)
    
    # Elbow depth (for rep counting)
    if elbow_angle < 90 and back_angle > 160:  # Deep enough and back is straight
        exercise_state = "down"
        elbow_line_color = GOOD_COLOR
        feedback_text = "Lower!"
        announce_feedback(feedback_text)
    
    elif elbow_angle > 160 and exercise_state == "down":  # Back up, rep complete
        exercise_state = "up"
        rep_counter += 1
        feedback_text = "Rep Complete!"
        elbow_line_color = GOOD_COLOR
        announce_feedback(feedback_text)
    
    elif elbow_angle > 160 and exercise_state == "up":  # Staying up, ready for next rep
        feedback_text = "Ready to lower!"
        elbow_line_color = GOOD_COLOR
        announce_feedback(feedback_text)
    else:
        elbow_line_color = BAD_COLOR
        if "back" not in feedback_text:
            feedback_text = "Push up or lower!"
            announce_feedback(feedback_text)
    
    # Populate drawing_specs
    drawing_specs = {
        "elbow_line_color": elbow_line_color,
        "back_line_color": back_line_color,
        "hip_circle_color": hip_circle_color,
        "left_elbow": left_elbow,
        "left_shoulder": left_shoulder,
        "left_hip": left_hip,
        "left_knee": left_knee,
        "left_wrist": left_wrist,
        "elbow_angle": int(elbow_angle),
        "back_angle": int(back_angle)
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


# --- Dummy Functions for Other Exercises ---

def process_barbell_squat(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Barbell Squat logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_deadlift(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Deadlift logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_chest_press(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Chest Press logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_shoulder_press(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Shoulder Press logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_pull_up(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    feedback_text = "Pull Up logic not implemented."
    return rep_counter, exercise_state, feedback_text, {}


def process_bicep_curl(keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text):
    """
    Simple bicep curl test - just curl your arm to test the system!
    Works great for testing without doing full push-ups.
    """
    
    # Initialize drawing specs
    drawing_specs = {}
    
    # Get keypoints for right arm (easier to test)
    right_shoulder = get_keypoint(keypoints, 6)
    right_elbow = get_keypoint(keypoints, 8)
    right_wrist = get_keypoint(keypoints, 10)
    
    # Check if all required keypoints are detected
    if None in [right_shoulder, right_elbow, right_wrist]:
        feedback_text = "Can't see your arm - adjust camera"
        return rep_counter, exercise_state, feedback_text, {}
    
    # Calculate elbow angle with smoothing
    elbow_angle_raw = calculate_angle(right_shoulder, right_elbow, right_wrist)
    elbow_angle = smooth_angle(elbow_angle_raw, elbow_angle_buffer)
    
    # --- Form checking and rep counting ---
    elbow_line_color = GOOD_COLOR
    
    # Curled position (arm bent)
    if elbow_angle < 50:  # Fully curled
        exercise_state = "curled"
        elbow_line_color = GOOD_COLOR
        feedback_text = "Good curl!"
        announce_feedback(feedback_text)
    
    # Extended position (arm straight) - completes the rep
    elif elbow_angle > 140 and exercise_state == "curled":
        exercise_state = "extended"
        rep_counter += 1
        feedback_text = "Rep complete!"
        elbow_line_color = GOOD_COLOR
        announce_feedback(feedback_text)
    
    # Ready position
    elif elbow_angle > 140 and exercise_state == "extended":
        feedback_text = "Curl your arm!"
        elbow_line_color = GOOD_COLOR
        announce_feedback(feedback_text)
    
    # In between
    else:
        if elbow_angle < 140 and elbow_angle > 50:
            feedback_text = "Keep going..."
            elbow_line_color = (255, 165, 0)  # Orange
    
    # Populate drawing specs
    drawing_specs = {
        "elbow_line_color": elbow_line_color,
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
        "elbow_angle": int(elbow_angle)
    }
    
    return rep_counter, exercise_state, feedback_text, drawing_specs


# --- Main Application Logic ---

# Global state variables
rep_counter = 0
exercise_state = "up"  # Can be "up" or "down"
feedback_text = ""
current_exercise = "bicep_curl"  # Default exercise for testing
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

print("AI Gym Coach Started with YOLOv8-Pose!")
print("Press 'q' to quit")
print(f"Voice announcements will occur every {ANNOUNCEMENT_COOLDOWN} seconds")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    frame_height, frame_width, _ = frame.shape
    
    # Run YOLOv8-Pose detection
    results = model(frame, verbose=False)  # verbose=False to reduce console output
    
    # Get the annotated frame with skeleton
    annotated_frame = results[0].plot()  # YOLOv8 draws the skeleton automatically
    
    # Reset drawing specs
    drawing_specs = {}
    
    # Extract keypoints
    try:
        keypoints = results[0].keypoints
        
        if keypoints is not None and len(keypoints.data) > 0:
            # --- Exercise-specific Logic Switch ---
            
            if current_exercise == "pushup":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_pushup(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            elif current_exercise == "barbell_squat":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_barbell_squat(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            elif current_exercise == "deadlift":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_deadlift(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            elif current_exercise == "chest_press":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_chest_press(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            elif current_exercise == "shoulder_press":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_shoulder_press(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            elif current_exercise == "pull_up":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_pull_up(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            elif current_exercise == "bicep_curl":
                rep_counter, exercise_state, feedback_text, drawing_specs = process_bicep_curl(
                    keypoints, frame_width, frame_height, rep_counter, exercise_state, feedback_text
                )
            else:
                feedback_text = "No exercise selected."
            
            # --- Draw Custom Visual Cues on the Body ---
            
            if current_exercise == "bicep_curl" and drawing_specs:
                specs = drawing_specs
                
                # Draw arm lines
                cv2.line(annotated_frame, specs["right_shoulder"], specs["right_elbow"], 
                        specs["elbow_line_color"], 4)
                cv2.line(annotated_frame, specs["right_elbow"], specs["right_wrist"], 
                        specs["elbow_line_color"], 4)
                
                # Draw elbow circle
                cv2.circle(annotated_frame, specs["right_elbow"], 12, specs["elbow_line_color"], -1)
                
                # Display angle
                cv2.putText(annotated_frame, f'Angle: {specs["elbow_angle"]}', 
                           (specs["right_elbow"][0] + 20, specs["right_elbow"][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
            
            if current_exercise == "pushup" and drawing_specs:
                specs = drawing_specs
                
                # Draw elbow circle
                cv2.circle(annotated_frame, specs["left_elbow"], 10, specs["elbow_line_color"], -1)
                
                # Draw back lines
                cv2.line(annotated_frame, specs["left_shoulder"], specs["left_hip"], 
                        specs["back_line_color"], 4)
                cv2.line(annotated_frame, specs["left_hip"], specs["left_knee"], 
                        specs["back_line_color"], 4)
                
                # Draw hip circle
                cv2.circle(annotated_frame, specs["left_hip"], 10, specs["hip_circle_color"], -1)
                
                # Highlight bad back with larger circle
                if specs["back_line_color"] == BAD_COLOR:
                    cv2.circle(annotated_frame, specs["left_hip"], 15, BAD_COLOR, -1)
                
                # Display angles for debugging (optional)
                cv2.putText(annotated_frame, f'Elbow: {specs["elbow_angle"]}', 
                           (specs["left_elbow"][0] + 20, specs["left_elbow"][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f'Back: {specs["back_angle"]}', 
                           (specs["left_hip"][0] + 20, specs["left_hip"][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        
        else:
            feedback_text = "No person detected - step into view"
    
    except Exception as e:
        # print(f"Error: {e}")  # Uncomment for debugging
        feedback_text = "Adjust camera or position"
    
    # --- Display Reps and General Feedback (GUI) ---
    
    # Create semi-transparent overlay
    overlay = annotated_frame.copy()
    alpha = 0.6
    
    # Reps and State box
    cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    cv2.putText(annotated_frame, 'REPS: ' + str(rep_counter), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, 'STATE: ' + exercise_state.upper(), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
    
    # Main Feedback Text (center bottom)
    text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height - 30
    
    # Feedback text background
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    cv2.putText(annotated_frame, feedback_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('AI Gym Coach - YOLOv8', annotated_frame)
    
    # Exit logic
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()