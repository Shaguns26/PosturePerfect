import cv2
import mediapipe as mp
import numpy as np

# --- UPDATED IMPORTS ---
# Import exercise processing functions from the 'exercise_logic' subdirectory
from exercise_logic.pushup import process_pushup
from exercise_logic.barbell_squat import process_barbell_squat
from exercise_logic.free_squat import process_air_squat
from exercise_logic.deadlift import process_deadlift
from exercise_logic.chest_press import process_chest_press
from exercise_logic.shoulder_press import process_shoulder_press
from exercise_logic.pullup import process_pull_up

# Import shared utilities
from utils import mp_pose, GOOD_COLOR, BAD_COLOR, TEXT_COLOR

# --- END OF UPDATED IMPORTS ---

# Initialize MediaPipe Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Main Application Logic ---

# Global state variables
rep_counter = 0
exercise_state = "up"  # Can be "up" or "down"
feedback_text = ""
current_exercise = "deadlift"  # <<< You can change this to test different exercises

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

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # --- Exercise-specific Logic Switch ---
        # Note the NEW function signatures:
        # 1. 'image' is passed in.
        # 2. 'drawing_specs' is no longer returned.

        if current_exercise == "pushup":
            rep_counter, exercise_state, feedback_text = process_pushup(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "barbell_squat":
            rep_counter, exercise_state, feedback_text = process_barbell_squat(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "air_squat":
            rep_counter, exercise_state, feedback_text = process_air_squat(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "deadlift":
            rep_counter, exercise_state, feedback_text = process_deadlift(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "chest_press":
            rep_counter, exercise_state, feedback_text = process_chest_press(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "shoulder_press":
            rep_counter, exercise_state, feedback_text = process_shoulder_press(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "pull_up":
            rep_counter, exercise_state, feedback_text = process_pull_up(
                image, landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        else:
            feedback_text = "No exercise selected."

        # --- Draw Visual Cues on the Body ---
        #
        # ALL of the exercise-specific drawing 'if/elif' blocks are GONE from this file.
        # That logic now lives inside each 'process_...' function.
        #

        # --- Display Reps and General Feedback (GUI) ---
        # This part STAYS because it's generic, not exercise-specific.

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