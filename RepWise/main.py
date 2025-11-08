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

# Import shared utilities (this file is in the same directory as the main script)
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
drawing_specs = {}  # Dictionary to hold drawing info

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
        # This now calls the functions imported from other files

        if current_exercise == "pushup":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_pushup(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "barbell_squat":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_barbell_squat(
                landmarks, frame_width, frame_height, rep_counter, exercise_state, feedback_text
            )
        elif current_exercise == "air_squat":
            rep_counter, exercise_state, feedback_text, drawing_specs = process_air_squat(
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

        # Draw cues for PUSHUP
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

            # Display angles
            cv2.putText(image, f'Elbow: {specs["elbow_angle"]}',
                        (specs["left_elbow_2d"][0] + 15, specs["left_elbow_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(image, f'Back: {specs["back_angle"]}', (specs["left_hip_2d"][0] + 15, specs["left_hip_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Draw cues for BARBELL SQUAT or AIR SQUAT
        elif (current_exercise == "barbell_squat" or current_exercise == "air_squat") and drawing_specs:
            specs = drawing_specs

            # Back line (Shoulder to Hip)
            cv2.line(image, specs["left_shoulder_2d"], specs["left_hip_2d"], specs["back_line_color"], 4)
            # Hip to Knee
            cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["back_line_color"], 4)

            # Knee line (Hip to Knee)
            cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["knee_line_color"], 4)
            # Knee to Ankle
            cv2.line(image, specs["left_knee_2d"], specs["left_ankle_2d"], specs["knee_line_color"], 4)

            # Draw circles on joints
            cv2.circle(image, specs["left_hip_2d"], 10, specs["back_line_color"], -1)
            cv2.circle(image, specs["left_knee_2d"], 10, specs["knee_line_color"], -1)

            # Highlight bad back
            if specs["back_line_color"] == BAD_COLOR:
                cv2.circle(image, specs["left_hip_2d"], 15, BAD_COLOR, -1)  # Larger red circle on hip

            # Display angles
            cv2.putText(image, f'Back: {specs["back_angle"]}', (specs["left_hip_2d"][0] + 15, specs["left_hip_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(image, f'Knee: {specs["knee_angle"]}',
                        (specs["left_knee_2d"][0] + 15, specs["left_knee_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Draw cues for DEADLIFT
        elif current_exercise == "deadlift" and drawing_specs:
            specs = drawing_specs

            # Back/Hinge line
            cv2.line(image, specs["left_shoulder_2d"], specs["left_hip_2d"], specs["hip_line_color"], 4)
            cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["hip_line_color"], 4)

            # Knee line
            cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["knee_line_color"], 4)
            cv2.line(image, specs["left_knee_2d"], specs["left_ankle_2d"], specs["knee_line_color"], 4)

            # Draw circles
            cv2.circle(image, specs["left_hip_2d"], 10, specs["hip_line_color"], -1)
            cv2.circle(image, specs["left_knee_2d"], 10, specs["knee_line_color"], -1)

            # Display angles
            cv2.putText(image, f'Hip: {specs["hip_angle"]}', (specs["left_hip_2d"][0] + 15, specs["left_hip_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(image, f'Knee: {specs["knee_angle"]}',
                        (specs["left_knee_2d"][0] + 15, specs["left_knee_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Draw cues for CHEST PRESS
        elif current_exercise == "chest_press" and drawing_specs:
            specs = drawing_specs

            # Arm line
            cv2.line(image, specs["left_shoulder_2d"], specs["left_elbow_2d"], specs["elbow_line_color"], 4)
            cv2.line(image, specs["left_elbow_2d"], specs["left_wrist_2d"], specs["elbow_line_color"], 4)

            # Shoulder line (for flare)
            cv2.line(image, specs["left_elbow_2d"], specs["left_shoulder_2d"], specs["shoulder_line_color"], 4)
            cv2.line(image, specs["left_shoulder_2d"], specs["left_hip_2d"], specs["shoulder_line_color"], 4)

            # Draw circles
            cv2.circle(image, specs["left_elbow_2d"], 10, specs["elbow_line_color"], -1)
            cv2.circle(image, specs["left_shoulder_2d"], 10, specs["shoulder_line_color"], -1)

            # Display angles
            cv2.putText(image, f'Elbow: {specs["elbow_angle"]}',
                        (specs["left_elbow_2d"][0] + 15, specs["left_elbow_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(image, f'Shoulder: {specs["shoulder_angle"]}',
                        (specs["left_shoulder_2d"][0] + 15, specs["left_shoulder_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Draw cues for SHOULDER PRESS
        elif current_exercise == "shoulder_press" and drawing_specs:
            specs = drawing_specs

            # Arm line
            cv2.line(image, specs["left_shoulder_2d"], specs["left_elbow_2d"], specs["arm_line_color"], 4)
            cv2.line(image, specs["left_elbow_2d"], specs["left_wrist_2d"], specs["arm_line_color"], 4)

            # Back line (for lean)
            cv2.line(image, specs["left_shoulder_2d"], specs["left_hip_2d"], specs["back_line_color"], 4)
            cv2.line(image, specs["left_hip_2d"], specs["left_knee_2d"], specs["back_line_color"],
                     4)

            # Draw circles
            cv2.circle(image, specs["left_elbow_2d"], 10, specs["arm_line_color"], -1)
            cv2.circle(image, specs["left_hip_2d"], 10, specs["back_line_color"], -1)

            # Display angles
            cv2.putText(image, f'Shoulder: {specs["shoulder_angle"]}',
                        (specs["left_shoulder_2d"][0] + 15, specs["left_shoulder_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(image, f'Back: {specs["back_angle"]}', (specs["left_hip_2d"][0] + 15, specs["left_hip_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Draw cues for PULL UP
        elif current_exercise == "pull_up" and drawing_specs:
            specs = drawing_specs

            # Arm line
            cv2.line(image, specs["left_shoulder_2d"], specs["left_elbow_2d"], specs["arm_line_color"], 4)
            cv2.line(image, specs["left_elbow_2d"], specs["left_wrist_2d"], specs["arm_line_color"], 4)

            # Draw circles
            cv2.circle(image, specs["left_elbow_2d"], 10, specs["arm_line_color"], -1)
            cv2.circle(image, specs["left_shoulder_2d"], 10, specs["arm_line_color"], -1)

            # Display angles
            cv2.putText(image, f'Elbow: {specs["elbow_angle"]}',
                        (specs["left_elbow_2d"][0] + 15, specs["left_elbow_2d"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

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