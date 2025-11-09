import cv2
from ultralytics import YOLO
import json
import time
from datetime import datetime
from collections import defaultdict

# --- UPDATED IMPORTS ---
from exercise_logic.pushup import process_pushup
from exercise_logic.barbell_squat import process_barbell_squat
from exercise_logic.free_squat import process_air_squat
from exercise_logic.deadlift import process_deadlift
from exercise_logic.chest_press import process_chest_press
from exercise_logic.shoulder_press import process_shoulder_press
from exercise_logic.pullup import process_pull_up
from exercise_logic.donkey_calf_raise import process_donkey_calf_raise
from exercise_logic.lunge import process_lunge
from exercise_logic.jump_squat import process_jump_squat
from exercise_logic.bulgarian_split_squat import process_bulgarian_split_squat
from exercise_logic.crunches import process_crunches
from exercise_logic.laying_leg_raises import process_laying_leg_raises
from exercise_logic.russian_twists import process_russian_twist
from exercise_logic.side_plank_up_down import process_side_plank_up_down
from exercise_logic.elbow_side_plank import process_elbow_side_plank
from exercise_logic.pike_press import process_pike_press
from exercise_logic.overhead_squat import process_overhead_squat
from exercise_logic.chin_ups import process_chin_ups
from exercise_logic.glute_bridge import process_glute_bridge
from exercise_logic.kickbacks import process_kickbacks
from exercise_logic.single_leg_rdl import process_single_leg_rdl
from exercise_logic.good_mornings import process_good_mornings
from exercise_logic.plank import process_plank, PLANK_STOPPED, format_duration  # Import format_duration

# Import shared utilities
from utils import GOOD_COLOR, BAD_COLOR, TEXT_COLOR, draw_yolo_skeleton, YOLO_KEYPOINT_MAP

# --- Initialize YOLO Pose Model ---
try:
    yolo_model = YOLO("yolov8n-pose.pt")
    print("✅ YOLOv8 Pose Model Loaded.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}. Ensure 'ultralytics' is installed and 'yolov8n-pose.pt' is accessible.")
    yolo_model = None

# --- GLOBAL TTS State (Simulated) ---
last_speech_time = time.time()
SPEECH_COOLDOWN = 2.0
API_KEY = ""
TTS_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={API_KEY}"


def speak_feedback(text):
    """Simulates Text-to-Speech."""
    global last_speech_time
    if time.time() - last_speech_time > SPEECH_COOLDOWN and text:
        print(f"🔊 TTS Triggered: '{text}'")
        last_speech_time = time.time()


class WorkoutAnalyzer:
    """Tracks workout metrics for analysis. Duration tracking added for time-based exercises."""

    def __init__(self):
        self.reset()
        self.total_duration_held = 0.0

    def reset(self):
        self.total_reps = 0
        self.good_reps = 0
        self.form_issues = defaultdict(int)
        self.feedback_history = []
        self.frame_count = 0
        self.good_form_frames = 0
        self.bad_form_frames = 0
        self.depth_issues = 0
        self.back_issues = 0
        self.elbow_issues = 0
        self.rep_timestamps = []
        self.total_duration_held = 0.0

    def log_frame(self, feedback_text, has_good_form=True):
        """Log each frame's feedback"""
        self.frame_count += 1
        self.feedback_history.append(feedback_text)

        if has_good_form:
            self.good_form_frames += 1
        else:
            self.bad_form_frames += 1

        if "back" in feedback_text.lower() and ("straight" in feedback_text.lower() or "flat" in feedback_text.lower()):
            self.back_issues += 1
            self.form_issues["Back not straight"] += 1
        if "hips up" in feedback_text.lower() or "hips down" in feedback_text.lower():
            self.form_issues["Hip Alignment Issue"] += 1
        if "depth" in feedback_text.lower() or "parallel" in feedback_text.lower():
            self.depth_issues += 1
            self.form_issues["Insufficient depth"] += 1
        if "elbow" in feedback_text.lower() or "tuck" in feedback_text.lower():
            self.elbow_issues += 1
            self.form_issues["Elbow positioning"] += 1
        if "lean" in feedback_text.lower():
            self.form_issues["Leaning back"] += 1
        if "squat" in feedback_text.lower() and "don't" in feedback_text.lower():
            self.form_issues["Squatting instead of hinging"] += 1

    def log_rep(self, is_good_form=True):
        """Log a completed rep (only used for rep-based exercises)"""
        self.total_reps += 1
        self.rep_timestamps.append(self.frame_count)
        if is_good_form:
            self.good_reps += 1

    def log_duration(self, duration_in_seconds):
        """Logs the total time held for time-based exercises."""
        # Update the stored duration with the highest recorded total time
        self.total_duration_held = max(self.total_duration_held, duration_in_seconds)

    def get_analysis_summary(self, exercise_name):
        """Generate comprehensive analysis summary"""
        if self.frame_count == 0:
            return None

        form_score = int((self.good_form_frames / self.frame_count) * 100)
        sorted_issues = sorted(
            self.form_issues.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommendations = []
        if self.back_issues > self.frame_count * 0.1:
            recommendations.append("Focus on keeping chest up and maintaining neutral spine")
        if self.depth_issues > self.frame_count * 0.1:
            recommendations.append("Work on mobility to achieve proper depth")
        if self.elbow_issues > self.frame_count * 0.1:
            recommendations.append("Practice keeping elbows tucked to protect shoulders")
        if self.good_reps < self.total_reps * 0.7:
            recommendations.append("Reduce weight and focus on perfect form")
        if not recommendations:
            recommendations = [
                "Great form! Keep maintaining this quality",
                "Consider progressive overload for continued gains"
            ]

        summary = {
            "exercise": exercise_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_reps": self.total_reps,
            "good_reps": self.good_reps,
            "form_score": form_score,
            "total_frames": self.frame_count,
            "good_form_frames": self.good_form_frames,
            "bad_form_frames": self.bad_form_frames,
            "form_issues": [
                {"issue": issue, "count": count, "severity": self._get_severity(count)}
                for issue, count in sorted(self.form_issues.items(), key=lambda x: x[1], reverse=True)
            ],
            "recommendations": recommendations,
            "rep_quality": f"{self.good_reps}/{self.total_reps}"
        }

        if exercise_name in ["plank"]:
            summary["total_time_held"] = format_duration(self.total_duration_held)  # Use plank.format_duration
            summary["bad_form_frames"] = self.bad_form_frames
        else:
            summary["total_reps"] = self.total_reps
            summary["good_reps"] = self.good_reps
            summary["rep_quality"] = f"{self.good_reps}/{self.total_reps}"
            summary["bad_form_frames"] = self.bad_form_frames

        return summary

    def _get_recommendations(self, exercise_name):
        recommendations = []
        if self.back_issues > self.frame_count * 0.1:
            recommendations.append("Focus on keeping chest up and maintaining neutral spine")
        if self.good_reps < self.total_reps * 0.7 and exercise_name not in ["plank"]:
            recommendations.append("Reduce weight and focus on perfect form")
        if exercise_name == "plank" and self.bad_form_frames > self.frame_count * 0.1:
            recommendations.append("Practice holding for shorter intervals with perfect hip alignment.")

        if not recommendations:
            recommendations = ["Great form! Keep maintaining this quality"]

        return recommendations

    def _get_severity(self, count):
        if count > self.frame_count * 0.2:
            return "high"
        elif count > self.frame_count * 0.1:
            return "medium"
        else:
            return "low"

    def save_analysis(self, filename="workout_analysis.json"):
        summary = self.get_analysis_summary("workout")
        if summary:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✓ Analysis saved to {filename}")


def display_analysis_summary(summary):
    """Display analysis summary in terminal"""
    print("\n" + "=" * 60)
    print("WORKOUT ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nExercise: {summary['exercise']}")
    print(f"Date: {summary['timestamp']}")

    if summary['exercise'] == "plank":
        print(f"Total Time Held: {summary['total_time_held']}")
    else:
        print(f"Reps: {summary['total_reps']} (Good form: {summary['good_reps']})")
        print(f"Rep Quality: {summary['rep_quality']}")

    print(f"Form Score: {summary['form_score']}%")

    print("\n--- FORM ISSUES ---")
    if summary['form_issues']:
        for issue in summary['form_issues']:
            severity_symbol = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🔵"
            print(f"{severity_symbol} {issue['issue']}: {issue['count']} times ({issue['severity'].upper()})")
    else:
        print("✓ No major form issues detected!")

    print("\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"{i}. {rec}")

    print("\n" + "=" * 60 + "\n")


def run_live_mode(exercise_name):
    """Run live webcam mode with real-time feedback using YOLO."""
    global yolo_model
    if not yolo_model:
        print("Cannot run live mode: YOLO model failed to load.")
        return

    print(f"\n🎥 Starting LIVE mode for {exercise_name.replace('_', ' ').title()} with YOLO")
    print("Press 'q' to quit\n")

    # For Plank: rep_or_duration = total accumulated time (float)
    # For Plank: plank_start_time = segment start timestamp (float) or PLANK_STOPPED (0.0)
    rep_or_duration = 0.0
    exercise_state = "up"  # Default string state for rep-based exercises
    feedback_text = ""
    analyzer = WorkoutAnalyzer()

    is_time_based = (exercise_name == "plank")
    plank_start_time = PLANK_STOPPED

    # Determine visibility requirements
    is_upper_body_exercise = (exercise_name == "shoulder_press")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    exercise_processor = get_exercise_processor(exercise_name)
    window_title = f'RepWise - Live Mode: {exercise_name.replace("_", " ").title()}'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_height, frame_width, _ = frame.shape
        image = frame.copy()

        # --- YOLO INFERENCE ---
        yolo_results = yolo_model(image, verbose=False)

        is_visible = False
        landmarks = None
        current_frame_feedback = "CENTER AND SHOW ENTIRE BODY"
        current_speech_text = ""

        if len(yolo_results[0].keypoints.data) > 0:
            person_keypoints = yolo_results[0].keypoints.data[0].cpu().numpy()

            # --- POSE VISIBILITY CHECK ---
            try:
                vis_nose = person_keypoints[YOLO_KEYPOINT_MAP["NOSE"]][2]

                if is_upper_body_exercise:
                    # Shoulder Press needs only torso and arms
                    vis_l_shoulder = person_keypoints[YOLO_KEYPOINT_MAP["LEFT_SHOULDER"]][2]
                    vis_r_shoulder = person_keypoints[YOLO_KEYPOINT_MAP["RIGHT_SHOULDER"]][2]
                    vis_l_wrist = person_keypoints[YOLO_KEYPOINT_MAP["LEFT_WRIST"]][2]
                    vis_r_wrist = person_keypoints[YOLO_KEYPOINT_MAP["RIGHT_WRIST"]][2]

                    if vis_nose > 0.5 and vis_l_shoulder > 0.5 and vis_r_shoulder > 0.5 and vis_l_wrist > 0.5 and vis_r_wrist > 0.5:
                        is_visible = True
                    # Update feedback for upper body
                    current_frame_feedback = "CENTER TORSO AND ARMS"
                else:
                    # Full body exercises need anchors (ankles)
                    vis_l_ankle = person_keypoints[YOLO_KEYPOINT_MAP["LEFT_ANKLE"]][2]
                    vis_r_ankle = person_keypoints[YOLO_KEYPOINT_MAP["RIGHT_ANKLE"]][2]

                    if vis_nose > 0.5 and vis_l_ankle > 0.5 and vis_r_ankle > 0.5:
                        is_visible = True
                    # Default feedback remains: CENTER AND SHOW ENTIRE BODY

                if is_visible:
                    landmarks = person_keypoints
            except IndexError:
                is_visible = False

        if landmarks is not None and is_visible:
            # --- PROCESS EXERCISE LOGIC ---
            try:
                prev_reps_or_duration = rep_or_duration

                # --- PLANK LOGIC (Pause/Resume) ---
                if is_time_based:

                    # Pass the accumulated duration (rep_or_duration) and segment start time
                    new_base_duration, plank_start_time, feedback_text, speech_text = exercise_processor(
                        image, landmarks, frame_width, frame_height,
                        rep_or_duration, plank_start_time, feedback_text
                    )

                    # Update the main state accumulator
                    rep_or_duration = new_base_duration

                    # Calculate live reported time for UI
                    if plank_start_time != PLANK_STOPPED:
                        # Running: Add current segment time to the base duration
                        current_time_for_ui = rep_or_duration + (time.time() - plank_start_time)
                    else:
                        # Paused: The total is fixed (already stored in rep_or_duration)
                        current_time_for_ui = rep_or_duration

                    # Set rep_or_duration to the current UI time for consistent logging/display
                    rep_or_duration = current_time_for_ui

                    # NOTE: For plank, the state passed to UI is plank_start_time (float/PLANK_STOPPED)


                # --- REP-BASED LOGIC ---
                else:
                    processor_results = exercise_processor(
                        image, landmarks, frame_width, frame_height,
                        int(rep_or_duration), exercise_state, feedback_text
                    )
                    if len(processor_results) == 4:
                        rep_or_duration, exercise_state, feedback_text, speech_text = processor_results
                    else:
                        rep_or_duration, exercise_state, feedback_text = processor_results
                        speech_text = ""
                    rep_or_duration = float(rep_or_duration)

                    # NOTE: For rep-based, the state passed to UI is exercise_state (string)

                current_frame_feedback = feedback_text
                current_speech_text = speech_text

                # --- LOGGING ---
                if not is_time_based and rep_or_duration > prev_reps_or_duration:
                    # Log Rep for rep-based exercises
                    has_good_form = "good" in feedback_text.lower() or "complete" in feedback_text.lower()
                    analyzer.log_rep(has_good_form)

                # Log frame
                has_good_form = "good" in feedback_text.lower() or "perfect" in feedback_text.lower() or "holding" in feedback_text.lower() or "rep complete" in feedback_text.lower()
                analyzer.log_frame(feedback_text, has_good_form)


            except Exception as e:
                current_frame_feedback = "Error processing pose data."
                print(f"Error in frame processing: {e}")

                # Render skeleton using the custom YOLO drawing function
            draw_yolo_skeleton(image, landmarks)

        else:
            # If no pose detected or visibility is low, revert state
            if is_time_based:
                # POSE LOST - PAUSE the timer
                if plank_start_time != PLANK_STOPPED:
                    # PAUSE: Calculate time held in the last segment and add to accumulator
                    segment_time = time.time() - plank_start_time
                    rep_or_duration += segment_time
                plank_start_time = PLANK_STOPPED  # Timer paused
                current_frame_feedback = "POSE LOST: Find your plank position to resume timer."
            else:
                exercise_state = "up"

            # Draw a box over the screen to emphasize the no-tracking state
            cv2.rectangle(image, (0, 0), (frame_width, frame_height), BAD_COLOR, 10)
            cv2.putText(image, current_frame_feedback, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BAD_COLOR, 2, cv2.LINE_AA)

        # --- Common UI and TTS ---
        speak_feedback(current_speech_text)

        # Update analyzer duration log for time-based exercise (always logs the current total)
        if is_time_based:
            analyzer.log_duration(rep_or_duration)

        # --- CRITICAL FIX HERE ---
        # If it's time-based (Plank), we pass plank_start_time as the state (float).
        # If it's rep-based, we pass the *string* exercise_state.
        ui_state_arg = plank_start_time if is_time_based else exercise_state

        display_live_ui(image, rep_or_duration, ui_state_arg, current_frame_feedback, frame_width, frame_height,
                        exercise_name)

        cv2.imshow(window_title, image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    summary = analyzer.get_analysis_summary(exercise_name)
    if summary:
        display_analysis_summary(summary)
        analyzer.save_analysis(f"{exercise_name}_live_analysis.json")


def analyze_recorded_video(video_path, exercise_name):
    """Analyze a recorded video and provide comprehensive summary."""
    global yolo_model
    if not yolo_model:
        print("Cannot analyze video: YOLO model failed to load.")
        return

    print(f"\n📹 Analyzing recorded video: {video_path}")
    print(f"Exercise: {exercise_name}\n")

    # In recorded video mode, we don't need pause/resume logic, just linear accumulation
    rep_or_duration = 0.0
    feedback_text = ""
    analyzer = WorkoutAnalyzer()

    is_time_based = (exercise_name == "plank")
    is_upper_body_exercise = (exercise_name == "shoulder_press")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_time_step = 1.0 / fps

    print(f"Video info: {total_frames} frames, {fps} FPS")
    print("Processing...\n")

    exercise_processor = get_exercise_processor(exercise_name)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames ({int(frame_num / total_frames * 100)}%)")

        frame_height, frame_width, _ = frame.shape
        image = frame.copy()

        # Process with YOLO
        yolo_results = yolo_model(image, verbose=False)
        landmarks = None

        if len(yolo_results[0].keypoints.data) > 0:
            try:
                landmarks = yolo_results[0].keypoints.data[0].cpu().numpy()
            except IndexError:
                pass

        if landmarks is not None:
            try:
                prev_reps_or_duration = rep_or_duration

                # For recorded video, we call the processor only for feedback/angles (not timing)
                # We use fixed PLANK_STOPPED and 0.0 for time states to force form check logic
                # We ignore the returned duration/start_time for recording analysis.
                _, _, feedback_text, _ = exercise_processor(
                    image, landmarks, frame_width, frame_height,
                    0.0, PLANK_STOPPED, feedback_text
                )

                # Accumulate time only if form is good
                if is_time_based:
                    # Check for "Perfect form" or "HOLDING" to indicate good form
                    if "perfect form" in feedback_text.lower() or "holding" in feedback_text.lower():
                        rep_or_duration += frame_time_step

                    analyzer.log_duration(rep_or_duration)
                else:
                    # REP-BASED (Normal logic)
                    processor_results = exercise_processor(
                        image, landmarks, frame_width, frame_height,
                        int(rep_or_duration), "down", feedback_text  # Use fixed state for analysis
                    )
                    if len(processor_results) == 4:
                        rep_or_duration, _, feedback_text, _ = processor_results
                    else:
                        rep_or_duration, _, feedback_text = processor_results
                    rep_or_duration = float(rep_or_duration)

                    if rep_or_duration > prev_reps_or_duration:
                        has_good_form = "good" in feedback_text.lower() or "complete" in feedback_text.lower()
                        analyzer.log_rep(has_good_form)

                # Log frame
                has_good_form = "good" in feedback_text.lower() or "perfect" in feedback_text.lower() or "holding" in feedback_text.lower()
                analyzer.log_frame(feedback_text, has_good_form)

            except:
                pass

    cap.release()

    print("\n✓ Analysis complete!\n")
    summary = analyzer.get_analysis_summary(exercise_name)

    if summary:
        display_analysis_summary(summary)
        analyzer.save_analysis(f"{exercise_name}_recorded_analysis.json")
    else:
        print("⚠ No valid data collected. Check video quality and framing.")


def display_live_ui(image, rep_or_duration, exercise_state, feedback_text, frame_width, frame_height, exercise_name):
    """Display UI elements for live mode, handling both reps and duration."""
    overlay = image.copy()
    alpha = 0.6

    # 1. Centered Exercise Title (Top)
    title_text = exercise_name.replace("_", " ").upper()
    title_color = TEXT_COLOR
    title_scale = 1.2
    title_thickness = 2
    title_box_height = 50

    cv2.rectangle(overlay, (0, 0), (frame_width, title_box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)[0]
    title_x = (frame_width - title_size[0]) // 2
    title_y = 35

    cv2.putText(image, title_text, (title_x, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_color, title_thickness, cv2.LINE_AA)

    # 2. Reps/Duration and State box (Top Left - below the title box)
    box_start_y = title_box_height
    cv2.rectangle(overlay, (0, box_start_y), (280, box_start_y + 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    if exercise_name == "plank":
        # Display duration using the new millisecond format
        display_metric = f"TIME: {format_duration(rep_or_duration)}"
        # exercise_state is plank_start_time, so we check if it's running (not PLANK_STOPPED)
        display_state = "RUNNING" if exercise_state != PLANK_STOPPED else "PAUSED"

        cv2.putText(image, display_metric, (10, box_start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'STATE: ' + display_state, (10, box_start_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    else:
        # Rep-based display
        cv2.putText(image, 'REPS: ' + str(int(rep_or_duration)), (10, box_start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'STATE: ' + exercise_state.upper(), (10, box_start_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    # 3. Main Feedback Text (Centered Horizontally at Bottom)
    text_scale = 1.0
    text_thickness = 2
    text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height - 30

    cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.putText(image, feedback_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)


def get_exercise_processor(exercise_name):
    """Return the appropriate exercise processor function"""
    processors = {
        "pushup": process_pushup,
        "barbell_squat": process_barbell_squat,
        "air_squat": process_air_squat,
        "deadlift": process_deadlift,
        "chest_press": process_chest_press,
        "shoulder_press": process_shoulder_press,
        "pull_up": process_pull_up,
        "donkey_calf_raise": process_donkey_calf_raise,
        "forward_lunge": process_lunge,
        "jump_squat": process_jump_squat,
        "bulgarian_split_squat": process_bulgarian_split_squat,
        "crunches": process_crunches,
        "laying_leg_raises": process_laying_leg_raises,
        "russian_twist": process_russian_twist,
        "side_plank_up_down": process_side_plank_up_down,
        "elbow_side_plank": process_elbow_side_plank,
        "pike_press": process_pike_press,
        "overhead_squat": process_overhead_squat,
        "chin_ups": process_chin_ups,
        "glute_bridge": process_glute_bridge,
        "kickbacks": process_kickbacks,
        "single_leg_rdl": process_single_leg_rdl,
        "good_mornings": process_good_mornings,
        "plank": process_plank,
    }
    return processors.get(exercise_name, process_pushup)


def main():
    """Main application with mode selection"""
    print("\n" + "=" * 60)
    print("REPWISE - AI GYM COACH")
    print("=" * 60)

    exercises = {
        "1": "pushup", "2": "barbell_squat", "3": "air_squat", "4": "deadlift", "5": "chest_press",
        "6": "shoulder_press",
        "7": "pull_up", "8": "donkey_calf_raise", "9": "forward_lunge", "10": "jump_squat",
        "11": "bulgarian_split_squat",
        "12": "crunches", "13": "laying_leg_raises", "14": "russian_twist", "15": "side_plank_up_down",
        "16": "elbow_side_plank",
        "17": "pike_press", "18": "overhead_squat", "19": "chin_ups", "20": "glute_bridge", "21": "kickbacks",
        "22": "single_leg_rdl", "23": "good_mornings",
        "24": "plank",
    }

    print("\nSelect Exercise:")
    print("1. Push-up\n2. Barbell Squat\n3. Air Squat\n4. Deadlift\n5. Chest Press\n6. Shoulder Press\n7. Pull-up")
    print(
        "8. Bodyweight Donkey Calf Raise\n9. Forward Lunges\n10. Jump Squats\n11. Bulgarian Split Squat\n12. Crunches")
    print(
        "13. Laying Leg Raises\n14. Bodyweight Russian Twist\n15. Side Plank Up Down\n16. Elbow Side Plank\n17. Bodyweight Pike Press")
    print(
        "18. Pole Overhead Squat\n19. Chin Ups\n20. Glute Bridge\n21. Kickbacks\n22. Single Legged Romanian Deadlifts\n23. Good Mornings")
    print("24. Plank (Time-based)")

    exercise_choice = input("\nEnter choice (1-24): ").strip()
    selected_exercise = exercises.get(exercise_choice, "pushup")

    print("\nSelect Mode:")
    print("1. Live Webcam (Real-time feedback)")
    print("2. Analyze Recorded Video (Comprehensive summary)")

    mode_choice = input("\nEnter choice (1-2): ").strip()

    if mode_choice == "1":
        run_live_mode(selected_exercise)
    elif mode_choice == "2":
        video_path = input("\nEnter video file path: ").strip()
        analyze_recorded_video(video_path, selected_exercise)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()