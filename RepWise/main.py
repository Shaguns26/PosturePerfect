import cv2
import mediapipe as mp
import numpy as np
import json
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
from exercise_logic.russian_twist import process_russian_twist
from exercise_logic.side_plank_up_down import process_side_plank_up_down
from exercise_logic.elbow_side_plank import process_elbow_side_plank
from exercise_logic.pike_press import process_pike_press
from exercise_logic.overhead_squat import process_overhead_squat
from exercise_logic.chin_ups import process_chin_ups
from exercise_logic.glute_bridge import process_glute_bridge
from exercise_logic.kickbacks import process_kickbacks
from exercise_logic.single_leg_rdl import process_single_leg_rdl
from exercise_logic.good_mornings import process_good_mornings
# Import shared utilities
from utils import mp_pose, GOOD_COLOR, BAD_COLOR, TEXT_COLOR

# --- Initialize MediaPipe Pose ---
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


class WorkoutAnalyzer:
    """Tracks workout metrics for analysis"""

    def __init__(self):
        self.reset()

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

    def log_frame(self, feedback_text, has_good_form=True):
        """Log each frame's feedback"""
        self.frame_count += 1
        self.feedback_history.append(feedback_text)

        if has_good_form:
            self.good_form_frames += 1
        else:
            self.bad_form_frames += 1

        # Track specific issues
        if "back" in feedback_text.lower() and "straight" in feedback_text.lower():
            self.back_issues += 1
            self.form_issues["Back not straight"] += 1
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
        """Log a completed rep"""
        self.total_reps += 1
        self.rep_timestamps.append(self.frame_count)
        if is_good_form:
            self.good_reps += 1

    def get_analysis_summary(self, exercise_name):
        """Generate comprehensive analysis summary"""
        if self.frame_count == 0:
            return None

        form_score = int((self.good_form_frames / self.frame_count) * 100)

        # Sort issues by frequency
        sorted_issues = sorted(
            self.form_issues.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Generate recommendations based on issues
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
                for issue, count in sorted_issues
            ],
            "recommendations": recommendations,
            "rep_quality": f"{self.good_reps}/{self.total_reps}"
        }

        return summary

    def _get_severity(self, count):
        """Determine severity based on frequency"""
        if count > self.frame_count * 0.2:
            return "high"
        elif count > self.frame_count * 0.1:
            return "medium"
        else:
            return "low"

    def save_analysis(self, filename="workout_analysis.json"):
        """Save analysis to JSON file"""
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
    print(f"\nReps: {summary['total_reps']} (Good form: {summary['good_reps']})")
    print(f"Form Score: {summary['form_score']}%")
    print(f"Rep Quality: {summary['rep_quality']}")

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
    """Run live webcam mode with real-time feedback"""
    print(f"\n🎥 Starting LIVE mode for {exercise_name}")
    print("Press 'q' to quit\n")

    rep_counter = 0
    exercise_state = "up"
    feedback_text = ""
    analyzer = WorkoutAnalyzer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get exercise processor
    exercise_processor = get_exercise_processor(exercise_name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_height, frame_width, _ = frame.shape

        # Process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            prev_reps = rep_counter

            # Process exercise-specific logic
            rep_counter, exercise_state, feedback_text = exercise_processor(
                image, landmarks, frame_width, frame_height,
                rep_counter, exercise_state, feedback_text
            )

            # Track if rep was completed
            if rep_counter > prev_reps:
                has_good_form = "good" in feedback_text.lower() or "complete" in feedback_text.lower()
                analyzer.log_rep(has_good_form)

            # Log frame
            has_good_form = "good" in feedback_text.lower()
            analyzer.log_frame(feedback_text, has_good_form)

            # Display UI
            display_live_ui(image, rep_counter, exercise_state, feedback_text, frame_width, frame_height)

        except Exception as e:
            cv2.putText(image, "Adjust camera or position", (50, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BAD_COLOR, 2, cv2.LINE_AA)

        # Render skeleton
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=2, circle_radius=2)
        )

        cv2.imshow('RepWise - Live Mode', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Show summary
    summary = analyzer.get_analysis_summary(exercise_name)
    if summary:
        display_analysis_summary(summary)
        analyzer.save_analysis(f"{exercise_name}_live_analysis.json")


def analyze_recorded_video(video_path, exercise_name):
    """Analyze a recorded video and provide comprehensive summary"""
    print(f"\n📹 Analyzing recorded video: {video_path}")
    print(f"Exercise: {exercise_name}\n")

    rep_counter = 0
    exercise_state = "up"
    feedback_text = ""
    analyzer = WorkoutAnalyzer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video info: {total_frames} frames, {fps} FPS")
    print("Processing...\n")

    # Get exercise processor
    exercise_processor = get_exercise_processor(exercise_name)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 30 == 0:  # Progress update every 30 frames
            print(f"Progress: {frame_num}/{total_frames} frames ({int(frame_num / total_frames * 100)}%)")

        frame_height, frame_width, _ = frame.shape

        # Process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            prev_reps = rep_counter

            # Process exercise-specific logic
            rep_counter, exercise_state, feedback_text = exercise_processor(
                image, landmarks, frame_width, frame_height,
                rep_counter, exercise_state, feedback_text
            )

            # Track if rep was completed
            if rep_counter > prev_reps:
                has_good_form = "good" in feedback_text.lower() or "complete" in feedback_text.lower()
                analyzer.log_rep(has_good_form)

            # Log frame
            has_good_form = "good" in feedback_text.lower()
            analyzer.log_frame(feedback_text, has_good_form)

        except:
            pass  # Skip frames where pose isn't detected

    cap.release()

    # Generate and display summary
    print("\n✓ Analysis complete!\n")
    summary = analyzer.get_analysis_summary(exercise_name)

    if summary:
        display_analysis_summary(summary)
        analyzer.save_analysis(f"{exercise_name}_recorded_analysis.json")
    else:
        print("⚠ No valid data collected. Check video quality and framing.")


def display_live_ui(image, rep_counter, exercise_state, feedback_text, frame_width, frame_height):
    """Display UI elements for live mode"""
    overlay = image.copy()
    alpha = 0.6

    # Reps and State box
    cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.putText(image, 'REPS: ' + str(rep_counter), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, 'STATE: ' + exercise_state.upper(), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    # Main Feedback Text
    text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height - 30

    cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.putText(image, feedback_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)


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
        "good_mornings": process_good_mornings
    }
    return processors.get(exercise_name, process_pushup)


def main():
    """Main application with mode selection"""
    print("\n" + "=" * 60)
    print("REPWISE - AI GYM COACH")
    print("=" * 60)

    # Exercise selection
    exercises = {
        "1": "pushup",
        "2": "barbell_squat",
        "3": "air_squat",
        "4": "deadlift",
        "5": "chest_press",
        "6": "shoulder_press",
        "7": "pull_up",
        "8": "donkey_calf_raise",
        "9": "forward_lunge",
        "10": "jump_squat",
        "11": "bulgarian_split_squat",
        "12": "crunches",
        "13": "laying_leg_raises",
        "14": "russian_twist",
        "15": "side_plank_up_down",
        "16": "elbow_side_plank",
        "17": "pike_press",
        "18": "overhead_squat",
        "19": "chin_ups",
        "20": "glute_bridge",
        "21": "kickbacks",
        "22": "single_leg_rdl",
        "23": "good_mornings"
    }

    print("\nSelect Exercise:")
    print("1. Push-up")
    print("2. Barbell Squat")
    print("3. Air Squat")
    print("4. Deadlift")
    print("5. Chest Press")
    print("6. Shoulder Press")
    print("7. Pull-up")
    print("8. Bodyweight Donkey Calf Raise")
    print("9. Forward Lunges")
    print("10. Jump Squats")
    print("11. Bulgarian Split Squat")
    print("12. Crunches")
    print("13. Laying Leg Raises")
    print("14. Bodyweight Russian Twist")
    print("15. Side Plank Up Down")
    print("16. Elbow Side Plank")
    print("17. Bodyweight Pike Press")
    print("18. Pole Overhead Squat")
    print("19. Chin Ups")
    print("20. Glute Bridge")
    print("21. Kickbacks")
    print("22. Single Legged Romanian Deadlifts")
    print("23. Good Mornings")

    exercise_choice = input("\nEnter choice (1-23): ").strip()
    selected_exercise = exercises.get(exercise_choice, "pushup")

    # Mode selection
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
    pose.close()