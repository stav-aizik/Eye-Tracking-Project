# Copyright © 2025  Stav Aizik. All rights reserved. See LICENSE for details.
#Import
import cv2, mediapipe as mp, numpy as np, pygame, time, os, csv, pandas as pd
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # metrics

# CONFIG
CALIB_GRID = 5 # Grid size for calibration points (5x5 = 25 points)
CALIB_FRAMES = 14 # Number of frames to average per calibration point
FRAME_DELAY = 0.03 # Delay between calibration frame captures (seconds)

# Scaling factors to compensate for head movement (X/Y direction):
HEAD_GAIN_X = 0.15
HEAD_GAIN_Y = 0.20

SMOOTH_WINDOW = 12 # Window size for smoothing gaze points (moving average)

# File paths for calibration database, fixations, extended tracking, and statistics
DB_FILE = "calib_db.csv"
FIXATION_CSV = "reading_trace.csv"
EXTENDED_CSV = "extended_eye_tracking.csv"
STATS_CSV = "reading_statistics.csv"

CAM_PREVIEW_SIZE = (320, 240)
FONT_SIZE = 45
LINE_SPACING = 140
TEXT_START_Y = 140

# MediaPipe FaceMesh landmark indices:

PUPIL_IDX = 468 # pupil center (iris midpoint)
NOSE_IDX = 1 # nose tip (reference for head movement compensation)
# 469 & 474 = left/right iris edges (used to estimate pupil diameter)
PUPIL_LEFT = 469
PUPIL_RIGHT = 474


RIGHT_MARGIN = 100
TOTAL_RENDERED_LINES = 0

# Replace with your real text (can be multi-line list; each item is a line)
TEXT_LINES = story = [
    "Once upon a time, there was a small and charming girl named Little Red Riding Hood.",
    "She got that name because of a red hood she always wore, a hood her mother had made for her.",
    "It was a unique hood, made of red fabric, symbolizing her mother’s great love for her daughter.",
    "Whenever she walked down the street, you couldn’t mistake her, because the hood was so bright and beautiful.",
    "Little Red Riding Hood’s mother was very caring, and she constantly warned her daughter to be careful in the forest, not to wander there alone, as there were mysterious and unknown creatures that could be dangerous.",
    "Nevertheless, Little Red Riding Hood was a very curious child, and from time to time she would gaze into the big forest and its deep darkness with great fascination.",
    "One day, Little Red Riding Hood’s mother asked her to carry a small basket and take it for a walk toward her grandmother’s house, which was located on a distant hill, beyond the forest.",
    "Little Red Riding Hood was very happy about the task, because she loved her grandmother very much, and it had been a long time since she had visited her.",
    "She took the basket of food, full of treats for her grandmother, and set off with a big smile on her face.",
    "Her mother smiled at her and said: 'Little Red Riding Hood, remember to be careful in the forest. Don’t take side paths and don’t talk to strangers.'",
    "'Don’t worry, Mom, I’ll be careful,' answered Little Red Riding Hood with shining eyes.",
    "After leaving for the forest, Little Red Riding Hood walked happily, taking the road that led to her grandmother, and the world around her was bright and beautiful.",
    "The sun was shining, and the flowers by the roadside were blooming, giving wonderful colors.",
    "On the way, she met many animals and took the time to watch them and admire them.",
    "She didn’t notice how time passed, until she realized she had gone deeper and deeper into the forest.",
    "At that very moment, in the forest, a hungry wolf was wandering.",
    "The wolf was hiding among the trees, sitting in the shade, waiting for someone or something to pass by.",
    "He didn’t like humans, but he was very hungry that day, and a tempting thought crossed his mind: 'Maybe I can ask that girl where she’s going, and maybe she’ll lead me to her grandmother’s house.'",
    "When he saw Little Red Riding Hood approaching, he came out from between the trees and stood in front of her.",
    "He looked at her with piercing eyes.",
    "'Hey, little girl, where are you going?' asked the wolf in a smooth, cunning voice.",
    "Little Red Riding Hood, not knowing the wolf’s wickedness, replied calmly: 'I’m on my way to my grandmother’s house; she lives beyond the forest.'",
    "'Oh, your grandmother? That sounds interesting,' said the wolf with a sly smile.",
    "'And where exactly does she live?'",
    "Little Red Riding Hood pointed to the path between the trees and said: 'Up there, beyond the hill, in a small house.'",
    "The wolf smiled, pleased with the information he had heard.",
    "He was quick in his thinking, so he ran through side paths to get to the grandmother’s house before Little Red Riding Hood, intending to surprise them both.",
    "After a while, Little Red Riding Hood continued on her way.",
    "She was unaware of what had happened at her grandmother’s house, because she was so focused on her mother’s advice to be careful in the forest.",
    "She had no idea that the wolf had already made his way there.",
    "She arrived at her grandmother’s house, knocked on the door, and when there was no answer, she opened the door herself.",
    "'Grandma, I’ve come,' she called softly, but there was no response.",
    "She went inside, and the atmosphere was not as calm as it used to be.",
    "'Grandma?' she called again, walking toward the bed where her grandmother always slept.",
    "Looking more closely, Little Red Riding Hood noticed something strange.",
    "'Grandma,' she asked curiously, 'why do you have such big ears?'",
    "The wolf, disguised as her grandmother, answered in a deep and calm voice: 'To hear you better, my child.'",
    "Little Red Riding Hood didn’t stop there.",
    "She continued asking questions, but each answer from the wolf was stranger and stranger.",
    "She asked, 'Grandma, why do you have such big eyes?'",
    "The wolf answered, 'To see you better.'",
    "Little Red Riding Hood didn’t give up and asked again: 'Grandma, why do you have such big teeth?'",
    "And the wolf answered, 'To eat you!'",
    "By that time, Little Red Riding Hood already suspected that her grandmother wasn’t really there, and as she turned around, the wolf leaped at her and turned into a predator!",
    "While the wolf chased Little Red Riding Hood, she ran with all her strength through the forest.",
    "In a critical moment, she found a thick tree to hide behind.",
    "She was so scared!",
    "Eventually, a passerby in the forest saw Little Red Riding Hood struggling, and he managed to free her."]
word_boxes = []

EXTENDED_HEADERS = [
    "session_id", "timestamp", "Stimulus", "Export Start [ms]", "Export End [ms]", "Participant",
    "AOI Name", "AOI Size [px]", "AOI Coverage [%]",
    "Fixation Start [ms]", "Fixation End [ms]", "Fixation Duration [ms]",
    "Fixation X [px]", "Fixation Y [px]",
    "Pupil Diameter [px]", "Dispersion X [px]", "Dispersion Y [px]",
    "Mouse X [px]", "Mouse Y [px]", "Behavior"
]

# Prepare model globally so calculate_model_accuracy can access it
poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())


def render_text(screen, font, lines, highlight=None, show_dot=None):
    """
    Draws text with automatic line wrapping:
    if a word doesn’t fit before the right margin – it moves to the next line.
    """
    global TOTAL_RENDERED_LINES
    screen.fill((255, 255, 255))
    word_boxes.clear()

    screen_w, screen_h = screen.get_size()
    x_start = 100
    max_x = screen_w - RIGHT_MARGIN

    x = x_start
    y = TEXT_START_Y
    rendered_lines = 0
    new_line_started = True

    for line in lines:
        for word in line.split():
            surf = font.render(word, True, (0, 0, 0))
            rect = surf.get_rect(topleft=(x, y))

            # # Wrap line if exceeds margin
            if rect.right > max_x and x != x_start:
                y += LINE_SPACING
                x = x_start
                rect.topleft = (x, y)
                new_line_started = True

            # If this is the first word in the line – count the line
            if new_line_started:
                rendered_lines += 1
                new_line_started = False

            screen.blit(surf, rect)
            word_boxes.append((word, rect))
            x += rect.width + 40  # spacing between words

        y += LINE_SPACING
        x = x_start
        new_line_started = True

    TOTAL_RENDERED_LINES = max(rendered_lines, 1)

    if highlight:
        pygame.draw.rect(screen, (0, 255, 0), highlight.inflate(14, 14), 4)
    if show_dot:
        pygame.draw.circle(screen, (255, 0, 0), show_dot, 12, 3)

# Initialize a 2D Kalman filter to smooth gaze points and reduce noise
def init_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

# Create a CSV file with headers if it does not already exist
def init_csv(fname, headers):
    if not os.path.exists(fname):
        with open(fname, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def get_latest_session_id(csv_file):
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file, encoding='utf-8')
    if "session_id" not in df.columns or df.empty:
        return None
    return df["session_id"].iloc[-1]

# Extract pupil center, nose tip, and estimate pupil diameter from MediaPipe FaceMesh
def extract_points(results, shape):
    if not results.multi_face_landmarks:
        return None, None, None
    lm = results.multi_face_landmarks[0].landmark
    ih, iw, _ = shape
    pupil = (int(lm[PUPIL_IDX].x * iw), int(lm[PUPIL_IDX].y * ih))
    nose = (int(lm[NOSE_IDX].x * iw), int(lm[NOSE_IDX].y * ih))
    # pupil diameter
    p1 = np.array([lm[PUPIL_LEFT].x * iw, lm[PUPIL_LEFT].y * ih])
    p2 = np.array([lm[PUPIL_RIGHT].x * iw, lm[PUPIL_RIGHT].y * ih])
    pupil_diameter = np.linalg.norm(p1 - p2)
    return pupil, nose, pupil_diameter

# Compute eye aspect ratio (EAR) used for blink detection
def eye_aspect_ratio(landmarks, indices, image_shape):
    ih, iw, _ = image_shape
    p = [np.array([landmarks[i].x * iw, landmarks[i].y * ih]) for i in indices]
    vertical = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / horizontal

# Log basic fixation event (word, duration, speed, movement, behavior) into CSV
def log_fixation_csv(word, x, y, start, end, speed, dx, dy, behavior, session_id):
    duration = end - start
    timestamp = datetime.now().isoformat(sep=' ', timespec='milliseconds')
    with open(FIXATION_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            session_id, timestamp, word, x, y,
            f"{duration:.3f}", f"{speed:.2f}",
            f"{dx:.1f}", f"{dy:.1f}", behavior
        ])

# Log extended fixation data including AOI size, coverage, pupil diameter, and mouse position
def log_extended_fixation(session_id, participant, stimulus, word, rect, ax, ay,
                          start, end, pupil_diameter, dx, dy, behavior):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end - start
    mouse_x, mouse_y = pygame.mouse.get_pos()
    AOI_size = rect.width * rect.height
    AOI_coverage = (AOI_size / (screen_w * screen_h)) * 100

    row = [
        session_id, timestamp, stimulus,
        round(start * 1000, 1), round(end * 1000, 1), participant,
        word, AOI_size, round(AOI_coverage, 2),
        round(start * 1000, 1), round(end * 1000, 1), round(duration * 1000, 1),
        ax, ay, round(pupil_diameter, 1),
        round(abs(dx), 1), round(abs(dy), 1),
        mouse_x, mouse_y, behavior
    ]

    with open(EXTENDED_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def plot_fixations(session_id):
    if not os.path.exists(FIXATION_CSV):
        return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    df = df[df["session_id"] == session_id]
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration', 'word'])
    if df.empty:
        print("No fixation data for this session.")
        return
    df['word_with_index'] = df.groupby('word').cumcount().astype(str) + '_' + df['word']
    plt.figure(figsize=(30, 9))
    plt.plot(df['word_with_index'], df['duration'], marker='o', linestyle='-', color='darkblue', linewidth=2)
    plt.title(f"Fixation Duration per Word - {session_id}", fontsize=18)
    plt.xlabel("Word (with index)", fontsize=18)
    plt.ylabel("Duration (sec)", fontsize=16)
    plt.xticks(rotation=75, ha='right', fontsize=16)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=1)
    plt.tight_layout()
    plt.show()


def plot_advanced_fixations(session_id):
    if not os.path.exists(FIXATION_CSV): return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    df = df[df["session_id"] == session_id]
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['duration', 'speed', 'behavior'])
    if df.empty:
        print("No data for the selected session.")
        return

    color_map = {'normal': 'green', 'skip': 'red', 'regression': 'blue'}
    colors = df['behavior'].map(color_map)
    rolling_duration = df['duration'].rolling(window=10, min_periods=1).mean()
    rolling_speed = df['speed'].rolling(window=10, min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    plt.scatter(df.index, df['duration'], c=colors, label='Fixation Duration (by behavior)', s=50)
    plt.plot(df.index, rolling_duration, color='black', linestyle='--', label='Moving Avg (duration)')
    plt.plot(df.index, rolling_speed, color='orange', linestyle='-', label='Moving Avg (speed)', alpha=0.6)
    legend_elements = [
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='Normal'),
        mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='Skip'),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='Regression'),
        mlines.Line2D([], [], color='black', linestyle='--', label='Moving Avg Duration'),
        mlines.Line2D([], [], color='orange', linestyle='-', label='Moving Avg Speed'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Fixation Duration & Speed with Behavior Coloring")
    plt.xlabel("Fixation #")
    plt.ylabel("Time (sec) / Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_word_fixation_counts(session_id):
    if not os.path.exists(FIXATION_CSV):
        print("Fixation CSV not found.")
        return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    df = df[df["session_id"] == session_id]
    if df.empty:
        print("No data for the selected session.")
        return

    word_counts = df['word'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    word_counts.plot(kind='bar', color='skyblue')
    plt.title(f"Fixation Count per Word - {session_id}")
    plt.xlabel("Word")
    plt.ylabel("Fixation Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_word_durations(session_id):
    if not os.path.exists(FIXATION_CSV):
        print("Fixation CSV not found.")
        return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    df = df[df["session_id"] == session_id]
    df["word"] = df["word"].astype(str)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    if df.empty:
        print("No data for the selected session.")
        return
    word_avg = df.groupby("word")['duration'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    word_avg.plot(kind='bar', color='teal')
    plt.title(f"Average Fixation Duration per Word - {session_id}")
    plt.xlabel("Word")
    plt.ylabel("Average Duration (sec)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def save_and_plot_statistics():
    if not os.path.exists(FIXATION_CSV): return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    session_id = get_latest_session_id(FIXATION_CSV)
    if session_id:
        df = df[df["session_id"] == session_id]
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration', 'behavior'])
    if len(df) == 0:
        print(" No fixations to process.")
        return

    start_time = df['timestamp'].iloc[0]
    avg_duration = df['duration'].mean()
    num_regressions = (df['behavior'] == 'regression').sum()
    total_time = df['duration'].sum()
    skip_time = df[df['behavior'] == 'skip']['duration'].sum()
    normal_time = df[df['behavior'] == 'normal']['duration'].sum()
    regression_time = df[df['behavior'] == 'regression']['duration'].sum()

    skip_ratio = (skip_time / total_time) * 100 if total_time > 0 else 0
    normal_ratio = (normal_time / total_time) * 100 if total_time > 0 else 0
    regression_ratio = (regression_time / total_time) * 100 if total_time > 0 else 0
    num_words_read = (df['behavior'] == 'normal').sum()
    time_minutes = total_time / 60
    reading_speed_wpm = num_words_read / time_minutes if time_minutes > 0 else 0

    stats = {
        "Session ID": session_id,
        "Start Time": start_time,
        "Average Fixation Duration (sec)": f"{avg_duration:.3f} sec",
        "Number of Regressions": int(num_regressions),
        "Skip Time Percentage": f"{skip_ratio:.2f}%",
        "Normal Fixation Time Percentage": f"{normal_ratio:.2f}%",
        "Regression Time Percentage": f"{regression_ratio:.2f}%",
        "Estimated Reading Speed (WPM)": f"{reading_speed_wpm:.1f} WPM"
    }

    with open(STATS_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in stats.items():
            writer.writerow([key, value])

    numeric_stats = {
        "Avg Fixation Duration": avg_duration,
        "Regr. Count": num_regressions,
        "Skip Time %": skip_ratio,
        "Normal Time %": normal_ratio,
        "Regression Time %": regression_ratio,
        "Reading Speed (WPM)": reading_speed_wpm
    }

    plt.figure(figsize=(10, 6))
    labels = list(numeric_stats.keys())
    values = list(numeric_stats.values())
    bars = plt.barh(labels, values, color=['blue', 'red', 'orange', 'green', 'purple', 'gray'])
    plt.title("Reading Statistics Analysis", fontsize=16)
    plt.xlabel("Value", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    for label, bar in zip(labels, bars):
        width = bar.get_width()
        unit = "%"
        if "Duration" in label:
            unit = " sec"
        elif "WPM" in label:
            unit = " WPM"
        elif "Count" in label:
            unit = ""
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.2f}{unit}", va='center', fontsize=10)

    plt.show()


def plot_word_timings(session_id):
    if not os.path.exists(FIXATION_CSV):
        print("Fixation CSV not found.")
        return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    df = df[df["session_id"] == session_id]
    if df.empty:
        print("No data for the selected session.")
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'word'])
    start_time = df['timestamp'].iloc[0]
    df['relative_time_ms'] = (df['timestamp'] - start_time).dt.total_seconds() * 1000
    plt.figure(figsize=(14, 6))
    plt.scatter(df['relative_time_ms'], df['word'], color='purple', s=40)
    for _, row in df.iterrows():
        plt.text(row['relative_time_ms'], row['word'], f"{row['relative_time_ms']:.0f} ms",
                 ha='center', va='bottom', fontsize=8, rotation=45)
    plt.title(f"Word Fixation Times - {session_id}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Word")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_blinks_over_time(blink_times, session_start_time):
    if not blink_times:
        print("[INFO] No blinks recorded during session.")
        return
    times_relative = [round(bt - session_start_time, 2) for bt in blink_times]
    plt.figure(figsize=(10, 4))
    plt.scatter(times_relative, [1] * len(times_relative), color='black', s=40)
    for t in times_relative:
        plt.text(t, 1.02, f"{t:.2f}s", ha='center', fontsize=9, rotation=45)
    plt.title("Blink Events Over Time")
    plt.xlabel("Time (seconds)")
    plt.yticks([])
    plt.ylim(0.95, 1.1)
    plt.tight_layout()
    plt.show()


# Helper: pick closest word to gaze
def highlight_current_word(gaze_position, word_boxes):
    min_distance = float('inf')
    current_word = None
    highlight_rect = None
    for word, rect in word_boxes:
        word_center = rect.center
        distance = np.linalg.norm(np.array(gaze_position) - np.array(word_center))
        if distance < min_distance:
            min_distance = distance
            current_word = word
            highlight_rect = rect
    return current_word, highlight_rect


# Accuracy calc uses global poly_model
def calculate_model_accuracy(eyes_captured, scr_captured):
    predictions = poly_model.predict(eyes_captured)
    rms = np.sqrt(mean_squared_error(scr_captured, predictions))
    mae = mean_absolute_error(scr_captured, predictions)
    r2 = r2_score(scr_captured, predictions)
    return rms, mae, r2


# Log accuracy rows into STATS_CSV 
def log_accuracy_to_csv(session_id, rms, mae, r2):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATS_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([session_id, timestamp, "RMS", f"{rms:.3f}"])
        writer.writerow([session_id, timestamp, "MAE", f"{mae:.3f}"])
        writer.writerow([session_id, timestamp, "R^2", f"{r2:.3f}"])




def clamp_text_offset(lines):
    """
    Restricts the vertical offset so that the text does not leave the screen,
    according to the actual number of lines rendered after wrapping.
    """
    global TEXT_START_Y, screen_h, TOTAL_RENDERED_LINES
    max_y = 140  # default top position
    lines_count = max(TOTAL_RENDERED_LINES, len(lines))
    min_y = screen_h - (lines_count * LINE_SPACING) - 100  # small bottom margin

    if min_y > max_y:
        min_y = max_y
    TEXT_START_Y = max(min(TEXT_START_Y, max_y), min_y)


def main():
    global screen_w, screen_h, poly_model, TEXT_START_Y
    session_start_time = None

    #  Session setup
    # Ask for participant ID, create session ID, initialize blink counters
    participant = input(" Enter participant ID: ")
    stimulus = " ".join(TEXT_LINES)[:50]

    # SESSION
    session_id = f"TRAIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # blink
    blink_counter = 0
    blink_timestamps = []
    blink_flag = False
    EAR_THRESHOLD = 0.2

    #  CSV initialization 
    # Ensure calibration, fixation, and extended CSV files exist
    init_csv(DB_FILE, ["pupil_x", "pupil_y", "scr_x", "scr_y", "timestamp"])
    init_csv(FIXATION_CSV, ["session_id", "timestamp", "word", "x", "y", "duration", "speed", "dx", "dy", "behavior"])
    init_csv(EXTENDED_CSV, EXTENDED_HEADERS)

    # PYGAME + CAMERA
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_w, screen_h = screen.get_size()
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit(" Webcam not available")

    # MediaPipe
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(80, 220, 100))

    # CALIB
    calib_pts = [(int((i % CALIB_GRID + 1) * screen_w / (CALIB_GRID + 1)),
                  int((i // CALIB_GRID + 1) * screen_h / (CALIB_GRID + 1))) for i in range(CALIB_GRID ** 2)]
    eyes_captured, scr_captured = [], []
    kalman = init_kalman()
    avg_queue = deque(maxlen=SMOOTH_WINDOW)
    baseline_nose = None
    fixation_start, fixation_word = None, None
    prev_pt, prev_t = None, None

    mode = "calib"
    idx = 0
    running = True
    render_text(screen, font, TEXT_LINES)

    while running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(frame_rgb)
        pupil, nose, pupil_diameter = extract_points(res, frame.shape)
        disp_frame = frame.copy()

        if res.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                disp_frame, res.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_TESSELATION, None, drawing_spec)

            # Blink detection only during run
            if mode == "run":
                lm = res.multi_face_landmarks[0].landmark
                ear_left = eye_aspect_ratio(lm, [362, 385, 387, 263, 373, 380], frame.shape)
                ear_right = eye_aspect_ratio(lm, [33, 160, 158, 133, 153, 144], frame.shape)
                ear_avg = (ear_left + ear_right) / 2.0
                if ear_avg < EAR_THRESHOLD and not blink_flag:
                    blink_counter += 1
                    blink_timestamps.append(time.time())
                    blink_flag = True
                elif ear_avg >= EAR_THRESHOLD:
                    blink_flag = False

        flipped_frame = cv2.flip(disp_frame, 1)
        cam_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(cv2.resize(flipped_frame, CAM_PREVIEW_SIZE), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

            # --- Manual scrolling controls (active only in "run" mode) ---
            if e.type == pygame.KEYDOWN and mode == "run":
                if e.key == pygame.K_UP:
                    TEXT_START_Y += 20
                    clamp_text_offset(TEXT_LINES)
                elif e.key == pygame.K_DOWN:
                    TEXT_START_Y -= 20
                    clamp_text_offset(TEXT_LINES)
                elif e.key == pygame.K_PAGEUP:
                    TEXT_START_Y += LINE_SPACING
                    clamp_text_offset(TEXT_LINES)
                elif e.key == pygame.K_PAGEDOWN:
                    TEXT_START_Y -= LINE_SPACING
                    clamp_text_offset(TEXT_LINES)
                elif e.key == pygame.K_HOME:
                    TEXT_START_Y = 140
                    clamp_text_offset(TEXT_LINES)
                elif e.key == pygame.K_END:
                    TEXT_START_Y = screen_h - (len(TEXT_LINES) * LINE_SPACING) - 100
                    clamp_text_offset(TEXT_LINES)

            elif e.type == pygame.MOUSEWHEEL and mode == "run":
                # e.y > 0 when wheel up; < 0 when wheel down
                TEXT_START_Y += e.y * 40
                clamp_text_offset(TEXT_LINES)

            #  Calibration mode 
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE and pupil and mode == "calib":
                samples = []
                for _ in range(CALIB_FRAMES):
                    _, fr = cap.read()
                    r = face_mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    p, _, _ = extract_points(r, fr.shape)
                    if p:
                        samples.append(p)
                        time.sleep(FRAME_DELAY)
                if samples:
                    avg_eye = tuple(map(int, np.mean(samples, axis=0)))
                    eyes_captured.append(avg_eye)
                    scr_captured.append(calib_pts[idx])
                    with open(DB_FILE, "a", encoding='utf-8') as f:
                        f.write(f"{avg_eye[0]},{avg_eye[1]},{calib_pts[idx][0]},{calib_pts[idx][1]},{time.time()}\n")
                    idx += 1
                    if idx == len(calib_pts):
                        # fit model
                        poly_model.fit(np.array(eyes_captured), np.array(scr_captured))
                        # after fitting, compute & log accuracy
                        rms, mae, r2 = calculate_model_accuracy(np.array(eyes_captured), np.array(scr_captured))
                        log_accuracy_to_csv(session_id, rms, mae, r2)
                        baseline_nose = nose
                        mode = "run"
                        render_text(screen, font, TEXT_LINES)
        # Run mode 
        # Track gaze using model + Kalman + smoothing
        # Detect blinks using EAR and log fixations
        if pupil and nose and mode == "run":
            if session_start_time is None:
                session_start_time = time.time()  # real start time
            gaze = poly_model.predict([pupil])[0]
            gx = int(gaze[0] + (baseline_nose[0] - nose[0]) * HEAD_GAIN_X)
            gy = int(gaze[1] - (nose[1] - baseline_nose[1]) * HEAD_GAIN_Y)

            measured = np.array([[np.float32(gx)], [np.float32(gy)]])
            kalman.correct(measured)
            predicted = kalman.predict()
            gx, gy = int(predicted[0].item()), int(predicted[1].item())

            now = time.time()
            if prev_pt and prev_t:
                dt = max(now - prev_t, 1e-3)
                dx, dy = gx - prev_pt[0], gy - prev_pt[1]
                speed = np.hypot(dx, dy) / dt
            else:
                dx, dy, speed = 0, 0, 0
            prev_pt, prev_t = (gx, gy), now
            avg_queue.append((gx, gy))
            ax = int(np.mean([p[0] for p in avg_queue]))
            ay = int(np.mean([p[1] for p in avg_queue]))

            current_word, highlight = highlight_current_word((ax, ay), word_boxes)

            if current_word:
                if fixation_word != current_word:
                    if fixation_word and fixation_start:
                        duration = time.time() - fixation_start
                        behavior = "normal"
                        if duration < 0.1:
                            behavior = "skip"
                        elif dx < 0:
                            behavior = "regression"
                        log_fixation_csv(fixation_word, ax, ay, fixation_start, time.time(), speed, dx, dy, behavior,
                                         session_id)
                        # find rect for previous fixation word (fallback to highlight if available)
                        prev_rect = None
                        for w, r in word_boxes:
                            if w == fixation_word:
                                prev_rect = r
                                break
                        if prev_rect is None:
                            prev_rect = highlight if highlight is not None else pygame.Rect(ax, ay, 1, 1)
                        log_extended_fixation(session_id, participant, stimulus, fixation_word, prev_rect, ax, ay,
                                              fixation_start, time.time(), pupil_diameter, dx, dy, behavior)

                    fixation_start = time.time()
                    fixation_word = current_word

            # Make sure position is clamped before rendering (in case screen resized later)
            clamp_text_offset(TEXT_LINES)
            render_text(screen, font, TEXT_LINES, highlight, (ax, ay))

        elif mode == "calib" and idx < CALIB_GRID ** 2:
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 0, 0), calib_pts[idx], 18)

        screen.blit(cam_surface, (0, screen_h - CAM_PREVIEW_SIZE[1]))
        pygame.display.flip()
        clock.tick(30)
        
    #  End of session
    # Close camera, quit pygame, and plot results/statistics
    pygame.quit()
    cap.release()
    plot_fixations(session_id)
    plot_word_fixation_counts(session_id)
    save_and_plot_statistics()
    plot_word_durations(session_id)
    plot_blinks_over_time(blink_timestamps, session_start_time)
    plot_word_timings(session_id)
    print(f"Total blinks detected: {blink_counter}")
    print(" Program completed successfully.")


if __name__ == "__main__":
    main()



