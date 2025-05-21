# Copyright © 2025  Stav Aizik , Tal Malka and Guy Elkayam. All rights reserved. See LICENSE for details.
import cv2, mediapipe as mp, numpy as np, pygame, time, os, csv, pandas as pd
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# CONFIG
CALIB_GRID = 5
CALIB_FRAMES = 14
FRAME_DELAY = 0.03
HEAD_GAIN_X = 0.15
HEAD_GAIN_Y = 0.20
SMOOTH_WINDOW = 12
DB_FILE = "calib_db.csv"
FIXATION_CSV = "reading_trace.csv"
EXTENDED_CSV = "extended_eye_tracking.csv"
STATS_CSV = "reading_statistics.csv"
CAM_PREVIEW_SIZE = (320, 240)
FONT_SIZE = 45
LINE_SPACING = 100
TEXT_START_Y = 140

PUPIL_IDX = 468
NOSE_IDX = 1
PUPIL_LEFT = 469
PUPIL_RIGHT = 474

TEXT_LINES = [
    "Reading is the gateway to knowledge.",
    "It allows you to explore worlds, ideas,",
    "and perspectives.",
    "Through reading, we learn, grow, and connect."

]
word_boxes = []

EXTENDED_HEADERS = [
    "timestamp", "Stimulus", "Export Start [ms]", "Export End [ms]", "Participant",
    "AOI Name", "AOI Size [px]", "AOI Coverage [%]",
    "Fixation Start [ms]", "Fixation End [ms]", "Fixation Duration [ms]",
    "Fixation X [px]", "Fixation Y [px]",
    "Pupil Diameter [px]", "Dispersion X [px]", "Dispersion Y [px]",
    "Mouse X [px]", "Mouse Y [px]", "Behavior"

]
def render_text(screen, font, lines, highlight=None, show_dot=None):
    screen.fill((255, 255, 255))
    word_boxes.clear()
    y = TEXT_START_Y
    for line in lines:
        x = 100
        for word in line.split():
            surf = font.render(word, True, (0, 0, 0))
            rect = surf.get_rect(topleft=(x, y))
            screen.blit(surf, rect)
            word_boxes.append((word, rect))
            x += rect.width + 40
        y += LINE_SPACING
    if highlight:
        pygame.draw.rect(screen, (0, 255, 0), highlight.inflate(14, 14), 4)
    if show_dot:
        pygame.draw.circle(screen, (255, 0, 0), show_dot, 12, 3)

def init_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

def init_csv(fname, headers):
    if not os.path.exists(fname):
        with open(fname, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
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

def log_fixation_csv(word, x, y, start, end, speed, dx, dy, behavior):
    duration = end - start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FIXATION_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, word, x, y,
            f"{duration:.3f}", f"{speed:.2f}",
            f"{dx:.1f}", f"{dy:.1f}", behavior
        ])


def log_extended_fixation(participant, stimulus, word, rect, ax, ay,
                          start, end, pupil_diameter, dx, dy, behavior):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end - start
    mouse_x, mouse_y = pygame.mouse.get_pos()
    AOI_size = rect.width * rect.height
    AOI_coverage = (AOI_size / (screen_w * screen_h)) * 100

    row = [
        timestamp, stimulus,
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


def plot_fixations():
    if not os.path.exists(FIXATION_CSV): return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    plt.figure(figsize=(10,5))
    plt.plot(df['duration'], marker='o')
    plt.title("Fixation Durations Over Time")
    plt.xlabel("Fixation #")
    plt.ylabel("Duration (sec)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_advanced_fixations():
    if not os.path.exists(FIXATION_CSV): return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['duration', 'speed', 'behavior'])

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

def plot_word_fixation_counts():
    if not os.path.exists(FIXATION_CSV):
        print("Fixation CSV not found.")
        return

    try:
        df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
        word_counts = df['word'].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(12, 5))
        word_counts.plot(kind='bar', color='skyblue')
        plt.title("Fixation Count per Word")
        plt.xlabel("Word")
        plt.ylabel("Fixation Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error generating word fixation graph:", e)

def save_and_plot_statistics():
    if not os.path.exists(FIXATION_CSV): return
    df = pd.read_csv(FIXATION_CSV, encoding='utf-8')
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

    skip_ratio = (skip_time / total_time) * 100 if total_time > 0 else 0
    normal_ratio = (normal_time / total_time) * 100 if total_time > 0 else 0
    num_words_read = (df['behavior'] == 'normal').sum()
    time_minutes = total_time / 60
    reading_speed_wpm = num_words_read / time_minutes if time_minutes > 0 else 0

    stats = {
        "Start Time": start_time,
        "Average Fixation Duration (sec)": round(avg_duration, 3),
        "Number of Regressions": int(num_regressions),
        "Skip Time Percentage": round(skip_ratio, 2),
        "Normal Fixation Time Percentage": round(normal_ratio, 2),
        "Estimated Reading Speed (WPM)": round(reading_speed_wpm, 1)
    }

    with open(STATS_CSV, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in stats.items():
            writer.writerow([key, value])

    plt.figure(figsize=(10, 5))
    labels = list(stats.keys())[1:]
    values = list(stats.values())[1:]
    bars = plt.barh(labels, values, color=['blue', 'red', 'orange', 'green', 'purple'])
    plt.title("Reading Statistics Analysis")
    plt.xlabel("Value")
    plt.tight_layout()
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width}', va='center')
    plt.show()
def main():
    global screen_w, screen_h

    # PARTICIPANT ID
    participant = input(" Enter participant ID: ")
    stimulus = " ".join(TEXT_LINES)[:50]

    # CSV
    init_csv(DB_FILE, ["pupil_x", "pupil_y", "scr_x", "scr_y", "timestamp"])
    init_csv(FIXATION_CSV, ["timestamp", "word", "x", "y", "duration", "speed", "dx", "dy", "behavior"])
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

    # REGRESSION
    poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())

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

        flipped_frame = cv2.flip(disp_frame, 1)
        cam_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(cv2.resize(flipped_frame, CAM_PREVIEW_SIZE), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
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
                        poly_model.fit(np.array(eyes_captured), np.array(scr_captured))
                        baseline_nose = nose
                        mode = "run"
                        render_text(screen, font, TEXT_LINES)

        if pupil and nose and mode == "run":
            gaze = poly_model.predict([pupil])[0]
            gx = int(gaze[0] + (baseline_nose[0] - nose[0]) * HEAD_GAIN_X)
            gy = int(gaze[1] - (nose[1] - baseline_nose[1]) * HEAD_GAIN_Y)

            measured = np.array([[np.float32(gx)], [np.float32(gy)]])
            kalman.correct(measured)
            predicted = kalman.predict()
            gx, gy = int(predicted[0]), int(predicted[1])

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

            highlight, current_word = None, None
            for word, rect in word_boxes:
                if rect.inflate(30, 30).collidepoint((ax, ay)):
                    current_word = word
                    highlight = rect
                    break

            if current_word:
                if fixation_word != current_word:
                    if fixation_word and fixation_start:
                        duration = time.time() - fixation_start
                        behavior = "normal"
                        if duration < 0.1:
                            behavior = "skip"
                        elif dx < 0:
                            behavior = "regression"
                        log_fixation_csv(fixation_word, ax, ay, fixation_start, time.time(), speed, dx, dy, behavior)
                        log_extended_fixation(participant, stimulus, fixation_word, highlight, ax, ay,
                                              fixation_start, time.time(), pupil_diameter, dx, dy, behavior)

                    fixation_start = time.time()
                    fixation_word = current_word

            render_text(screen, font, TEXT_LINES, highlight, (ax, ay))

        elif mode == "calib" and idx < CALIB_GRID ** 2:
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 0, 0), calib_pts[idx], 18)

        screen.blit(cam_surface, (0, screen_h - CAM_PREVIEW_SIZE[1]))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    cap.release()
    plot_fixations()
    plot_advanced_fixations()
    save_and_plot_statistics()
    plot_word_fixation_counts()
    print(" Program completed successfully.")

if __name__ == "__main__":
    main()
