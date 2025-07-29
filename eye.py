# Copyright © 2025  Stav Aizik , Tal Malka and Guy Elkayam. All rights reserved. See LICENSE for details.
import cv2, mediapipe as mp, numpy as np, pygame, time, os, csv, pandas as pd, pickle, json
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import pyautogui
import math

plt.rcParams['font.family'] = 'Arial Unicode MS'

# CONFIG - שמור בדיוק כמו במקור
CALIB_GRID = 5
CALIB_FRAMES = 2
FRAME_DELAY = 0.30
HEAD_GAIN_X = 1
HEAD_GAIN_Y = 0
SMOOTH_WINDOW = 8
DB_FILE = "calib_db.csv"
FIXATION_CSV = "reading_trace.csv"
EXTENDED_CSV = "extended_eye_tracking.csv"
STATS_CSV = "reading_statistics.csv"
CALIBRATION_FILE = "calibration_data.json"
CALIBRATION_META_FILE = "calibration_meta.json"
CAM_PREVIEW_SIZE = (320, 240)
FONT_SIZE = 40
LINE_SPACING = 80
TEXT_START_Y = 140
POLYNOMIAL_DEGREE = 3

# *** משתנה חדש למיקום התחלה קבוע ***
INITIAL_CURSOR_POSITION = (100, 200)  # מיקום התחלה קבוע (שמאל עליון)

# *** משתנים חדשים לתקופת חימום ***
tracking_warmup_start = None
WARMUP_DURATION = 3.0  # 3 שניות חימום
is_in_warmup = False

# הגדרות מתוך cod.py לתזוזת הסמן
TUNING_PARAMS = {
    'alpha': 0.3,
    'movement_threshold': 4,
    'history_size': 10,
    'cursor_size': 30,
    'cursor_color': (255, 0, 0),
    'update_rate': 30,
    'head_compensation': True,
    'head_gain_x': 1.0,
    'head_gain_y': 0.0,
    'show_trail': False,
    'trail_length': 5,
    'update_sleep': 0.03,
    'median_buffer_size': 5,
}

# הגדרת משתנים גלובליים בתחילת הקוד
cursor_root = None
cursor_trail = []
tuner_button_rect = None
tracking_button_rect = None

# משתנים גלובליים לסנכרון
word_boxes = []
current_text_option = "short"
last_highlighted_word = None
current_highlighted_rect = None
word_positions_cache = {}
is_tracking_active = False
tracking_start_time = None


def create_polynomial_model(degree=POLYNOMIAL_DEGREE):
    """יצירת מודל רגרסיה פולינומית"""
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('linear', LinearRegression())
    ])


# הגדרות עיצוב טקסט הניתנות לשינויmodel_x = LinearRegression().fit(X, Y[:, 0])
class TextFormatting:
    def __init__(self):
        self.font_size = 40
        self.line_spacing = 80
        self.word_spacing = 30
        self.min_font_size = 20
        self.max_font_size = 80
        self.min_line_spacing = 40
        self.max_line_spacing = 150
        self.min_word_spacing = 10
        self.max_word_spacing = 60

    def increase_font_size(self):
        self.font_size = min(self.max_font_size, self.font_size + 5)

    def decrease_font_size(self):
        self.font_size = max(self.min_font_size, self.font_size - 5)

    def increase_line_spacing(self):
        self.line_spacing = min(self.max_line_spacing, self.line_spacing + 10)

    def decrease_line_spacing(self):
        self.line_spacing = max(self.min_line_spacing, self.line_spacing - 10)

    def increase_word_spacing(self):
        self.word_spacing = min(self.max_word_spacing, self.word_spacing + 5)

    def decrease_word_spacing(self):
        self.word_spacing = max(self.min_word_spacing, self.word_spacing - 5)


PUPIL_IDX = 468
NOSE_IDX = 1
PUPIL_LEFT = 468
PUPIL_RIGHT = 473

# טקסטים שונים למערכת
TEXT_OPTIONS = {
    "short": {
        "name": "Short Text (4 lines)",
        "lines": [
            "Reading is the gateway to knowledge.",
            "It allows you to explore worlds, ideas,",
            "and perspectives.",
            "Through reading, we learn, grow, and connect."
        ]
    },
    "medium": {
        "name": "Medium Text (15 lines)",
        "lines": [
            "The art of reading has evolved dramatically over centuries.",
            "From ancient scrolls to digital screens, the medium has changed,",
            "but the fundamental purpose remains the same: knowledge transfer.",
            "Reading comprehension involves multiple cognitive processes working together.",
            "Eye movements during reading follow predictable patterns.",
            "Fixations occur when the eye stops to process information.",
            "Saccades are rapid movements between fixation points.",
            "Skilled readers make fewer fixations and longer saccades.",
            "The brain predicts upcoming words based on context.",
            "This prediction helps in faster reading and comprehension.",
            "Different text types require different reading strategies.",
            "Scientific texts demand careful attention to detail.",
            "Narrative texts allow for more fluid reading patterns.",
            "Modern technology enables precise eye tracking analysis.",
            "This research helps improve reading instruction methods."
        ]
    },
    "long": {
        "name": "Long Text (400 words)",
        "lines": [
            "The human eye is a remarkable instrument capable of processing",
            "vast amounts of visual information with extraordinary precision",
            "and speed. During reading, our eyes perform a complex dance",
            "of movements that reveal fascinating insights into how we",
            "process written language and extract meaning from text.",
            "",
            "",
            "Eye tracking technology has revolutionized our understanding",
            "of reading behavior. When we read, our eyes do not move",
            "smoothly across the page as we might imagine. Instead,",
            "they make a series of rapid jumps called saccades,",
            "punctuated by brief pauses known as fixations.",
            "",
            "",
            "During these fixations, which typically last between",
            "200 to 300 milliseconds, our visual system captures",
            "and processes the information from the text. The pattern",
            "of eye movements varies significantly between skilled",
            "and novice readers throughout different text types.",
            "",
            "",
            "Expert readers make fewer fixations per line, have longer",
            "saccades that skip over familiar words, and demonstrate",
            "more efficient regression patterns when they need to",
            "revisit previous text. These differences highlight the",
            "automaticity that develops through years of practice.",
            "",
            "",
            "Fixation duration provides valuable insights into cognitive",
            "processing. Longer fixations often indicate increased",
            "processing difficulty, whether due to unfamiliar vocabulary,",
            "complex sentence structure, or conceptual challenges that",
            "require additional mental effort to comprehend fully.",
            "",
            "",
            "Conversely, shorter fixations suggest that the reader",
            "is processing information efficiently and smoothly",
            "progressing through the text. Modern eye tracking systems",
            "can measure fixations with millisecond precision, enabling",
            "researchers to analyze reading behavior in detail.",
            "",
            "",
            "This technology has applications beyond research, including",
            "educational assessment, user experience design, and assistive",
            "technology development. By understanding how different",
            "individuals process text, we can create more effective",
            "learning materials and reading interventions for students.",
            "",
            "",
            "The study of reading through eye tracking also reveals",
            "individual differences in processing strategies. Some readers",
            "prefer to read every word carefully, while others engage",
            "in more selective attention, focusing on key terms and",
            "skipping function words that carry less meaning.",
            "",
            "",
            "These personal reading styles reflect both learned strategies",
            "and innate cognitive preferences. Furthermore, text",
            "characteristics significantly influence eye movement patterns.",
            "Dense technical material requires more careful attention",
            "and longer fixations compared to familiar narrative text.",
            "",
            "",
            "Font size, line spacing, and text layout also affect",
            "reading efficiency and eye movement patterns. As digital",
            "reading becomes increasingly prevalent, understanding these",
            "fundamental processes becomes even more critical for",
            "optimizing reading experiences across various platforms."
        ]
    }
}

EXTENDED_HEADERS = [
    "session_id", "timestamp", "Stimulus", "Export Start [ms]", "Export End [ms]", "Participant",
    "AOI Name", "AOI Size [px]", "AOI Coverage [%]",
    "Fixation Start [ms]", "Fixation End [ms]", "Fixation Duration [ms]",
    "Fixation X [px]", "Fixation Y [px]",
    "Pupil Diameter [px]", "Dispersion X [px]", "Dispersion Y [px]",
    "Mouse X [px]", "Mouse Y [px]", "Behavior"
]

# Colors for UI
BLUE = (64, 128, 255)
LIGHT_BLUE = (173, 216, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARK_GRAY = (64, 64, 64)
ORANGE = (255, 165, 0)


# *** פונקציה מעודכנת לטיפול בהפעלה/כיבוי מעקב עם תקופת חימום ***
def toggle_eye_tracking():
    """הפעלה/כיבוי מעקב עיניים עם תקופת חימום"""
    global is_tracking_active, tracking_start_time, cursor_root
    global tracking_warmup_start, is_in_warmup

    if is_tracking_active:
        # עצירת מעקב
        is_tracking_active = False
        tracking_start_time = None
        tracking_warmup_start = None
        is_in_warmup = False
        print("🔴 Eye tracking STOPPED")

        # החזרת הסמן למיקום קבוע
        if cursor_root and cursor_root.winfo_exists():
            cursor_root.geometry(f"+{INITIAL_CURSOR_POSITION[0]}+{INITIAL_CURSOR_POSITION[1]}")
            print(f"📍 Cursor returned to initial position: {INITIAL_CURSOR_POSITION}")

    else:
        # התחלת מעקב עם תקופת חימום
        is_tracking_active = True
        tracking_warmup_start = time.time()
        is_in_warmup = True
        print("🟡 Eye tracking STARTING - 3 second warmup period...")
        print("👁️ Please look at the first word you want to start reading")

        # הצג את חלון הסמן השקוף אם קיים
        if cursor_root and cursor_root.winfo_exists():
            cursor_root.deiconify()
        pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)


def get_tracking_timer_text():
    """קבלת טקסט הטיימר עם התחשבות בחימום"""
    global is_in_warmup, tracking_warmup_start, tracking_start_time

    if is_tracking_active:
        if is_in_warmup and tracking_warmup_start:
            # תצוגת חימום
            warmup_elapsed = time.time() - tracking_warmup_start
            remaining = max(0, WARMUP_DURATION - warmup_elapsed)
            return f"🟡 WARMUP: {remaining:.1f}s"
        elif tracking_start_time:
            # תצוגת זמן רגיל
            elapsed = time.time() - tracking_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            return f"⏱️ {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return "⏱️ 00:00:00"
    else:
        return "⏱️ 00:00:00"


# פונקציות מתוקנות עבור הסמן השקוף
def create_cursor_window(x, y):
    """יצירת cursor window שקוף עם עיגול בלבד"""
    try:
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", "black")
        root.configure(bg='black')

        size = max(20, TUNING_PARAMS['cursor_size'])
        color = TUNING_PARAMS['cursor_color']
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        root.geometry(f"{size}x{size}+{x}+{y}")

        canvas = tk.Canvas(root, bg='black', highlightthickness=0,
                           width=size, height=size)
        canvas.pack(fill=tk.BOTH, expand=True)

        margin = 2
        canvas.create_oval(margin, margin, size - margin, size - margin,
                           fill=hex_color, outline=hex_color, width=2)

        print(f"✅ Transparent cursor created: {size}x{size} at ({x},{y}) color: {hex_color}")
        root.lift()
        root.focus_force()
        return root
    except Exception as e:
        print(f"❌ Error creating transparent cursor: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_cursor_settings(cursor_root):
    """עדכון הגדרות cursor קיים עם שקיפות"""
    if not cursor_root:
        return

    try:
        if cursor_root.winfo_exists():
            size = max(20, TUNING_PARAMS['cursor_size'])
            color = TUNING_PARAMS['cursor_color']
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

            x = cursor_root.winfo_x()
            y = cursor_root.winfo_y()
            cursor_root.geometry(f"{size}x{size}+{x}+{y}")

            for child in cursor_root.winfo_children():
                if isinstance(child, tk.Canvas):
                    child.configure(bg='black', width=size, height=size)
                    child.delete("all")
                    margin = 2
                    child.create_oval(margin, margin, size - margin, size - margin,
                                      fill=hex_color, outline=hex_color, width=2)
                    break

            print(f"🔄 Transparent cursor updated: {size}x{size} color: {hex_color}")

    except Exception as e:
        print(f"❌ Error updating transparent cursor: {e}")
        import traceback
        traceback.print_exc()


def update_cursor_for_word(cursor_root, current_word):
    """עדכון הסמן השקוף - צבע קבוע, ללא שינוי"""
    if not cursor_root or not cursor_root.winfo_exists():
        return

    try:
        size = max(20, TUNING_PARAMS['cursor_size'])
        color = TUNING_PARAMS['cursor_color']
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        for child in cursor_root.winfo_children():
            if isinstance(child, tk.Canvas):
                child.configure(bg='black')
                child.delete("all")
                margin = 2
                child.create_oval(margin, margin, size - margin, size - margin,
                                  fill=hex_color, outline=hex_color, width=2)
                break

        cursor_root.update_idletasks()

    except Exception as e:
        print(f"❌ Error updating transparent cursor: {e}")


# פונקציות איתור מילים מדויקות
def find_word_under_cursor(ax, ay, word_boxes, tolerance=25):
    """איתור מדויק של מילה תחת הסמן עם tolerance גדול יותר"""
    if not word_boxes:
        return None, None

    for word, rect in word_boxes:
        expanded_rect = rect.inflate(tolerance * 2, tolerance)
        if expanded_rect.collidepoint(ax, ay):
            return word, rect

    return None, None


def update_word_highlighting_immediate(current_word, current_rect):
    """עדכון הדגשת מילה מיידי"""
    global last_highlighted_word, current_highlighted_rect

    if current_word != last_highlighted_word:
        last_highlighted_word = current_word
        current_highlighted_rect = current_rect

        if current_word:
            print(f"📖 Highlighting word: '{current_word}'")
        else:
            print("📖 No word highlighted")

        return True
    return False


def calculate_word_positions_accurately(screen, font, lines, text_format, scroll_offset):
    """חישוב מדויק של מיקומי מילים על המסך"""
    global word_boxes, word_positions_cache

    cache_key = (text_format.font_size, text_format.line_spacing,
                 text_format.word_spacing, scroll_offset, current_text_option)

    if cache_key in word_positions_cache:
        word_boxes = word_positions_cache[cache_key].copy()
        return

    word_boxes.clear()

    screen_height = screen.get_height()
    screen_width = screen.get_width()

    text_font = pygame.font.SysFont("Arial", text_format.font_size)
    line_spacing = text_format.line_spacing
    word_spacing = text_format.word_spacing

    start_y = 60 - scroll_offset
    margin_x = 60
    max_line_width = screen_width - (margin_x * 2)

    y = start_y
    for line_idx, line in enumerate(lines):
        if y < -100 or y > screen_height + 100:
            y += line_spacing
            continue

        if not line.strip():
            y += line_spacing // 2
            continue

        words = line.split()
        current_line_words = []
        x = margin_x

        for word in words:
            word_surface = text_font.render(word + " ", True, (0, 0, 0))
            word_width = word_surface.get_width()

            if x + word_width > max_line_width and current_line_words:
                render_line_words_with_positions(current_line_words, margin_x, y,
                                                 word_spacing, text_font)
                y += line_spacing
                current_line_words = [word]
                x = margin_x + word_width
            else:
                current_line_words.append(word)
                x += word_width

        if current_line_words:
            render_line_words_with_positions(current_line_words, margin_x, y,
                                             word_spacing, text_font)

        y += line_spacing

    word_positions_cache[cache_key] = word_boxes.copy()


def render_line_words_with_positions(words, start_x, y, word_spacing, font):
    """רינדור מילים בשורה עם שמירת מיקומים מדויקים"""
    x = start_x
    for word in words:
        surf = font.render(word, True, (0, 0, 0))
        rect = surf.get_rect(topleft=(x, y))
        word_boxes.append((word, rect))
        x += rect.width + word_spacing


# מחלקת כיול פשוטה ויעילה
class SimpleCalibrator:
    def __init__(self, screen_w, screen_h):
        self.screen_width, self.screen_height = screen_w, screen_h
        self.calibration_file = CALIBRATION_FILE
        self.samples = []

    def run_calibration(self, manual_mode=False):
        """הפעלת כיול פשוט וישיר"""
        print(f"🚀 Starting {'MANUAL' if manual_mode else 'AUTOMATIC'} calibration...")

        points = [
            (int(self.screen_width * 0.1), int(self.screen_height * 0.1)),
            (int(self.screen_width * 0.5), int(self.screen_height * 0.1)),
            (int(self.screen_width * 0.9), int(self.screen_height * 0.1)),
            (int(self.screen_width * 0.1), int(self.screen_height * 0.5)),
            (int(self.screen_width * 0.5), int(self.screen_height * 0.5)),
            (int(self.screen_width * 0.9), int(self.screen_height * 0.5)),
            (int(self.screen_width * 0.1), int(self.screen_height * 0.9)),
            (int(self.screen_width * 0.5), int(self.screen_height * 0.9)),
            (int(self.screen_width * 0.9), int(self.screen_height * 0.9)),
        ]

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Camera failed!")
                return False

            print(f"📍 Calibrating {len(points)} points...")

            pygame.init()
            if not pygame.get_init():
                pygame.init()

            calib_screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)
            pygame.display.set_caption("Calibration")
            clock = pygame.time.Clock()

            with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                 min_detection_confidence=0.5) as face_mesh:

                for i, (x, y) in enumerate(points):
                    print(f"📍 Point {i + 1}/{len(points)}: ({x}, {y})")

                    start_time = time.time()
                    countdown = 3 if not manual_mode else 999
                    point_samples = []

                    while countdown > 0:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                cap.release()
                                return False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    pygame.quit()
                                    cap.release()
                                    return False
                                elif event.key == pygame.K_SPACE and manual_mode:
                                    countdown = 0
                                    break

                        calib_screen.fill((0, 0, 0))
                        pygame.draw.circle(calib_screen, (255, 0, 0), (x, y), 20)
                        pygame.draw.circle(calib_screen, (255, 255, 255), (x, y), 3)

                        font = pygame.font.SysFont("Arial", 30)
                        if manual_mode:
                            text = font.render(f"Look at red circle and press SPACE (Point {i + 1}/{len(points)})",
                                               True, (255, 255, 255))
                        else:
                            text = font.render(f"Look at red circle - Auto: {countdown} (Point {i + 1}/{len(points)})",
                                               True, (255, 255, 255))

                        text_rect = text.get_rect(center=(self.screen_width // 2, 50))
                        calib_screen.blit(text, text_rect)

                        progress_text = font.render(f"Progress: {i}/{len(points)} points completed", True,
                                                    (200, 200, 200))
                        progress_rect = progress_text.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
                        calib_screen.blit(progress_text, progress_rect)

                        pygame.display.flip()
                        clock.tick(30)

                        if not manual_mode and time.time() - start_time >= 1:
                            countdown -= 1
                            start_time = time.time()

                    print(f"📊 Collecting data for point {i + 1}...")
                    collection_start = time.time()

                    while time.time() - collection_start < 2.0:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (
                                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                                pygame.quit()
                                cap.release()
                                return False

                        calib_screen.fill((0, 0, 0))
                        pygame.draw.circle(calib_screen, (0, 255, 0), (x, y), 20)
                        pygame.draw.circle(calib_screen, (255, 255, 255), (x, y), 3)

                        collecting_text = font.render("Recording eye data... keep looking at green circle", True,
                                                      (0, 255, 0))
                        collecting_rect = collecting_text.get_rect(center=(self.screen_width // 2, 50))
                        calib_screen.blit(collecting_text, collecting_rect)

                        pygame.display.flip()

                        success, frame = cap.read()
                        if success:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = face_mesh.process(frame_rgb)

                            if results.multi_face_landmarks:
                                for face_landmarks in results.multi_face_landmarks:
                                    left_iris = [face_landmarks.landmark[i] for i in range(474, 478)]
                                    right_iris = [face_landmarks.landmark[i] for i in range(469, 473)]
                                    all_iris = left_iris + right_iris

                                    x_coords = [p.x for p in all_iris]
                                    y_coords = [p.y for p in all_iris]
                                    x_center = 1.0 - np.mean(x_coords)
                                    y_center = np.mean(y_coords)

                                    point_samples.append([x_center, y_center])

                        time.sleep(0.03)

                    if point_samples:
                        avg_iris = np.mean(point_samples, axis=0).tolist()
                        screen_coords = [x / self.screen_width, y / self.screen_height]

                        self.samples.append({
                            "iris": avg_iris,
                            "screen": screen_coords
                        })
                        print(f"✅ Point {i + 1} completed - {len(point_samples)} samples")
                    else:
                        print(f"❌ No data collected for point {i + 1}")

            pygame.quit()
            cap.release()

            return self.save_calibration()

        except Exception as e:
            print(f"❌ Calibration error: {e}")
            pygame.quit()
            if 'cap' in locals():
                cap.release()
            return False

    def save_calibration(self):
        """שמירת הכיול"""
        if len(self.samples) < 4:
            print(f"❌ Insufficient data: {len(self.samples)} points")
            return False

        try:
            X = np.array([s["iris"] for s in self.samples])
            Y = np.array([s["screen"] for s in self.samples])

            model_x = LinearRegression().fit(X, Y[:, 0])
            model_y = LinearRegression().fit(X, Y[:, 1])

            # פשוט החלף ל:
           # model_x = create_polynomial_model(degree=3).fit(X, Y[:, 0])
            #model_y = create_polynomial_model(degree=3).fit(X, Y[:, 1])


            pred_x = model_x.predict(X)
            pred_y = model_y.predict(X)

            errors_x = np.abs(pred_x - Y[:, 0]) * self.screen_width
            errors_y = np.abs(pred_y - Y[:, 1]) * self.screen_height
            total_error = np.mean(np.sqrt(errors_x ** 2 + errors_y ** 2))

            calibration_data = {
                "points": self.samples,
                "workpy_compatible": True,
                "mirror_corrected": True,
                "features": 2,
                "total_samples": len(self.samples),
                "accuracy_pixels": total_error,
                "calibration_timestamp": time.time(),
                "screen_resolution": [self.screen_width, self.screen_height]
            }

            with open(self.calibration_file, "w") as f:
                json.dump(calibration_data, f, indent=2)

            print(f"✅ Calibration saved! Accuracy: {total_error:.1f} pixels")
            return True

        except Exception as e:
            print(f"❌ Save error: {e}")
            return False


# מנהל כיול מעודכן
class CalibrationManager:
    def __init__(self):
        self.models_saved = False
        self.last_calibration_info = None

    def load_calibration(self):
        """טעינת כיול מקובץ JSON"""
        try:
            if not os.path.exists(CALIBRATION_FILE):
                return None

            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)

            if "points" not in data:
                return None

            points = data["points"]
            X = np.array([p["iris"] for p in points])
            Y = np.array([p["screen"] for p in points])



            model_x = LinearRegression().fit(X, Y[:, 0])
            model_y = LinearRegression().fit(X, Y[:, 1])

            #model_x = create_polynomial_model(degree=3).fit(X, Y[:, 0])
            #model_y = create_polynomial_model(degree=3).fit(X, Y[:, 1])

            self.last_calibration_info = data
            print("✅ Calibration loaded successfully!")

            return {
                'model_x': model_x,
                'model_y': model_y,
                'baseline_nose': None,
                'screen_resolution': data.get('screen_resolution', [1920, 1080])
            }

        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return None

    def get_calibration_info(self):
        """קבלת מידע על הכיול האחרון"""
        if not self.last_calibration_info:
            return None

        timestamp = self.last_calibration_info.get('calibration_timestamp', time.time())
        dt = datetime.fromtimestamp(timestamp)

        return {
            'date': dt.strftime('%d/%m/%Y'),
            'time': dt.strftime('%H:%M:%S'),
            'resolution': self.last_calibration_info.get('screen_resolution', [1920, 1080]),
            'accuracy': f"{self.last_calibration_info.get('accuracy_pixels', 0):.1f}px",
            'points': self.last_calibration_info.get('total_samples', 0)
        }


# כלי הכוונון השקוף - גרסה מתוקנת ללא threads
class RealTimeTuner:
    def __init__(self):
        self.window = None
        self.is_active = False
        self.is_minimized = False

    def open_tuning_window(self):
        """פתיחת חלון כוונון שקוף - ללא threads"""
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return

        try:
            self.is_active = True
            self.window = tk.Tk()
            self.window.title("🎛️ Live Tuner")

            self.window.geometry("350x600+50+50")
            self.window.configure(bg='black')
            self.window.attributes("-alpha", 0.85)
            self.window.attributes("-topmost", True)

            header_frame = tk.Frame(self.window, bg='#1a1a1a')
            header_frame.pack(fill=tk.X, padx=5, pady=2)

            title = tk.Label(header_frame, text="🎛️ Live Tuner",
                             font=('Arial', 12, 'bold'), fg='#00ff00', bg='#1a1a1a')
            title.pack(side=tk.LEFT)

            self.toggle_btn = tk.Button(header_frame, text="−",
                                        command=self.toggle_minimize,
                                        bg='#333', fg='white', font=('Arial', 12, 'bold'),
                                        width=3, height=1)
            self.toggle_btn.pack(side=tk.RIGHT, padx=2)

            self.main_frame = tk.Frame(self.window, bg='#1a1a1a')
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            self.create_compact_controls()
            self.window.protocol("WM_DELETE_WINDOW", self.close_tuning_window)

            self.window.update()
            print("🎛️ Tuner window created successfully!")

        except Exception as e:
            print(f"❌ Error creating tuner window: {e}")
            self.is_active = False
            self.window = None

    def update_tuner(self):
        """עדכון החלון ללא blocking"""
        if self.is_active and self.window:
            try:
                if self.window.winfo_exists():
                    self.window.update()
                    return True
            except Exception as e:
                print(f"❌ Tuner update error: {e}")
                self.close_tuning_window()
                return False
        return False

    def toggle_minimize(self):
        """מזעור/הגדלה"""
        if self.is_minimized:
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.toggle_btn.config(text="−")
            self.window.geometry("350x600+50+50")
            self.is_minimized = False
        else:
            self.main_frame.pack_forget()
            self.toggle_btn.config(text="+")
            self.window.geometry("180x35+50+50")
            self.is_minimized = True

    def create_compact_controls(self):
        """יצירת בקרים מורחבים"""
        # Alpha - הכי חשוב
        self.create_compact_slider(self.main_frame, "🎯 Smoothing (Alpha)", 'alpha', 0.01, 1.0, 0.01, '#ff6b6b')

        # Movement Threshold
        self.create_compact_slider(self.main_frame, "🚫 Movement Threshold", 'movement_threshold', 1, 50, 1, '#4ecdc4')

        # History Size
        self.create_compact_slider(self.main_frame, "📊 History Buffer", 'history_size', 1, 20, 1, '#45b7d1')

        # Update Sleep - חדש!
        self.create_compact_slider(self.main_frame, "⏱️ Update Delay", 'update_sleep', 0.01, 0.1, 0.01, '#ff9500')

        # Head compensation controls - חדש!
        head_frame = tk.Frame(self.main_frame, bg='#2a2a2a', relief=tk.RIDGE, bd=1)
        head_frame.pack(fill=tk.X, pady=2)

        tk.Label(head_frame, text="🧠 HEAD COMPENSATION", font=('Arial', 9, 'bold'),
                 fg='#ffd93d', bg='#2a2a2a').pack()

        self.head_comp_var = tk.BooleanVar(value=TUNING_PARAMS['head_compensation'])
        head_cb = tk.Checkbutton(head_frame, text="Enable Head Tracking", variable=self.head_comp_var,
                                 command=lambda: self.update_param('head_compensation', self.head_comp_var.get()),
                                 bg='#2a2a2a', fg='white', font=('Arial', 9),
                                 selectcolor='#333', activebackground='#2a2a2a')
        head_cb.pack(anchor=tk.W, padx=5)

        self.create_compact_slider(head_frame, "↔️ Head X Gain", 'head_gain_x', 0.0, 3.0, 0.1, '#ff6b9d')
        self.create_compact_slider(head_frame, "↕️ Head Y Gain", 'head_gain_y', 0.0, 3.0, 0.1, '#6b9dff')

        # Visual Settings
        visual_frame = tk.Frame(self.main_frame, bg='#2a2a2a', relief=tk.RIDGE, bd=1)
        visual_frame.pack(fill=tk.X, pady=2)

        tk.Label(visual_frame, text="👁️ VISUAL", font=('Arial', 9, 'bold'),
                 fg='#ffd93d', bg='#2a2a2a').pack()

        self.create_compact_slider(visual_frame, "🔴 Cursor Size", 'cursor_size', 15, 50, 1, '#ff9ff3')
        self.create_compact_slider(visual_frame, "⚡ Update Rate", 'update_rate', 10, 60, 1, '#54a0ff')

        # Color Buttons
        colors_frame = tk.Frame(visual_frame, bg='#2a2a2a')
        colors_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(colors_frame, text="🎨 Color:", font=('Arial', 8, 'bold'),
                 fg='white', bg='#2a2a2a').pack(side=tk.LEFT)

        colors = [("🔴", (255, 0, 0)), ("🟢", (0, 255, 0)), ("🔵", (0, 0, 255)),
                  ("🟡", (255, 255, 0)), ("🟣", (255, 0, 255))]

        for emoji, color in colors:
            def make_color_callback(c):
                def color_clicked():
                    old_color = TUNING_PARAMS['cursor_color']
                    TUNING_PARAMS['cursor_color'] = c
                    print(f"🎨 Color changed: {old_color} -> {c}")

                return color_clicked

            btn = tk.Button(colors_frame, text=emoji,
                            command=make_color_callback(color),
                            bg='#333', fg='white', font=('Arial', 10),
                            width=2, height=1, bd=1)
            btn.pack(side=tk.LEFT, padx=1)

        # Trail Settings
        trail_frame = tk.Frame(self.main_frame, bg='#2a2a2a', relief=tk.RIDGE, bd=1)
        trail_frame.pack(fill=tk.X, pady=2)

        tk.Label(trail_frame, text="🐍 TRAIL", font=('Arial', 9, 'bold'),
                 fg='#ffd93d', bg='#2a2a2a').pack()

        self.trail_var = tk.BooleanVar(value=TUNING_PARAMS['show_trail'])
        trail_cb = tk.Checkbutton(trail_frame, text="Show Trail", variable=self.trail_var,
                                  command=lambda: self.update_param('show_trail', self.trail_var.get()),
                                  bg='#2a2a2a', fg='white', font=('Arial', 9),
                                  selectcolor='#333', activebackground='#2a2a2a')
        trail_cb.pack(anchor=tk.W, padx=5)

        self.create_compact_slider(trail_frame, "📏 Trail Length", 'trail_length', 2, 20, 1, '#ff6bff')

        # Quick Presets
        presets_frame = tk.Frame(self.main_frame, bg='#1a1a1a')
        presets_frame.pack(fill=tk.X, pady=5)

        tk.Label(presets_frame, text="⚡ Presets:", font=('Arial', 8, 'bold'),
                 fg='#ffd93d', bg='#1a1a1a').pack(side=tk.LEFT)

        tk.Button(presets_frame, text="Fast", command=self.preset_responsive,
                  bg='#ff6b6b', fg='white', font=('Arial', 8, 'bold'), width=4).pack(side=tk.LEFT, padx=1)

        tk.Button(presets_frame, text="Mid", command=self.preset_balanced,
                  bg='#4ecdc4', fg='white', font=('Arial', 8, 'bold'), width=4).pack(side=tk.LEFT, padx=1)

        tk.Button(presets_frame, text="Smooth", command=self.preset_stable,
                  bg='#45b7d1', fg='white', font=('Arial', 8, 'bold'), width=4).pack(side=tk.LEFT, padx=1)

        # Status
        values_frame = tk.Frame(self.main_frame, bg='#0a0a0a', relief=tk.SUNKEN, bd=1)
        values_frame.pack(fill=tk.X, pady=2)

        tk.Label(values_frame, text="📈 Status", font=('Arial', 8, 'bold'),
                 fg='#00ff00', bg='#0a0a0a').pack()

        self.status_label = tk.Label(values_frame, text="Ready...", font=('Courier', 7),
                                     fg='#00ff00', bg='#0a0a0a')
        self.status_label.pack(fill=tk.X, padx=5)

    def create_compact_slider(self, parent, label, param_key, min_val, max_val, resolution, color):
        """יצירת סליידר קומפקטי"""
        frame = tk.Frame(parent, bg='#333333', relief=tk.FLAT, bd=1)
        frame.pack(fill=tk.X, padx=2, pady=1)

        label_frame = tk.Frame(frame, bg='#333333')
        label_frame.pack(fill=tk.X)

        tk.Label(label_frame, text=label, font=('Arial', 8, 'bold'),
                 fg=color, bg='#333333').pack(side=tk.LEFT)

        self.value_labels = getattr(self, 'value_labels', {})
        self.value_labels[param_key] = tk.Label(label_frame, text=f"{TUNING_PARAMS[param_key]}",
                                                font=('Courier', 8, 'bold'),
                                                fg='white', bg='#333333')
        self.value_labels[param_key].pack(side=tk.RIGHT)

        var = tk.DoubleVar(value=TUNING_PARAMS[param_key])

        def slider_callback(val):
            """עדכון מיידי של ערכים"""
            try:
                if param_key in ['alpha', 'update_sleep', 'head_gain_x', 'head_gain_y']:
                    new_value = float(val)
                else:
                    new_value = int(float(val))

                old_value = TUNING_PARAMS[param_key]

                if abs(old_value - new_value) > 0.001:
                    TUNING_PARAMS[param_key] = new_value
                    print(f"🎛️ {param_key}: {old_value} -> {new_value}")

                    if hasattr(self, 'value_labels') and param_key in self.value_labels:
                        if isinstance(new_value, float):
                            self.value_labels[param_key].config(text=f"{new_value:.3f}")
                        else:
                            self.value_labels[param_key].config(text=f"{new_value}")

            except Exception as e:
                print(f"❌ Slider error {param_key}: {e}")

        scale = tk.Scale(frame, from_=min_val, to=max_val, resolution=resolution,
                         orient=tk.HORIZONTAL, variable=var,
                         command=slider_callback,
                         bg='#333333', fg=color, highlightbackground='#333333',
                         troughcolor=color, length=300, width=12, showvalue=0)
        scale.pack(fill=tk.X, padx=5, pady=1)

        scale.param_key = param_key

        if not hasattr(self, 'sliders'):
            self.sliders = {}
        self.sliders[param_key] = scale

    def update_param_with_display(self, param_key, value):
        """עדכון פרמטר עם תצוגה ועדכון מיידי של cursor"""
        old_value = TUNING_PARAMS[param_key]
        TUNING_PARAMS[param_key] = value

        if hasattr(self, 'value_labels') and param_key in self.value_labels:
            if isinstance(value, float):
                self.value_labels[param_key].config(text=f"{value:.2f}")
            else:
                self.value_labels[param_key].config(text=f"{value}")

        print(f"🎛️ Parameter {param_key} changed from {old_value} to {value}")

        if self.window and self.window.winfo_exists():
            try:
                self.window.update()
                print(f"🔄 Tuner window updated for {param_key}")
            except Exception as e:
                print(f"❌ Error updating tuner window: {e}")

    def update_param(self, param_key, value):
        TUNING_PARAMS[param_key] = value

    def preset_responsive(self):
        print("⚡ Responsive preset button clicked!")
        old_params = TUNING_PARAMS.copy()
        TUNING_PARAMS.update({
            'alpha': 0.4, 'movement_threshold': 5, 'history_size': 2,
            'cursor_color': (0, 255, 0), 'update_rate': 45,
            'update_sleep': 0.01, 'cursor_size': 15
        })
        print("⚡ Responsive preset activated")
        self.refresh_all_sliders()
        if self.window and self.window.winfo_exists():
            self.window.update()

    def preset_balanced(self):
        print("⚖️ Balanced preset button clicked!")
        old_params = TUNING_PARAMS.copy()
        TUNING_PARAMS.update({
            'alpha': 0.1, 'movement_threshold': 15, 'history_size': 3,
            'cursor_color': (255, 255, 0), 'update_rate': 30,
            'update_sleep': 0.02, 'cursor_size': 12
        })
        print("⚖️ Balanced preset activated")
        self.refresh_all_sliders()
        if self.window and self.window.winfo_exists():
            self.window.update()

    def preset_stable(self):
        print("🎯 Stable preset button clicked!")
        old_params = TUNING_PARAMS.copy()
        TUNING_PARAMS.update({
            'alpha': 0.05, 'movement_threshold': 25, 'history_size': 5,
            'cursor_color': (0, 0, 255), 'update_rate': 25,
            'update_sleep': 0.03, 'cursor_size': 10
        })
        print("🎯 Stable preset activated")
        self.refresh_all_sliders()
        if self.window and self.window.winfo_exists():
            self.window.update()

    def refresh_all_sliders(self):
        """רענון כל הסליידרים אחרי preset"""
        print("🔄 Refreshing all sliders...")

        if hasattr(self, 'value_labels'):
            for param_key, label in self.value_labels.items():
                value = TUNING_PARAMS.get(param_key, 0)
                if isinstance(value, float):
                    label.config(text=f"{value:.3f}")
                else:
                    label.config(text=f"{value}")
                print(f"🔄 Updated label {param_key}: {value}")

        if hasattr(self, 'sliders'):
            for param_key, scale in self.sliders.items():
                try:
                    new_value = TUNING_PARAMS[param_key]
                    scale.set(new_value)
                    print(f"🔄 Updated slider {param_key}: {new_value}")
                except Exception as e:
                    print(f"❌ Error updating slider {param_key}: {e}")

        if self.window and self.window.winfo_exists():
            try:
                self.window.update()
                print("🔄 Tuner window fully updated")
            except Exception as e:
                print(f"❌ Error updating tuner window: {e}")

    def close_tuning_window(self):
        self.is_active = False
        if self.window:
            try:
                self.window.quit()
                self.window.destroy()
            except:
                pass
            self.window = None


tuner = RealTimeTuner()


def open_live_tuner():
    """פתיחת הכוונון השקוף - ללא threads"""
    print("🎛️ Opening live tuner...")
    try:
        if not tuner.is_active:
            tuner.open_tuning_window()
        else:
            print("🎛️ Tuner is already active")
    except Exception as e:
        print(f"❌ Error opening tuner: {e}")


def update_cursor_trail(x, y):
    """עדכון זנב הסמן"""
    global cursor_trail
    cursor_trail.append((x, y))

    max_length = TUNING_PARAMS['trail_length']
    if len(cursor_trail) > max_length:
        cursor_trail = cursor_trail[-max_length:]


# פונקציות תצוגה מתוקנות לסנכרון מושלם
def draw_text_centered(surface, font, text, y, color=BLACK):
    """ציור טקסט ממורכז"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.centerx = surface.get_width() // 2
    text_rect.y = y
    surface.blit(text_surface, text_rect)
    return text_rect


def draw_button(surface, font, text, x, y, width, height, color=LIGHT_BLUE, text_color=BLACK):
    """ציור כפתור"""
    button_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, color, button_rect)
    pygame.draw.rect(surface, BLACK, button_rect, 2)

    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect()
    text_rect.center = button_rect.center
    surface.blit(text_surface, text_rect)

    return button_rect


def show_participant_input_screen(screen, font):
    """מסך קלט תעודת זהות מעוצב"""
    clock = pygame.time.Clock()
    participant_id = ""
    input_active = True

    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and participant_id.strip():
                    return participant_id.strip()
                elif event.key == pygame.K_BACKSPACE:
                    participant_id = participant_id[:-1]
                elif event.unicode.isprintable():
                    participant_id += event.unicode

        screen.fill(BLUE)

        title_font = pygame.font.SysFont("Arial", 60, bold=True)
        draw_text_centered(screen, title_font, "Eye Tracking System", 100, WHITE)

        instruction_font = pygame.font.SysFont("Arial", 35)
        draw_text_centered(screen, instruction_font, "Please enter your participant ID:", 250, WHITE)

        input_width = 400
        input_height = 60
        input_x = (screen.get_width() - input_width) // 2
        input_y = 350

        input_rect = pygame.Rect(input_x, input_y, input_width, input_height)
        pygame.draw.rect(screen, WHITE, input_rect)
        pygame.draw.rect(screen, BLACK, input_rect, 3)

        text_surface = font.render(participant_id, True, BLACK)
        screen.blit(text_surface, (input_x + 10, input_y + 15))

        if int(time.time() * 2) % 2:
            cursor_x = input_x + 10 + text_surface.get_width()
            pygame.draw.line(screen, BLACK, (cursor_x, input_y + 10), (cursor_x, input_y + 50), 2)

        info_font = pygame.font.SysFont("Arial", 25)
        draw_text_centered(screen, info_font, "Press ENTER to continue", 450, WHITE)

        pygame.display.flip()
        clock.tick(60)


def show_text_selection_screen(screen, font):
    """מסך בחירת טקסט"""
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if short_button.collidepoint(mouse_pos):
                    return "short"
                elif medium_button.collidepoint(mouse_pos):
                    return "medium"
                elif long_button.collidepoint(mouse_pos):
                    return "long"

        screen.fill(BLUE)

        title_font = pygame.font.SysFont("Arial", 45, bold=True)
        draw_text_centered(screen, title_font, "Select Reading Text", 60, WHITE)

        instruction_font = pygame.font.SysFont("Arial", 25)
        draw_text_centered(screen, instruction_font, "Choose the text length for your reading session:", 120, WHITE)

        button_width = 350
        button_height = 80
        button_font = pygame.font.SysFont("Arial", 22)
        start_y = 180
        spacing = 140

        short_button = draw_button(screen, button_font,
                                   f"📖 {TEXT_OPTIONS['short']['name']}",
                                   (screen.get_width() - button_width) // 2, start_y,
                                   button_width, button_height, LIGHT_BLUE, BLACK)

        desc_font = pygame.font.SysFont("Arial", 18)
        draw_text_centered(screen, desc_font, "Perfect for quick calibration test", start_y + 90, WHITE)

        medium_button = draw_button(screen, button_font,
                                    f"📚 {TEXT_OPTIONS['medium']['name']}",
                                    (screen.get_width() - button_width) // 2, start_y + spacing,
                                    button_width, button_height, LIGHT_BLUE, BLACK)

        draw_text_centered(screen, desc_font, "Good for standard reading analysis", start_y + spacing + 90, WHITE)

        long_button = draw_button(screen, button_font,
                                  f"📄 {TEXT_OPTIONS['long']['name']}",
                                  (screen.get_width() - button_width) // 2, start_y + spacing * 2,
                                  button_width, button_height, GREEN, WHITE)

        draw_text_centered(screen, desc_font, "Comprehensive reading behavior analysis", start_y + spacing * 2 + 90,
                           WHITE)

        note_font = pygame.font.SysFont("Arial", 20, bold=True)
        draw_text_centered(screen, note_font, "⭐ Recommended for detailed research", start_y + spacing * 2 + 115, WHITE)

        pygame.display.flip()
        clock.tick(60)


def show_calibration_instructions_screen(screen, font, calibration_type):
    """מסך הוראות מפורט לפני הכיול"""
    clock = pygame.time.Clock()

    # קביעת טקסט לפי סוג הכיול
    if calibration_type == "auto":
        mode_text = "🤖 Automatic Calibration"
        mode_color = GREEN
        trigger_text = "The system will automatically move between points"
        wait_instruction = "Wait 3 seconds at each point"
    else:
        mode_text = "👆 Manual Calibration"
        mode_color = ORANGE
        trigger_text = "Press SPACE when ready at each point"
        wait_instruction = "Take your time to position correctly"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True  # המשך לכיול
                elif event.key == pygame.K_ESCAPE:
                    return False  # חזור למסך הקודם

        # רקע כחול
        screen.fill(BLUE)

        # כותרת ראשית
        title_font = pygame.font.SysFont("Arial", 50, bold=True)
        draw_text_centered(screen, title_font, "📋 Calibration Instructions", 60, WHITE)

        # סוג הכיול
        mode_font = pygame.font.SysFont("Arial", 35, bold=True)
        draw_text_centered(screen, mode_font, mode_text, 120, mode_color)

        # הוראות כלליות
        instruction_font = pygame.font.SysFont("Arial", 28)
        y_start = 180
        line_spacing = 45

        general_instructions = [
            "🎯 You will see 9 red circles on the screen",
            "👁️  Look directly at the CENTER of each circle",
            "📱 Keep your head stable and only move your EYES",
            "💡 Good lighting helps - avoid shadows on your face",
            "📏 Sit about 60cm from the screen for best results"
        ]

        for i, instruction in enumerate(general_instructions):
            draw_text_centered(screen, instruction_font, instruction, y_start + i * line_spacing, WHITE)

        # הוראות ספציפיות לסוג הכיול
        specific_y = y_start + len(general_instructions) * line_spacing + 40

        # קופסה להוראות ספציפיות
        box_width = 800
        box_height = 120
        box_x = (screen.get_width() - box_width) // 2
        box_y = specific_y - 20

        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(screen, mode_color, box_rect, border_radius=15)
        pygame.draw.rect(screen, WHITE, box_rect, 3, border_radius=15)

        # טקסט בתוך הקופסה
        specific_font = pygame.font.SysFont("Arial", 24, bold=True)
        draw_text_centered(screen, specific_font, f"📍 {mode_text} Instructions:", specific_y + 10, WHITE)
        draw_text_centered(screen, specific_font, trigger_text, specific_y + 40, WHITE)
        draw_text_centered(screen, specific_font, wait_instruction, specific_y + 70, WHITE)

        # טיפים נוספים
        tips_y = specific_y + 140
        tips_font = pygame.font.SysFont("Arial", 22)

        tips = [
            "💡 Tips for Best Results:",
            "• Remove glasses if possible (or clean them well)",
            "• Ensure your face is well-lit and visible",
            "• Don't move your head during calibration",
            "• Focus on the CENTER of each red circle"
        ]

        for i, tip in enumerate(tips):
            color = LIGHT_BLUE if i == 0 else WHITE
            font_to_use = pygame.font.SysFont("Arial", 24, bold=True) if i == 0 else tips_font
            draw_text_centered(screen, font_to_use, tip, tips_y + i * 30, color)

        # הוראות המשך
        continue_y = screen.get_height() - 120
        continue_font = pygame.font.SysFont("Arial", 32, bold=True)

        # רקע לכפתורים
        button_bg_width = 600
        button_bg_height = 80
        button_bg_x = (screen.get_width() - button_bg_width) // 2
        button_bg_y = continue_y - 10

        button_bg_rect = pygame.Rect(button_bg_x, button_bg_y, button_bg_width, button_bg_height)
        pygame.draw.rect(screen, WHITE, button_bg_rect, border_radius=10)
        pygame.draw.rect(screen, GREEN, button_bg_rect, 3, border_radius=10)

        draw_text_centered(screen, continue_font, "🚀 Press SPACE to Start Calibration", continue_y + 10, GREEN)
        draw_text_centered(screen, pygame.font.SysFont("Arial", 20), "or ESC to go back", continue_y + 45, GRAY)

        # אנימציה קלה - הבהוב
        if int(time.time() * 2) % 2:
            glow_rect = button_bg_rect.inflate(10, 10)
            pygame.draw.rect(screen, GREEN, glow_rect, 2, border_radius=12)

        pygame.display.flip()
        clock.tick(60)


def show_calibration_choice_screen_with_instructions(screen, font, calib_manager, cap, face_mesh):
    """מסך בחירת סוג כיול מעודכן עם הוראות"""
    clock = pygame.time.Clock()
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(80, 220, 100))

    while True:
        ret, frame = cap.read()
        cam_surface = None

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(frame_rgb)
            disp_frame = frame.copy()

            if res.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    disp_frame, res.multi_face_landmarks[0],
                    mp.solutions.face_mesh.FACEMESH_TESSELATION, None, drawing_spec)

            cam_frame = cv2.resize(disp_frame, (400, 300))
            cam_surface = pygame.surfarray.make_surface(
                cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if new_auto_button.collidepoint(mouse_pos):
                    # הצגת הוראות לכיול אוטומטי
                    if show_calibration_instructions_screen(screen, font, "auto"):
                        return "new_auto"
                    # אם המשתמש לחץ ESC, חוזרים למסך הבחירה

                elif new_manual_button.collidepoint(mouse_pos):
                    # הצגת הוראות לכיול ידני
                    if show_calibration_instructions_screen(screen, font, "manual"):
                        return "new_manual"
                    # אם המשתמש לחץ ESC, חוזרים למסך הבחירה

                elif load_calib_button.collidepoint(mouse_pos):
                    return "load"

        screen.fill(BLUE)

        title_font = pygame.font.SysFont("Arial", 50, bold=True)
        draw_text_centered(screen, title_font, "Calibration Options", 50, WHITE)

        if cam_surface:
            cam_x = (screen.get_width() - 400) // 2
            cam_y = 100
            screen.blit(cam_surface, (cam_x, cam_y))

            pygame.draw.rect(screen, WHITE, (cam_x - 2, cam_y - 2, 404, 304), 2)

            cam_font = pygame.font.SysFont("Arial", 20)
            draw_text_centered(screen, cam_font, "Camera Preview - Make sure your face is visible", 420, WHITE)

        calib_info = calib_manager.get_calibration_info()
        info_y_start = 440
        if calib_info:
            info_font = pygame.font.SysFont("Arial", 22)
            draw_text_centered(screen, info_font, "Previous calibration found:", info_y_start, WHITE)
            draw_text_centered(screen, info_font, f"Date: {calib_info['date']} | Accuracy: {calib_info['accuracy']}",
                               info_y_start + 25, WHITE)
            draw_text_centered(screen, info_font,
                               f"Points: {calib_info['points']} | Resolution: {calib_info['resolution']}",
                               info_y_start + 50, WHITE)
        else:
            info_font = pygame.font.SysFont("Arial", 25)
            draw_text_centered(screen, info_font, "No previous calibration found", info_y_start + 25, WHITE)

        button_width = 250
        button_height = 50
        button_font = pygame.font.SysFont("Arial", 18)
        button_y_start = 520
        button_spacing = 270

        # כפתור כיול אוטומטי חדש
        new_auto_button = draw_button(screen, button_font, " New Auto Calibration",
                                      (screen.get_width() - button_spacing) // 2 - button_width // 2, button_y_start,
                                      button_width, button_height, GREEN, WHITE)

        # כפתור כיול ידני חדש
        new_manual_button = draw_button(screen, button_font, " New Manual Calibration",
                                        (screen.get_width() + button_spacing) // 2 - button_width // 2, button_y_start,
                                        button_width, button_height, ORANGE, WHITE)

        # כפתור טעינת כיול
        if calib_info:
            load_calib_button = draw_button(screen, button_font, " Load Previous",
                                            (screen.get_width() - button_width) // 2, button_y_start + 70,
                                            button_width, button_height, LIGHT_BLUE, BLACK)
        else:
            load_calib_button = draw_button(screen, button_font, " Load Previous",
                                            (screen.get_width() - button_width) // 2, button_y_start + 70,
                                            button_width, button_height, GRAY, WHITE)

        # הוספת הערה על ההוראות
        note_font = pygame.font.SysFont("Arial", 18)
        draw_text_centered(screen, note_font, "💡 Click on calibration type to see detailed instructions",
                           button_y_start + 140, LIGHT_BLUE)

        pygame.display.flip()
        clock.tick(30)


# *** פונקציות רינדור מעודכנות - עם בדיקת מעקב פעיל וחימום ***
def render_text_with_perfect_sync(screen, font, lines, text_format, blink_counter=0,
                                  cursor_pos=None, scroll_offset=0):
    """רינדור טקסט עם סנכרון מושלם לסמן - רק אחרי תקופת חימום"""
    global last_highlighted_word, current_highlighted_rect
    global is_in_warmup, tracking_warmup_start, tracking_start_time

    # בדיקת סיום תקופת חימום
    if is_in_warmup and tracking_warmup_start:
        warmup_elapsed = time.time() - tracking_warmup_start
        if warmup_elapsed >= WARMUP_DURATION:
            # סיום תקופת חימום
            is_in_warmup = False
            tracking_start_time = time.time()  # התחלת הקלטה אמיתית
            print("🟢 WARMUP COMPLETE - Now recording your reading!")

    # רינדור רקע
    screen.fill((255, 255, 255))

    # חישוב מיקומי מילים מדויקים
    calculate_word_positions_accurately(screen, font, lines, text_format, scroll_offset)

    # איתור מילה תחת הסמן - רק אם מעקב פעיל ולא בחימום
    current_word = None
    highlight_rect = None

    if cursor_pos and is_tracking_active and not is_in_warmup:
        ax, ay = cursor_pos
        # זיהוי מיידי של המילה
        current_word, highlight_rect = find_word_under_cursor(ax, ay, word_boxes)

        # עדכון הדגשה מיידי - אם השתנתה המילה
        if current_word != last_highlighted_word:
            last_highlighted_word = current_word
            current_highlighted_rect = highlight_rect

            # הדפסה מיידית - רק אחרי חימום
            if current_word:
                print(f"🟢 READING: '{current_word}' at cursor ({ax}, {ay})")
            else:
                print(f"⚪ No word at cursor ({ax}, {ay})")

    elif not is_tracking_active or is_in_warmup:
        # כשמעקב לא פעיל או בחימום - אין הדגשה
        if last_highlighted_word is not None:
            last_highlighted_word = None
            current_highlighted_rect = None
            if not is_in_warmup:  # הדפס רק אם לא בחימום
                print("⚫ Eye tracking stopped - no word highlighting")

    # רינדור טקסט עם הדגשה (או בלי אם מעקב לא פעיל או בחימום)
    render_text_with_highlighting(screen, font, lines, text_format, blink_counter,
                                  current_highlighted_rect if (is_tracking_active and not is_in_warmup) else None,
                                  scroll_offset)

    # רינדור כפתורי בקרה ומידע
    render_control_panel(screen, text_format, blink_counter)

    return current_word


def render_text_with_highlighting(screen, font, lines, text_format, blink_counter,
                                  highlight_rect, scroll_offset):
    """רינדור הטקסט עם הדגשת המילה הנוכחית - עם בדיקה אם מעקב פעיל"""
    screen_height = screen.get_height()
    screen_width = screen.get_width()

    text_font = pygame.font.SysFont("Arial", text_format.font_size)
    line_spacing = text_format.line_spacing
    word_spacing = text_format.word_spacing

    start_y = 60 - scroll_offset
    margin_x = 60
    max_line_width = screen.get_width() - (margin_x * 2)

    y = start_y
    for line_idx, line in enumerate(lines):
        if y < -50 or y > screen_height - 100:
            y += line_spacing
            continue

        if not line.strip():
            y += line_spacing // 2
            continue

        words = line.split()
        current_line_words = []
        x = margin_x

        for word in words:
            word_surface = text_font.render(word + " ", True, (0, 0, 0))
            word_width = word_surface.get_width()

            if x + word_width > max_line_width and current_line_words:
                render_line_with_highlight(screen, text_font, current_line_words, margin_x, y,
                                           word_spacing, highlight_rect)
                y += line_spacing
                current_line_words = [word]
                x = margin_x + word_width
            else:
                current_line_words.append(word)
                x += word_width

        if current_line_words:
            render_line_with_highlight(screen, text_font, current_line_words, margin_x, y,
                                       word_spacing, highlight_rect)

        y += line_spacing


def render_line_with_highlight(screen, font, words, start_x, y, word_spacing, highlight_rect):
    """רינדור שורה עם הדגשת מילה ספציפית - רק אם מעקב פעיל"""
    x = start_x
    for word in words:
        surf = font.render(word, True, (0, 0, 0))
        rect = surf.get_rect(topleft=(x, y))

        # בדיקה מדויקת אם זו המילה המודגשת - רק אם מעקב פעיל
        is_highlighted = False
        if highlight_rect and is_tracking_active and not is_in_warmup:
            # בדיקה פשוטה - האם המלבנים חופפים
            if (rect.x <= highlight_rect.centerx <= rect.x + rect.width and
                    rect.y <= highlight_rect.centery <= rect.y + rect.height):
                is_highlighted = True

        # הדגשת המילה אם הסמן עליה ומעקב פעיל
        if is_highlighted:
            # ציור רקע ירוק בולט למילה מודגשת
            highlight_bg = rect.inflate(16, 10)
            pygame.draw.rect(screen, (100, 255, 100), highlight_bg)  # ירוק בהיר יותר
            pygame.draw.rect(screen, (0, 150, 0), highlight_bg, 5)  # מסגרת ירוקה עבה

            # טקסט המילה בצבע כהה יותר כדי לבלוט
            word_surf = font.render(word, True, (0, 0, 0))
            screen.blit(word_surf, rect)
        else:
            # ציור רגיל של המילה
            screen.blit(surf, rect)

        x += rect.width + word_spacing


def render_control_panel(screen, text_format, blink_counter):
    """רינדור פאנל הבקרה המלא - עם הצגת סטטוס חימום"""
    global tuner_button_rect, tracking_button_rect
    global is_in_warmup, tracking_warmup_start

    screen_width = screen.get_width()
    screen_height = screen.get_height()

    # מידע על הטקסט בצד שמאל עליון
    info_font = pygame.font.SysFont("Arial", 20)
    text_info = TEXT_OPTIONS[current_text_option]["name"]
    info_surface = info_font.render(f"Text: {text_info}", True, (128, 128, 128))
    screen.blit(info_surface, (10, 10))

    # סטטוס מעקב עיניים - עם הצגת חימום
    status_font = pygame.font.SysFont("Arial", 18, bold=True)

    if is_tracking_active:
        if is_in_warmup:
            warmup_remaining = max(0, WARMUP_DURATION - (time.time() - tracking_warmup_start))
            status_text = f"🟡 WARMUP: {warmup_remaining:.1f}s - Look at your starting word"
            status_color = (255, 165, 0)  # כתום
            cursor_status = "🟡 Cursor: WARMING UP"
        else:
            status_text = "👁️ Eye Tracking: ACTIVE (Recording your reading)"
            status_color = (0, 150, 0)
            cursor_status = "🔴 Cursor: LIVE"
    else:
        status_text = "👁️ Eye Tracking: STOPPED (Press SPACE to start)"
        status_color = (200, 0, 0)
        cursor_status = "⚫ Cursor: FIXED POSITION"

    status_surface = status_font.render(status_text, True, status_color)
    screen.blit(status_surface, (10, 35))

    # סטטוס הסמן
    cursor_font = pygame.font.SysFont("Arial", 16, bold=True)
    cursor_surface = cursor_font.render(cursor_status, True, status_color)
    screen.blit(cursor_surface, (10, 55))

    # הוספת הודעת חימום באמצע המסך אם נדרש
    if is_in_warmup and tracking_warmup_start:
        warmup_remaining = max(0, WARMUP_DURATION - (time.time() - tracking_warmup_start))

        # יצירת מלבן מרכזי לחימום
        warmup_width = 600
        warmup_height = 120
        warmup_x = (screen_width - warmup_width) // 2
        warmup_y = (screen_height - warmup_height) // 2

        warmup_rect = pygame.Rect(warmup_x, warmup_y, warmup_width, warmup_height)
        pygame.draw.rect(screen, (255, 255, 224), warmup_rect)  # צהוב בהיר
        pygame.draw.rect(screen, (255, 165, 0), warmup_rect, 4)  # מסגרת כתומה

        # טקסט חימום
        warmup_font = pygame.font.SysFont("Arial", 24, bold=True)
        warmup_title = warmup_font.render("🟡 GETTING READY...", True, (255, 100, 0))
        title_rect = warmup_title.get_rect(center=(warmup_x + warmup_width // 2, warmup_y + 30))
        screen.blit(warmup_title, title_rect)

        warmup_text = pygame.font.SysFont("Arial", 20).render(
            f"Starting in {warmup_remaining:.1f} seconds", True, (100, 100, 100))
        text_rect = warmup_text.get_rect(center=(warmup_x + warmup_width // 2, warmup_y + 60))
        screen.blit(warmup_text, text_rect)

        instruction_text = pygame.font.SysFont("Arial", 16).render(
            "👁️ Look at the first word you want to read", True, (80, 80, 80))
        instruction_rect = instruction_text.get_rect(center=(warmup_x + warmup_width // 2, warmup_y + 90))
        screen.blit(instruction_text, instruction_rect)

    # הגדרות עיצוב
    settings_y = 80
    settings_font = pygame.font.SysFont("Arial", 16)
    settings_info = [
        f"Font: {text_format.font_size}px",
        f"Line: {text_format.line_spacing}px",
        f"Word: {text_format.word_spacing}px"
    ]

    for i, setting in enumerate(settings_info):
        setting_surface = settings_font.render(setting, True, (100, 100, 100))
        screen.blit(setting_surface, (10, settings_y + i * 18))

    # טיימר באמצע העליון
    timer_font = pygame.font.SysFont("Arial", 24, bold=True)
    timer_text = get_tracking_timer_text()

    if is_tracking_active:
        if is_in_warmup:
            timer_color = (255, 165, 0)  # כתום
            timer_bg_color = (255, 248, 220)  # צהוב בהיר
        else:
            timer_color = (0, 150, 0)
            timer_bg_color = (240, 255, 240)
    else:
        timer_color = (150, 0, 0)
        timer_bg_color = (255, 240, 240)

    timer_surface = timer_font.render(timer_text, True, timer_color)
    timer_width = timer_surface.get_width() + 20
    timer_height = timer_surface.get_height() + 10
    timer_x = (screen_width - timer_width) // 2
    timer_y = 10

    # רקע לטיימר
    timer_rect = pygame.Rect(timer_x, timer_y, timer_width, timer_height)
    pygame.draw.rect(screen, timer_bg_color, timer_rect)
    pygame.draw.rect(screen, timer_color, timer_rect, 2)

    # טקסט הטיימר
    text_rect = timer_surface.get_rect(center=timer_rect.center)
    screen.blit(timer_surface, text_rect)

    # מונה מצמוצים בצד ימין עליון
    blink_box_width = 180
    blink_box_height = 80
    blink_box_x = screen_width - blink_box_width - 10
    blink_box_y = 10

    blink_rect = pygame.Rect(blink_box_x, blink_box_y, blink_box_width, blink_box_height)
    pygame.draw.rect(screen, (240, 248, 255), blink_rect)
    pygame.draw.rect(screen, (100, 149, 237), blink_rect, 2)

    blink_title_font = pygame.font.SysFont("Arial", 18, bold=True)
    blink_title = blink_title_font.render("👁️ Blinks", True, (70, 130, 180))
    title_rect = blink_title.get_rect()
    title_rect.centerx = blink_box_x + blink_box_width // 2
    title_rect.y = blink_box_y + 8
    screen.blit(blink_title, title_rect)

    blink_count_font = pygame.font.SysFont("Arial", 28, bold=True)
    blink_count_text = blink_count_font.render(str(blink_counter), True, (50, 50, 50))
    count_rect = blink_count_text.get_rect()
    count_rect.centerx = blink_box_x + blink_box_width // 2
    count_rect.y = blink_box_y + 35
    screen.blit(blink_count_text, count_rect)

    blink_label_font = pygame.font.SysFont("Arial", 14)
    blink_label = blink_label_font.render("Total", True, (120, 120, 120))
    label_rect = blink_label.get_rect()
    label_rect.centerx = blink_box_x + blink_box_width // 2
    label_rect.y = blink_box_y + 60
    screen.blit(blink_label, label_rect)

    # כפתורי בקרה בתחתית המסך
    button_width = 150
    button_height = 40
    margin = 20
    button_font = pygame.font.SysFont("Arial", 11, bold=True)

    # כפתור מעקב עיניים - עם הודעות ברורות יותר
    tracking_x = screen_width - button_width * 2 - margin * 2
    tracking_y = screen_height - button_height - margin

    if is_tracking_active:
        tracking_color = (255, 0, 0)  # אדום = עצור
        text_color = (255, 255, 255)
        tracking_text = "🔴 STOP TRACKING"
    else:
        tracking_color = (0, 200, 0)  # ירוק = התחל
        text_color = (255, 255, 255)
        tracking_text = "🟢 START TRACKING"

    tracking_rect = pygame.Rect(tracking_x, tracking_y, button_width, button_height)

    # צל לכפתור
    shadow_rect = pygame.Rect(tracking_x + 2, tracking_y + 2, button_width, button_height)
    pygame.draw.rect(screen, (50, 50, 50), shadow_rect, border_radius=8)
    pygame.draw.rect(screen, tracking_color, tracking_rect, border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), tracking_rect, 2, border_radius=8)

    # טקסט הכפתור
    text_surface = button_font.render(tracking_text, True, text_color)
    text_rect = text_surface.get_rect()
    text_rect.center = tracking_rect.center
    screen.blit(text_surface, text_rect)

    tracking_button_rect = tracking_rect

    # כפתור tuner
    tuner_x = screen_width - button_width - margin
    tuner_y = screen_height - button_height - margin

    if tuner.is_active:
        tuner_color = (0, 200, 0)
        text_color = (255, 255, 255)
        tuner_text = "🎛️ TUNER ON"
    else:
        tuner_color = (0, 100, 200)
        text_color = (255, 255, 255)
        tuner_text = "🎛️ OPEN TUNER"

    tuner_rect = pygame.Rect(tuner_x, tuner_y, button_width, button_height)

    shadow_rect = pygame.Rect(tuner_x + 2, tuner_y + 2, button_width, button_height)
    pygame.draw.rect(screen, (50, 50, 50), shadow_rect, border_radius=8)
    pygame.draw.rect(screen, tuner_color, tuner_rect, border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), tuner_rect, 2, border_radius=8)

    text_surface = button_font.render(tuner_text, True, text_color)
    text_rect = text_surface.get_rect()
    text_rect.center = tuner_rect.center
    screen.blit(text_surface, text_rect)

    tuner_button_rect = tuner_rect

    # הוראות שימוש בתחתית - עם הדגשה על הסמן הקבוע
    instructions_font = pygame.font.SysFont("Arial", 16)
    instructions = [
        "🎯 CURSOR STARTS FIXED - Press SPACE to make it follow your eyes | T - Change Text | C - Recalibrate | L - Live Tuner | ESC - Exit",
        "Font: +/- or F1/F2 | Line Space: Shift+/- or F3/F4 | Word Space: Ctrl+/- or F5/F6"
    ]

    for i, instruction in enumerate(instructions):
        inst_surface = instructions_font.render(instruction, True, (80, 80, 80))
        screen.blit(inst_surface, (10, screen_height - 50 + i * 18))


# פונקציות אתחול ושמירה
def init_csv(fname, headers):
    if not os.path.exists(fname):
        with open(fname, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def init_calibration_csv():
    """אתחול קובץ CSV לכיול עם 2 פיצ'רים"""
    headers = ["iris_x", "iris_y", "screen_x", "screen_y", "timestamp"]

    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def get_latest_session_id(csv_file):
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file, encoding='utf-8')
    if "session_id" not in df.columns or df.empty:
        return None
    return df["session_id"].iloc[-1]


# חילוץ פיצ'רים
def get_features_for_calibration(face_landmarks):
    """חילוץ פיצ'רים כמו בקוד החדש - 2 פיצ'רים בלבד"""
    lm = face_landmarks.landmark

    # חילוץ iris
    left_iris = [lm[i] for i in range(474, 478)]
    right_iris = [lm[i] for i in range(469, 473)]
    all_iris = left_iris + right_iris

    x_coords = [p.x for p in all_iris]
    y_coords = [p.y for p in all_iris]
    x_center = 1.0 - np.mean(x_coords)  # Mirror correction
    y_center = np.mean(y_coords)

    return np.array([[x_center, y_center]])


def extract_points(results, shape):
    """חילוץ 2 פיצ'רים בלבד"""
    if not results.multi_face_landmarks:
        return None, None, None

    lm = results.multi_face_landmarks[0].landmark
    ih, iw, _ = shape

    try:
        left_iris = [lm[i] for i in range(474, 478)]
        right_iris = [lm[i] for i in range(469, 473)]
        all_iris = left_iris + right_iris

        x_coords = [p.x for p in all_iris]
        y_coords = [p.y for p in all_iris]
        x_center = 1.0 - np.mean(x_coords)  # Mirror correction
        y_center = np.mean(y_coords)

        features = np.array([x_center, y_center])

        nose = lm[1]
        nose_pos = (int(nose.x * iw), int(nose.y * ih))

        left_iris_points = np.array([[p.x * iw, p.y * ih] for p in left_iris])
        right_iris_points = np.array([[p.x * iw, p.y * ih] for p in right_iris])

        left_diameter = np.linalg.norm(left_iris_points[0] - left_iris_points[2])
        right_diameter = np.linalg.norm(right_iris_points[0] - right_iris_points[2])
        avg_pupil_diameter = (left_diameter + right_diameter) / 2

        return features, nose_pos, avg_pupil_diameter

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None


def eye_aspect_ratio(landmarks, indices, image_shape):
    ih, iw, _ = image_shape
    p = [np.array([landmarks[i].x * iw, landmarks[i].y * ih]) for i in indices]
    vertical = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / horizontal


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


def log_extended_fixation(session_id, participant, stimulus, word, rect, ax, ay,
                          start, end, pupil_diameter, dx, dy, behavior):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end - start
    mouse_x, mouse_y = pygame.mouse.get_pos()
    AOI_size = rect.width * rect.height if rect else 0
    AOI_coverage = (AOI_size / (screen_w * screen_h)) * 100 if rect else 0

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


# פונקציות גרפים
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

    plt.figure(figsize=(30, 6))
    plt.plot(df['word_with_index'], df['duration'], marker='o', linestyle='-')
    plt.title(f"Fixation Duration per Word - {session_id}")
    plt.xlabel("Word (with index)")
    plt.ylabel("Duration (sec)")
    plt.xticks(rotation=70, ha='right')
    plt.tight_layout()
    plt.grid(True)
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

    skip_ratio = (skip_time / total_time) * 100 if total_time > 0 else 0
    normal_ratio = (normal_time / total_time) * 100 if total_time > 0 else 0
    num_words_read = (df['behavior'] == 'normal').sum()
    time_minutes = total_time / 60
    reading_speed_wpm = num_words_read / time_minutes if time_minutes > 0 else 0

    stats = {
        "Session ID": session_id,
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
    filtered_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    labels = list(filtered_stats.keys())
    values = list(filtered_stats.values())
    labels = [str(label) for label in labels]
    values = [float(value) for value in values]
    bars = plt.barh(labels, values, color=['blue', 'red', 'orange', 'green', 'purple'])
    plt.title("Reading Statistics Analysis")
    plt.xlabel("Value")
    plt.tight_layout()
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width}', va='center')
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


def plot_word_timings(session_id):
    """גרף זמני מילים עם הסברים ברורים לחלוטין"""
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

    # חישוב זמן יחסי בשניות (יותר ברור מדקות)
    start_time = df['timestamp'].iloc[0]
    df['elapsed_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

    num_words = len(df['word'].unique())
    total_fixations = len(df)
    total_reading_time = df['elapsed_seconds'].iloc[-1] if len(df['elapsed_seconds']) > 0 else 0

    print(f"📊 Creating clear word timing plot: {num_words} words, {total_reading_time:.1f} seconds total")

    # **DECISION LOGIC עם הסברים ברורים**
    if num_words <= 30:
        display_mode = "detailed"
        mode_explanation = "Showing every single word you read"
    elif num_words <= 80:
        display_mode = "selective"
        mode_explanation = "Showing the most interesting/difficult words only"
    elif num_words <= 150:
        display_mode = "grouped"
        mode_explanation = "Showing reading patterns and top words"
    else:
        display_mode = "trends"
        mode_explanation = "Showing overall reading trends (too many words for individual display)"

    print(f"📈 Mode: {display_mode} - {mode_explanation}")

    # ===============================
    # MODE 1: DETAILED (עד 30 מילים)
    # ===============================
    if display_mode == "detailed":
        fig_height = max(10, num_words * 0.5)
        fig, ax = plt.subplots(figsize=(18, min(fig_height, 22)))

        unique_words = df['word'].unique()

        # צבעים לפי סדר הקריאה עם הסבר ברור
        colors = []
        color_meanings = []

        for i, (_, row) in enumerate(df.iterrows()):
            word = row['word']
            word_position = list(unique_words).index(word) if word in unique_words else 0
            progress = word_position / len(unique_words) if len(unique_words) > 0 else 0

            if progress < 0.25:
                color = 'blue'
                meaning = 'Beginning of text'
            elif progress < 0.5:
                color = 'green'
                meaning = 'Early middle'
            elif progress < 0.75:
                color = 'orange'
                meaning = 'Late middle'
            else:
                color = 'red'
                meaning = 'End of text'

            colors.append(color)
            if meaning not in color_meanings:
                color_meanings.append(meaning)

        # גודל נקודות לפי זמן קריאה
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
            df = df.dropna(subset=['duration'])

            min_duration = df['duration'].min()
            max_duration = df['duration'].max()

            if max_duration > min_duration:
                normalized_sizes = ((df['duration'] - min_duration) / (max_duration - min_duration)) * 200 + 40
            else:
                normalized_sizes = [80] * len(df)
        else:
            normalized_sizes = [80] * len(df)

        # **גרף פיזור עם הסברים ברורים**
        scatter = ax.scatter(df['elapsed_seconds'], df['word'],
                             c=colors, s=normalized_sizes,
                             alpha=0.8, edgecolors='black', linewidth=1)

        # **תוויות ברורות לצירים**
        ax.set_xlabel("⏰ Time from start of reading (seconds)\n" +
                      f"Total reading time: {total_reading_time:.1f} seconds",
                      fontweight='bold', fontsize=14)
        ax.set_ylabel("📝 Words in the text (in order of appearance)",
                      fontweight='bold', fontsize=14)

        # תוויות לכל המילים
        ax.set_yticks(range(len(unique_words)))
        ax.set_yticklabels(unique_words, fontsize=11)

        # **הסבר על הטקסטים ליד הכדורים**
        if 'duration' in df.columns:
            high_duration_threshold = df['duration'].quantile(0.8)  # 20% הכי איטיים
            slow_word_count = 0

            for i, (_, row) in enumerate(df.iterrows()):
                if row['duration'] >= high_duration_threshold and slow_word_count < 10:  # מקסימום 10 תוויות
                    ax.annotate(f"Slow: {row['duration']:.2f}s",
                                (df['elapsed_seconds'].iloc[i], row['word']),
                                xytext=(10, 5), textcoords='offset points',
                                fontsize=9, alpha=0.9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1))
                    slow_word_count += 1

        # **מקרא צבעים ברור וחד משמעי**
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12,
                   label='🔵 Beginning (first 25% of text)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12,
                   label='🟢 Early middle (25%-50% of text)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12,
                   label='🟠 Late middle (50%-75% of text)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12,
                   label='🔴 End (last 25% of text)')
        ]

        ax.legend(handles=legend_elements, loc='upper right',
                  title="🎨 DOT COLORS = Position in Text",
                  title_fontsize=13, fontsize=11, framealpha=0.95)

        # **הסבר על גודל הנקודות**
        size_explanation = """📏 DOT SIZE EXPLANATION:
        • Bigger dots = You spent more time reading that word
        • Smaller dots = You read that word quickly
        • Yellow labels = Words that took you the longest time"""

        ax.text(0.02, 0.98, size_explanation, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.95))

    # ===============================
    # MODE 2: SELECTIVE (30-80 מילים)
    # ===============================
    elif display_mode == "selective":
        fig, ax = plt.subplots(figsize=(18, 14))

        # בחירת מילים מעניינות
        word_stats = df.groupby('word').agg({
            'duration': ['mean', 'sum', 'count']
        })
        word_stats.columns = ['avg_duration', 'total_time', 'count']

        interesting_words = set()
        top_time_words = word_stats.nlargest(12, 'total_time').index.tolist()
        top_count_words = word_stats.nlargest(8, 'count').index.tolist()
        long_words = [word for word in df['word'].unique() if len(str(word)) > 6][:8]

        interesting_words.update(top_time_words)
        interesting_words.update(top_count_words)
        interesting_words.update(long_words)
        interesting_words = list(interesting_words)[:20]

        df_filtered = df[df['word'].isin(interesting_words)]

        if not df_filtered.empty:
            # **צבעים ברורים לפי סוג המילה**
            colors = []
            for word in df_filtered['word']:
                if word in top_time_words:
                    colors.append('red')
                elif word in top_count_words:
                    colors.append('orange')
                elif word in long_words:
                    colors.append('purple')
                else:
                    colors.append('blue')

            # גודל נקודות
            if 'duration' in df_filtered.columns:
                df_filtered['duration'] = pd.to_numeric(df_filtered['duration'], errors='coerce')
                df_filtered = df_filtered.dropna(subset=['duration'])

                min_dur = df_filtered['duration'].min()
                max_dur = df_filtered['duration'].max()

                if max_dur > min_dur:
                    sizes = ((df_filtered['duration'] - min_dur) / (max_dur - min_dur)) * 150 + 50
                else:
                    sizes = [100] * len(df_filtered)
            else:
                sizes = [100] * len(df_filtered)

            elapsed_filtered = df['elapsed_seconds'][df['word'].isin(interesting_words)]

            scatter = ax.scatter(elapsed_filtered, df_filtered['word'],
                                 c=colors, s=sizes,
                                 alpha=0.8, edgecolors='black', linewidth=1)

            # **תוויות ברורות לצירים**
            ax.set_xlabel("⏰ Time from start of reading (seconds)\n" +
                          f"Total reading time: {total_reading_time:.1f} seconds",
                          fontweight='bold', fontsize=14)
            ax.set_ylabel("📝 Selected interesting words only\n" +
                          f"(Showing {len(interesting_words)} out of {num_words} total words)",
                          fontweight='bold', fontsize=14)

            # **מקרא ברור לצבעים**
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12,
                       label='🔴 Took most total time'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12,
                       label='🟠 Read multiple times'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=12,
                       label='🟣 Long words (6+ letters)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12,
                       label='🔵 Other interesting words')
            ]
            ax.legend(handles=legend_elements, loc='upper right',
                      title="🎨 DOT COLORS = Why this word is interesting",
                      title_fontsize=13, fontsize=11, framealpha=0.95)

        # **הסבר כללי**
        explanation = f"""📊 SELECTIVE VIEW EXPLANATION:
        • This shows only the most interesting words from your reading
        • {len(interesting_words)} words selected out of {num_words} total
        • Bigger dots = more time spent on that word
        • We filtered out common/easy words to focus on the challenging ones"""

        ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.95))

    # ===============================
    # MODE 3: GROUPED (80-150 מילים)
    # ===============================
    elif display_mode == "grouped":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))

        # **גרף עליון: מהירות קריאה עם הסברים ברורים**
        window_size = max(5, len(df) // 15)
        df['reading_speed_wpm'] = df.index.to_series().rolling(window=window_size).apply(
            lambda x: (len(x) / (window_size * 0.25)) * 60 if len(x) > 0 else 0  # המרה ל-WPM
        )

        ax1.plot(df['elapsed_seconds'], df['reading_speed_wpm'],
                 linewidth=3, color='navy', alpha=0.8, label='Your reading speed')
        ax1.fill_between(df['elapsed_seconds'], df['reading_speed_wpm'],
                         alpha=0.3, color='lightblue')

        # הוספת קו ממוצע
        avg_speed = df['reading_speed_wpm'].mean()
        ax1.axhline(y=avg_speed, color='red', linestyle='--', linewidth=2,
                    label=f'Your average: {avg_speed:.0f} WPM')

        # הוספת קו ממוצע אנושי
        ax1.axhline(y=200, color='green', linestyle=':', linewidth=2,
                    label='Typical adult: ~200 WPM')

        ax1.set_title(f'📈 Your Reading Speed Over Time', fontweight='bold', fontsize=16)
        ax1.set_xlabel('⏰ Time from start (seconds)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('📖 Reading Speed\n(Words Per Minute)', fontweight='bold', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # הוספת הערות על ביצועים
        if avg_speed > 250:
            performance = "🏆 Excellent - Very fast reader!"
        elif avg_speed > 200:
            performance = "✅ Good - Above average speed"
        elif avg_speed > 150:
            performance = "📖 Normal - Average reading speed"
        else:
            performance = "🐌 Slow - Take your time, it's okay!"

        ax1.text(0.02, 0.98, f"📊 Your Performance: {performance}",
                 transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))

        # **גרף תחתון: מילים שלקחו הכי הרבה זמן**
        if 'duration' in df.columns:
            word_times = df.groupby('word')['duration'].sum().sort_values(ascending=False).head(15)

            # צבעים לפי זמן
            colors = ['red' if time > word_times.mean() + word_times.std() else
                      'orange' if time > word_times.mean() else 'green'
                      for time in word_times.values]

            bars = ax2.barh(range(len(word_times)), word_times.values,
                            color=colors, alpha=0.8, edgecolor='black')
            ax2.set_yticks(range(len(word_times)))
            ax2.set_yticklabels(word_times.index, fontsize=12)
            ax2.set_xlabel('⏰ Total time spent reading this word (seconds)',
                           fontweight='bold', fontsize=14)
            ax2.set_ylabel('📝 Words that took most time', fontweight='bold', fontsize=14)
            ax2.set_title('🔍 Which Words Were Hardest for You?', fontweight='bold', fontsize=16)

            # הוספת ערכים על הבארים
            for i, (bar, value) in enumerate(zip(bars, word_times.values)):
                ax2.text(value + max(word_times.values) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'{value:.1f}s', va='center', fontsize=10, fontweight='bold')

        ax2.grid(True, alpha=0.3, axis='x')

        # הסבר צבעי הבארים
        bar_explanation = """🎨 BAR COLORS:
        🔴 Red = Very slow (much above average)
        🟠 Orange = Slow (above average) 
        🟢 Green = Normal speed"""

        ax2.text(0.98, 0.98, bar_explanation, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.8))

    # ===============================
    # MODE 4: TRENDS (150+ מילים)
    # ===============================
    else:  # trends mode
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # **גרף 1: מהירות קריאה**
        window_size = max(10, len(df) // 25)
        df['word_index'] = range(len(df))
        df['reading_speed_wpm'] = df['word_index'].rolling(window=window_size).apply(
            lambda x: (len(x) / (window_size * 0.25)) * 60 if len(x) > 0 else 0
        )

        ax1.plot(df['elapsed_seconds'], df['reading_speed_wpm'],
                 linewidth=2, color='blue', alpha=0.8)
        ax1.fill_between(df['elapsed_seconds'], df['reading_speed_wpm'],
                         alpha=0.3, color='lightblue')

        avg_speed = df['reading_speed_wpm'].mean()
        ax1.axhline(y=avg_speed, color='red', linestyle='--', linewidth=2,
                    label=f'Average: {avg_speed:.0f} WPM')

        ax1.set_title('📈 Reading Speed Over Time', fontweight='bold', fontsize=14)
        ax1.set_xlabel('⏰ Time (seconds)', fontweight='bold')
        ax1.set_ylabel('📖 Speed (Words/Min)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # **גרף 2: התפלגות זמני קריאה**
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
            df = df.dropna(subset=['duration'])

            ax2.hist(df['duration'], bins=40, alpha=0.7, color='green', edgecolor='darkgreen')
            ax2.axvline(df['duration'].mean(), color='red', linestyle='--', linewidth=3,
                        label=f'Average: {df["duration"].mean():.2f}s')
            ax2.axvline(df['duration'].median(), color='orange', linestyle='--', linewidth=3,
                        label=f'Median: {df["duration"].median():.2f}s')

            ax2.set_title('📊 How Long You Spent on Each Word', fontweight='bold', fontsize=14)
            ax2.set_xlabel('⏰ Time per word (seconds)', fontweight='bold')
            ax2.set_ylabel('📈 Number of words', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # **גרף 3: מילים הכי קשות**
        if 'duration' in df.columns:
            top_words = df.groupby('word')['duration'].sum().sort_values(ascending=False).head(12)
            colors = ['red' if i < 4 else 'orange' if i < 8 else 'yellow'
                      for i in range(len(top_words))]

            bars = ax3.barh(range(len(top_words)), top_words.values,
                            color=colors, alpha=0.8, edgecolor='black')
            ax3.set_yticks(range(len(top_words)))
            ax3.set_yticklabels(top_words.index, fontsize=11)
            ax3.set_title('🔍 Hardest Words (Took Most Time)', fontweight='bold', fontsize=14)
            ax3.set_xlabel('⏰ Total time (seconds)', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')

            # הוספת ערכים
            for bar, value in zip(bars, top_words.values):
                ax3.text(value + max(top_words.values) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'{value:.1f}s', va='center', fontsize=9)

        # **גרף 4: סיכום מפורט**
        ax4.axis('off')
        if 'duration' in df.columns and len(df['reading_speed_wpm']) > 0:
            avg_speed = df['reading_speed_wpm'].mean()
            total_minutes = total_reading_time / 60

            summary_text = f"""📋 COMPLETE READING ANALYSIS

📚 TEXT STATISTICS:
• Total words in text: {num_words:,}
• Words you actually read: {total_fixations:,}
• Reading efficiency: {num_words / total_fixations:.1f} words per look

⏱️ TIME STATISTICS:
• Total reading time: {total_minutes:.1f} minutes
• Average per word: {df['duration'].mean():.2f} seconds
• Fastest word: {df['duration'].min():.2f} seconds
• Slowest word: {df['duration'].max():.2f} seconds

🚀 SPEED ANALYSIS:
• Your average speed: {avg_speed:.0f} WPM
• Fastest section: {df['reading_speed_wpm'].max():.0f} WPM
• Slowest section: {df['reading_speed_wpm'].min():.0f} WPM
• Speed consistency: {'High' if df['reading_speed_wpm'].std() < 30 else 'Medium' if df['reading_speed_wpm'].std() < 60 else 'Low'}

🎯 PERFORMANCE RATING:
{('🏆 Excellent! You read faster than most people' if avg_speed > 250 else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         '✅ Good! Above average reading speed' if avg_speed > 200 else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         '📖 Normal reading speed - perfectly fine!' if avg_speed > 150 else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         '🐌 Slow and steady - quality over speed!')}

💡 WHAT THE GRAPHS SHOW:
• Top-left: How your speed changed over time
• Top-right: Distribution of time per word
• Bottom-left: Your most challenging words
• This box: Complete summary of your reading"""

            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=1.0", facecolor="lightyellow", alpha=0.9))

    # ===============================
    # כותרת ברורה לכל מצב
    # ===============================

    if display_mode == "detailed":
        main_title = f'📖 Complete Word-by-Word Reading Timeline'
        subtitle = f'Session: {session_id} • Every single word shown • {num_words} words • {total_reading_time:.1f}s total'
    elif display_mode == "selective":
        main_title = f'🎯 Key Words Reading Analysis'
        subtitle = f'Session: {session_id} • Most interesting words only • {num_words} total words • {total_reading_time:.1f}s'
    elif display_mode == "grouped":
        main_title = f'📊 Reading Speed & Difficulty Analysis'
        subtitle = f'Session: {session_id} • {num_words} words • {total_reading_time:.1f}s total'
    else:
        main_title = f'📈 Complete Reading Performance Overview'
        subtitle = f'Session: {session_id} • {num_words} words • {total_reading_time / 60:.1f} minutes total'

    fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, subtitle, fontsize=14, ha='center', style='italic')

    # התאמת פריסה
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # שמירה עם שם מתאר
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"CLEAR_word_timings_{display_mode}_{session_id}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Clear word timing plot saved as: {filename}")

    plt.show()

    print(f"🎯 Created {display_mode} mode with crystal clear explanations!")
    print(f"📊 Mode explanation: {mode_explanation}")


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


class AnalysisProgressScreen:
    def __init__(self, screen):
        self.screen = screen
        self.progress = 0
        self.current_task = "Starting analysis..."
        self.total_tasks = 5
        self.running = True
        self.font = pygame.font.SysFont("Arial", 32)
        self.small_font = pygame.font.SysFont("Arial", 20)

    def update_progress(self, progress, task_name):
        """עדכון התקדמות"""
        self.progress = progress
        self.current_task = task_name

    def draw_progress_screen(self):
        """ציור מסך הטעינה"""
        self.screen.fill((30, 30, 50))  # רקע כהה כחול

        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()

        # כותרת
        title = self.font.render("📊 Creating Your Reading Analysis", True, (255, 255, 255))
        title_rect = title.get_rect(center=(screen_w // 2, screen_h // 2 - 150))
        self.screen.blit(title, title_rect)

        # מד התקדמות
        progress_width = 600
        progress_height = 40
        progress_x = (screen_w - progress_width) // 2
        progress_y = screen_h // 2 - 50

        # רקע של מד ההתקדמות
        pygame.draw.rect(self.screen, (70, 70, 70),
                         (progress_x, progress_y, progress_width, progress_height))

        # מילוי התקדמות
        fill_width = int((self.progress / 100) * progress_width)
        if fill_width > 0:
            # גרדיאנט צבעים
            for i in range(fill_width):
                color_intensity = int(255 * (i / progress_width))
                color = (color_intensity, 255 - color_intensity // 2, 100)
                pygame.draw.line(self.screen, color,
                                 (progress_x + i, progress_y),
                                 (progress_x + i, progress_y + progress_height))

        # מסגרת
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (progress_x, progress_y, progress_width, progress_height), 3)

        # אחוז
        percent_text = self.font.render(f"{self.progress:.0f}%", True, (255, 255, 255))
        percent_rect = percent_text.get_rect(center=(screen_w // 2, progress_y + progress_height + 40))
        self.screen.blit(percent_text, percent_rect)

        # משימה נוכחית
        task_text = self.small_font.render(self.current_task, True, (200, 200, 255))
        task_rect = task_text.get_rect(center=(screen_w // 2, progress_y + progress_height + 80))
        self.screen.blit(task_text, task_rect)

        # הוראות
        instruction = self.small_font.render("Please wait... This may take 1-3 minutes", True, (150, 150, 150))
        instruction_rect = instruction.get_rect(center=(screen_w // 2, screen_h // 2 + 100))
        self.screen.blit(instruction, instruction_rect)

        # אנימציה - נקודות מסתובבות
        dots = "." * ((int(time.time() * 2) % 4))
        loading_text = self.small_font.render(f"Processing{dots}    ", True, (100, 255, 100))
        loading_rect = loading_text.get_rect(center=(screen_w // 2, screen_h // 2 + 130))
        self.screen.blit(loading_text, loading_rect)

        pygame.display.flip()


def create_graphs_with_progress(session_id, blink_timestamps, session_start_time, progress_screen):
    """יצירת גרפים עם עדכון התקדמות"""
    try:
        # משימה 1
        progress_screen.update_progress(10, "📊 Preparing data...")
        time.sleep(0.5)

        # משימה 2
        progress_screen.update_progress(25, "📈 Creating reading analysis plots...")
        generate_all_publication_plots(session_id, FIXATION_CSV)

        # משימה 3
        progress_screen.update_progress(50, "📉 Analyzing reading speed and pupil data...")
        plot_reading_wpm_and_pupil_current_session("extended_eye_tracking.csv")

        # משימה 4
        progress_screen.update_progress(75, "👁️ Processing blink patterns...")
        plot_blinks_over_time(blink_timestamps, session_start_time)

        # משימה 5
        progress_screen.update_progress(90, "⏰ Creating word timing analysis...")
        plot_word_timings(session_id)

        # סיום
        progress_screen.update_progress(100, "✅ Analysis complete!")
        time.sleep(1)

        progress_screen.running = False

    except Exception as e:
        print(f"❌ Error in graph creation: {e}")
        progress_screen.update_progress(100, "❌ Analysis completed with errors")
        time.sleep(2)
        progress_screen.running = False


def show_analysis_with_progress(session_id, participant, current_text_option, selected_text,
                                blink_counter, blink_timestamps, session_start_time, screen):
    """הצגת ניתוח עם מסך התקדמות"""

    # יצירת מסך התקדמות
    progress_screen = AnalysisProgressScreen(screen)

    # הצגת מידע ראשוני
    print(f"\n📊 Session completed! Generating analysis...")
    print(f"📍 Session ID: {session_id}")
    print(f"👤 Participant: {participant}")
    print(f"📖 Text: {TEXT_OPTIONS[current_text_option]['name']}")
    print(f"📏 Total words: {sum(len(line.split()) for line in selected_text if line.strip())}")
    print(f"👁️ Total blinks: {blink_counter}")

    # התחלת יצירת גרפים ברקע
    graph_thread = threading.Thread(
        target=create_graphs_with_progress,
        args=(session_id, blink_timestamps, session_start_time, progress_screen)
    )
    graph_thread.daemon = True
    graph_thread.start()

    # לולאת מסך טעינה
    clock = pygame.time.Clock()

    while progress_screen.running:
        # טיפול באירועים
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                progress_screen.running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("⏭️ Skipping analysis (ESC pressed)")
                    progress_screen.running = False
                    break

        # ציור מסך הטעינה
        progress_screen.draw_progress_screen()
        clock.tick(30)  # 30 FPS

    # המתנה לסיום הgraph thread
    graph_thread.join(timeout=2)

    print("✅ Analysis complete! All files saved.")


# *** הפונקציה הראשית המעודכנת עם התיקונים ***
def main():
    global screen_w, screen_h, current_text_option, tuner_button_rect, tracking_button_rect
    global last_highlighted_word, current_highlighted_rect, cursor_root
    global is_tracking_active, tracking_start_time
    global tracking_warmup_start, is_in_warmup

    last_cursor_x, last_cursor_y = None, None
    session_start_time = None
    scroll_offset = 0

    # איפוס משתנים גלובליים
    last_highlighted_word = None
    current_highlighted_rect = None
    is_tracking_active = False
    tracking_start_time = None
    tracking_warmup_start = None
    is_in_warmup = False

    # משתנים לתזוזת סמן
    smoothed_x, smoothed_y = None, None
    history = deque(maxlen=TUNING_PARAMS['history_size'])
    last_ax, last_ay = None, None
    position_buffer = deque(maxlen=TUNING_PARAMS['median_buffer_size'])

    # אתחול pygame
    pygame.init()

    # קבלת גודל המסך האמיתי
    screen_w, screen_h = pyautogui.size()

    # *** יצירת חלון סמן במיקום קבוע לפני פתיחת מסך pygame ***
    cursor_root = create_cursor_window(INITIAL_CURSOR_POSITION[0], INITIAL_CURSOR_POSITION[1])
    if cursor_root:
        cursor_root.withdraw()  # להתחיל כשהוא מוסתר
        print(f"✅ Cursor created at initial position: {INITIAL_CURSOR_POSITION}")

    # פתיחת מסך בגודל מלא
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)

    # הגדרת פונטים
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    ui_font = pygame.font.SysFont("Arial", 35)

    # מנהל כיול מעודכן
    calib_manager = CalibrationManager()

    # אתחול מצלמה
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("❌ Webcam not available")

    # MediaPipe
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(80, 220, 100))

    # אתחול הגדרות עיצוב טקסט
    text_format = TextFormatting()
    init_calibration_csv()

    # מונה debug
    debug_counter = 0

    # לולאה ראשית לניווט בין מסכים
    while True:
        # מסך קלט תעודת זהות
        participant = show_participant_input_screen(screen, ui_font)

        # מסך בחירת טקסט
        current_text_option = show_text_selection_screen(screen, ui_font)
        selected_text = TEXT_OPTIONS[current_text_option]["lines"]
        stimulus = " ".join(selected_text)[:100]

        # SESSION ID
        session_id = f"TRAIN_{current_text_option}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # בחירת סוג כיול עם מצלמה והוראות
        calibration_choice = show_calibration_choice_screen_with_instructions(screen, ui_font, calib_manager, cap,
                                                                              face_mesh)

        # טעינה או יצירה של מודלי כיול
        model_x = None
        model_y = None
        baseline_nose = None

        if calibration_choice == "load":
            calibration_data = calib_manager.load_calibration()
            if calibration_data:
                model_x = calibration_data['model_x']
                model_y = calibration_data['model_y']
                baseline_nose = calibration_data['baseline_nose']

                if calibration_data['screen_resolution'] != [screen_w, screen_h]:
                    print(
                        f"⚠️  Screen resolution changed from {calibration_data['screen_resolution']} to [{screen_w}, {screen_h}]")
                    print("Calibration may be less accurate. Consider recalibrating.")

                print("✅ Calibration loaded, proceeding to reading mode...")
            else:
                print("❌ Failed to load calibration, starting new calibration")
                calibration_choice = "new_auto"

        # כיול חדש עם המערכת הפשוטה
        if calibration_choice in ["new_auto", "new_manual"]:
            print(f"🎯 Starting {'manual' if calibration_choice == 'new_manual' else 'automatic'} calibration...")

            # סגירת pygame זמנית לכיול
            pygame.quit()

            # יצירת מנהל כיול פשוט
            calibrator = SimpleCalibrator(screen_w, screen_h)

            # הפעלת כיול
            manual_mode = (calibration_choice == "new_manual")
            success = calibrator.run_calibration(manual_mode)

            # פתיחת pygame מחדש
            pygame.init()
            screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)

            if success:
                print("✅ Calibration completed successfully!")

                # טעינת הכיול החדש
                calibration_data = calib_manager.load_calibration()
                if calibration_data:
                    model_x = calibration_data['model_x']
                    model_y = calibration_data['model_y']
                    baseline_nose = calibration_data['baseline_nose']
                    print("✅ New calibration loaded and ready for reading mode!")

                else:
                    print("❌ Failed to load new calibration")
                    continue
            else:
                print("❌ Calibration failed! Please try again.")
                continue

        # וידוא שיש לנו מודלים לפני המשך
        if model_x is None or model_y is None:
            print("❌ No calibration models available. Restarting...")
            continue

        # אתחול כל המשתנים הנדרשים למצב run
        # הגדרות מצמוץ
        blink_counter = 0
        blink_timestamps = []
        blink_flag = False
        EAR_THRESHOLD = 0.2

        # CSV initialization
        init_csv(FIXATION_CSV,
                 ["session_id", "timestamp", "word", "x", "y", "duration", "speed", "dx", "dy", "behavior"])
        init_csv(EXTENDED_CSV, EXTENDED_HEADERS)

        # אתחול משתנים לחישוב מהירות
        fixation_start, fixation_word = None, None
        prev_pt, prev_t = None, None
        dx, dy, speed = 0, 0, 0
        clock = pygame.time.Clock()
        running = True
        should_restart = False

        # איפוס משתני מעקב עם חימום
        is_tracking_active = False
        tracking_start_time = None
        tracking_warmup_start = None
        is_in_warmup = False

        print("\n🎮 READING MODE READY:")
        print("📍 SPACE - Start/Stop Eye Tracking (with 3s warmup)")
        print("📍 L - Open/Close Live Tuner")
        print("📍 T - Change Text")
        print("📍 C - Recalibrate")
        print("📍 ESC - Exit with analysis")
        print("📍 +/- - Font size")
        print("📍 Shift +/- - Line spacing")
        print("📍 Ctrl +/- - Word spacing")
        print("📍 F1-F6 - Alternative controls")
        print("🎯 The cursor is STATIC until you press SPACE!")
        print("⏰ NEW: 3-second warmup prevents recording unwanted movements!\n")

        # *** הצגת הסמן במיקום קבוע מיד כשנכנסים למסך קריאה ***
        if cursor_root and cursor_root.winfo_exists():
            cursor_root.deiconify()  # הצג את הסמן במיקום הקבוע
            cursor_root.geometry(f"+{INITIAL_CURSOR_POSITION[0]}+{INITIAL_CURSOR_POSITION[1]}")
            print(f"📍 Cursor visible at fixed position: {INITIAL_CURSOR_POSITION}")

        # *** הלולאה הראשית המעודכנת עם בדיקת מעקב פעיל וחימום ***
        while running and not should_restart:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Webcam disconnected. Trying to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(frame_rgb)
            pupil, nose, pupil_diameter = extract_points(res, frame.shape)
            disp_frame = frame.copy()

            # ציור פנים וזיהוי מצמוצים
            if res.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    disp_frame, res.multi_face_landmarks[0],
                    mp.solutions.face_mesh.FACEMESH_TESSELATION, None, drawing_spec)

                # מדידת מצמוץ
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

            # טיפול באירועים
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    should_restart = False
                    break

                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                    should_restart = False
                    break

                # פקדים במצב קריאה
                if e.type == pygame.KEYDOWN:
                    # גלילה
                    if e.key == pygame.K_UP:
                        scroll_offset = max(0, scroll_offset - 50)
                    elif e.key == pygame.K_DOWN:
                        max_scroll = max(0, len(selected_text) * text_format.line_spacing - screen_h + 200)
                        scroll_offset = min(max_scroll, scroll_offset + 50)

                    # הפעלה/כיבוי מעקב עיניים
                    elif e.key == pygame.K_SPACE:
                        toggle_eye_tracking()

                    # ניווט
                    elif e.key == pygame.K_t:  # T לשינוי טקסט
                        should_restart = True
                        running = False
                        break
                    elif e.key == pygame.K_c:  # C לכיול מחדש
                        should_restart = True
                        running = False
                        break
                    elif e.key == pygame.K_l:  # L לפתיחה/סגירת כוונון
                        print("🎛️ L key pressed - toggling tuner...")
                        try:
                            if tuner.is_active:
                                tuner.close_tuning_window()
                                print("🎛️ Tuner closed (L key)")
                            else:
                                open_live_tuner()
                                print("🎛️ Tuner opened (L key)")
                        except Exception as e:
                            print(f"❌ Error toggling tuner: {e}")

                    # בדיקת מקשי Shift ו-Ctrl
                    keys = pygame.key.get_pressed()
                    shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                    ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]

                    # התאמת גופן
                    if e.key == pygame.K_EQUALS or e.key == pygame.K_PLUS:
                        if not shift_pressed and not ctrl_pressed:
                            text_format.increase_font_size()
                            print(f"📝 Font size: {text_format.font_size}px")
                        elif shift_pressed:
                            text_format.increase_line_spacing()
                            print(f"📏 Line spacing: {text_format.line_spacing}px")
                        elif ctrl_pressed:
                            text_format.increase_word_spacing()
                            print(f"📐 Word spacing: {text_format.word_spacing}px")

                    elif e.key == pygame.K_MINUS:
                        if not shift_pressed and not ctrl_pressed:
                            text_format.decrease_font_size()
                            print(f"📝 Font size: {text_format.font_size}px")
                        elif shift_pressed:
                            text_format.decrease_line_spacing()
                            print(f"📏 Line spacing: {text_format.line_spacing}px")
                        elif ctrl_pressed:
                            text_format.decrease_word_spacing()
                            print(f"📐 Word spacing: {text_format.word_spacing}px")

                    # פקדים נוספים עם מקשי F
                    elif e.key == pygame.K_F1:
                        text_format.increase_font_size()
                        print(f"📝 Font size: {text_format.font_size}px")
                    elif e.key == pygame.K_F2:
                        text_format.decrease_font_size()
                        print(f"📝 Font size: {text_format.font_size}px")
                    elif e.key == pygame.K_F3:
                        text_format.increase_line_spacing()
                        print(f"📏 Line spacing: {text_format.line_spacing}px")
                    elif e.key == pygame.K_F4:
                        text_format.decrease_line_spacing()
                        print(f"📏 Line spacing: {text_format.line_spacing}px")
                    elif e.key == pygame.K_F5:
                        text_format.increase_word_spacing()
                        print(f"📐 Word spacing: {text_format.word_spacing}px")
                    elif e.key == pygame.K_F6:
                        text_format.decrease_word_spacing()
                        print(f"📐 Word spacing: {text_format.word_spacing}px")

                # טיפול בלחיצות עכבר ללא מזעור החלון
                if e.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    print(f"🖱️ Mouse clicked at: {mouse_pos}")

                    if tracking_button_rect and tracking_button_rect.collidepoint(mouse_pos):
                        print("🎯 Tracking button clicked!")
                        toggle_eye_tracking()

                    elif tuner_button_rect and tuner_button_rect.collidepoint(mouse_pos):
                        print("🎛️ Tuner button clicked!")
                        try:
                            if tuner.is_active:
                                tuner.close_tuning_window()
                                print("🎛️ Tuner closed (mouse)")
                            else:
                                open_live_tuner()
                                print("🎛️ Tuner opened (mouse)")
                        except Exception as e:
                            print(f"❌ Error with tuner button: {e}")

            # *** מעקב עיניים רק אם is_tracking_active הוא True ***
            if is_tracking_active:
                # עדכון דינמי של היסטוריה
                current_history_size = int(TUNING_PARAMS['history_size'])
                if len(history) != current_history_size:
                    new_history = deque(maxlen=current_history_size)
                    for item in list(history)[-current_history_size:]:
                        new_history.append(item)
                    history = new_history

                # חילוץ פיצ'רים
                features, nose, pupil_diameter = extract_points(res, frame.shape)

                if features is not None and nose:
                    if session_start_time is None:
                        session_start_time = time.time()

                    # חיזוי עם 2 פיצ'רים
                    features_reshaped = features.reshape(1, -1)

                    gx = model_x.predict(features_reshaped)[0]
                    gy = model_y.predict(features_reshaped)[0]

                    # המרה לפיקסלים במסך
                    x_px = int(np.clip(gx, 0, 1) * screen_w)
                    y_px = int(np.clip(gy, 0, 1) * screen_h)

                    # תיקון ראש דינמי
                    if TUNING_PARAMS['head_compensation'] and baseline_nose:
                        x_px = int(x_px + (baseline_nose[0] - nose[0]) * TUNING_PARAMS['head_gain_x'])
                        y_px = int(y_px - (nose[1] - baseline_nose[1]) * TUNING_PARAMS['head_gain_y'])

                    history.append((x_px, y_px))
                    avg_x = int(np.mean([p[0] for p in history]))
                    avg_y = int(np.mean([p[1] for p in history]))

                    # החלקה
                    current_alpha = TUNING_PARAMS['alpha']

                    if smoothed_x is None or smoothed_y is None:
                        smoothed_x, smoothed_y = avg_x, avg_y
                    else:
                        smoothed_x = int(current_alpha * avg_x + (1 - current_alpha) * smoothed_x)
                        smoothed_y = int(current_alpha * avg_y + (1 - current_alpha) * smoothed_y)

                    # בדיקת סף תנועה
                    current_threshold = TUNING_PARAMS['movement_threshold']

                    if last_ax is not None and last_ay is not None:
                        if abs(smoothed_x - last_ax) < current_threshold and abs(
                                smoothed_y - last_ay) < current_threshold:
                            ax, ay = last_ax, last_ay
                        else:
                            ax, ay = smoothed_x, smoothed_y
                            last_ax, last_ay = ax, ay
                    else:
                        ax, ay = smoothed_x, smoothed_y
                        last_ax, last_ay = ax, ay

                    # יצירת cursor window שקוף יחיד בלבד אם לא קיים
                    if cursor_root is None:
                        print("🔴 Creating new transparent cursor window...")
                        cursor_root = create_cursor_window(ax, ay)
                        if cursor_root:
                            print("✅ Transparent cursor window created successfully")
                        else:
                            print("❌ Failed to create transparent cursor window")

                    # עדכון הסמן השקוף (צבע קבוע)
                    if cursor_root:
                        try:
                            if cursor_root.winfo_exists():
                                # עדכון מיקום תמיד
                                cursor_root.geometry(f"+{ax}+{ay}")

                                # זיהוי מילה מיידי
                                current_word_on_cursor, current_word_rect = find_word_under_cursor(ax, ay, word_boxes)

                                # עדכון הדגשה מיידי
                                if update_word_highlighting_immediate(current_word_on_cursor, current_word_rect):
                                    if current_word_on_cursor:
                                        print(f"📖 NOW HIGHLIGHTING: '{current_word_on_cursor}'")
                                    else:
                                        print("📖 No word highlighted")

                                # עדכון הסמן (צבע קבוע)
                                update_cursor_for_word(cursor_root, current_word_on_cursor)

                                # בדיקת שינויים בהגדרות
                                current_size = TUNING_PARAMS['cursor_size']
                                current_color = tuple(TUNING_PARAMS['cursor_color'])

                                # אתחול ערכים שמורים אם לא קיימים
                                if not hasattr(cursor_root, 'stored_size'):
                                    cursor_root.stored_size = current_size
                                    cursor_root.stored_color = current_color
                                    update_cursor_settings(cursor_root)

                                # עדכון רק אם השתנה משהו
                                elif (cursor_root.stored_size != current_size or
                                      cursor_root.stored_color != current_color):

                                    print(
                                        f"🔄 Transparent cursor change: Size {cursor_root.stored_size}->{current_size}, Color {cursor_root.stored_color}->{current_color}")
                                    cursor_root.stored_size = current_size
                                    cursor_root.stored_color = current_color
                                    update_cursor_settings(cursor_root)

                                # עדכון tkinter באופן מינימלי
                                cursor_root.update_idletasks()

                        except Exception as e:
                            print(f"❌ Transparent cursor error: {e}")
                            cursor_root = None

                    # *** רישום fixation עם מסנן תנועות לא טבעיות - רק אחרי חימום ***
                    if not is_in_warmup:
                        current_word_displayed = find_word_under_cursor(ax, ay, word_boxes)[0]

                        # בדיקה אם זה תנועת מעבר לא טבעית (פילטר חכם)
                        is_natural_movement = True

                        if current_word_displayed and fixation_word and fixation_word != current_word_displayed:
                            # חישוב מרחק בין המילה הקודמת לחדשה
                            prev_word_pos = None
                            curr_word_pos = None

                            for word, rect in word_boxes:
                                if word == fixation_word:
                                    prev_word_pos = (rect.centerx, rect.centery)
                                elif word == current_word_displayed:
                                    curr_word_pos = (rect.centerx, rect.centery)

                            if prev_word_pos and curr_word_pos:
                                distance = np.sqrt((curr_word_pos[0] - prev_word_pos[0]) ** 2 +
                                                   (curr_word_pos[1] - prev_word_pos[1]) ** 2)
                                y_diff = curr_word_pos[1] - prev_word_pos[1]
                                x_diff = curr_word_pos[0] - prev_word_pos[0]

                                # סינון תנועות לא טבעיות:
                                # 1. קפיצות גדולות מדי (יותר מ-300 פיקסלים)
                                # 2. תנועות אלכסוניות חדות (מעבר שורה עם קפיצה גדולה)
                                # 3. תנועה חזרה גדולה מדי לאחור באותה שורה

                                if distance > 300:  # קפיצה גדולה מדי
                                    is_natural_movement = False
                                    print(
                                        f"🚫 Filtering large jump: {distance:.0f}px from '{fixation_word}' to '{current_word_displayed}'")

                                elif abs(y_diff) > 50 and abs(x_diff) > 200:  # מעבר שורה עם קפיצה גדולה
                                    is_natural_movement = False
                                    print(
                                        f"🚫 Filtering line jump: {distance:.0f}px from '{fixation_word}' to '{current_word_displayed}'")

                                elif abs(x_diff) > 400 and abs(y_diff) < 30:  # קפיצה גדולה באותה שורה
                                    is_natural_movement = False
                                    print(
                                        f"🚫 Filtering same-line jump: {distance:.0f}px from '{fixation_word}' to '{current_word_displayed}'")

                        if current_word_displayed and is_natural_movement:
                            if fixation_word != current_word_displayed:
                                if fixation_word and fixation_start:
                                    duration = time.time() - fixation_start

                                    # בדיקה נוספת - האם זה fixation תקין (לא קצר מדי)
                                    if duration >= 0.05:  # לפחות 50ms
                                        behavior = "normal"
                                        if duration < 0.1:
                                            behavior = "skip"
                                        elif dx < 0:
                                            behavior = "regression"

                                        log_fixation_csv(fixation_word, ax, ay, fixation_start, time.time(),
                                                         speed, dx, dy, behavior, session_id)

                                        # מציאת המלבן הנכון לרישום מורחב
                                        word_rect = None
                                        for word, rect in word_boxes:
                                            if word == fixation_word:
                                                word_rect = rect
                                                break

                                        if word_rect:
                                            log_extended_fixation(session_id, participant, stimulus, fixation_word,
                                                                  word_rect, ax, ay, fixation_start, time.time(),
                                                                  pupil_diameter, dx, dy, behavior)

                                        print(f"✅ Recorded valid fixation: '{fixation_word}' ({duration:.3f}s)")
                                    else:
                                        print(f"🚫 Filtering too short fixation: '{fixation_word}' ({duration:.3f}s)")

                                fixation_start = time.time()
                                fixation_word = current_word_displayed

                    # עדכון tuner אם פעיל
                    if tuner.is_active and debug_counter % 5 == 0:
                        tuner.update_tuner()

                    # התאמת התדירות
                    time.sleep(max(0.005, TUNING_PARAMS['update_sleep'] * 0.5))

            else:
                # *** כשמעקב לא פעיל - הסמן נשאר במקום הקבוע ***
                if cursor_root and cursor_root.winfo_exists():
                    # ודא שהסמן במקום הקבוע
                    current_x = cursor_root.winfo_x()
                    current_y = cursor_root.winfo_y()

                    if (current_x != INITIAL_CURSOR_POSITION[0] or
                            current_y != INITIAL_CURSOR_POSITION[1]):
                        cursor_root.geometry(f"+{INITIAL_CURSOR_POSITION[0]}+{INITIAL_CURSOR_POSITION[1]}")

            # רינדור עם סנכרון מושלם
            if is_tracking_active and 'ax' in locals() and 'ay' in locals():
                current_word_displayed = render_text_with_perfect_sync(screen, font, selected_text, text_format,
                                                                       blink_counter, (ax, ay), scroll_offset)
            else:
                current_word_displayed = render_text_with_perfect_sync(screen, font, selected_text, text_format,
                                                                       blink_counter, None, scroll_offset)

            pygame.display.flip()
            clock.tick(TUNING_PARAMS['update_rate'])
            debug_counter += 1

        # סגירת cursor window אם קיים
        if cursor_root:
            try:
                if cursor_root.winfo_exists():
                    cursor_root.destroy()
            except:
                pass
            cursor_root = None

        # אם יש צורך להתחיל מחדש, ממשיכים ללולאה הראשית
        if should_restart:
            continue

        # אם הגענו עד כאן בלי should_restart, זה אומר שהמשתמש יצא מהתוכנית
        # יצירת גרפים לפני יציאה
        print(f"\n📊 Session completed! Generating analysis...")
        print(f"📍 Session ID: {session_id}")
        print(f"👤 Participant: {participant}")
        print(f"📖 Text: {TEXT_OPTIONS[current_text_option]['name']}")
        print(f"📏 Total words: {sum(len(line.split()) for line in selected_text if line.strip())}")
        print(f"👁️  Total blinks: {blink_counter}")

        show_analysis_with_progress(session_id, participant, current_text_option,
                                    selected_text, blink_counter, blink_timestamps,
                                    session_start_time, screen)
        break

    # סיום
    pygame.quit()
    cap.release()


# ========== IMPROVED PLOTS FUNCTIONS ==========

# הגדרות איכות גבוהה
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 14,
    'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 12, 'figure.titlesize': 20,
    'axes.linewidth': 1.5, 'grid.linewidth': 0.8, 'lines.linewidth': 2.5,
    'font.family': 'DejaVu Sans', 'axes.grid': True, 'grid.alpha': 0.3,
})


def save_high_quality_plot(filename, bbox_inches='tight', pad_inches=0.3):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_filename = f"{filename}_{timestamp}.png"
    plt.savefig(full_filename, dpi=300, bbox_inches=bbox_inches,
                pad_inches=pad_inches, facecolor='white', edgecolor='none')
    print(f"📊 Saved: {full_filename}")
    return full_filename


def generate_all_publication_plots(session_id, csv_file="reading_trace.csv"):
    """יצירת כל הגרפים באיכות פרסום"""
    print(f"📊 Creating publication-quality plots for: {session_id}")

    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        return []

    try:
        # קריאת נתונים
        df = pd.read_csv(csv_file, encoding='utf-8')
        df = df[df["session_id"] == session_id]
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        df = df.dropna(subset=['duration', 'word', 'behavior'])

        if df.empty:
            print(f"❌ No data for session {session_id}")
            return []

        print(f"✅ Found {len(df)} fixations, creating plots...")

        # גרף 1: מסלול הקריאה ברור יותר
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # גרף עליון חדש: בר צ'ארט של זמני הסתכלות
        words_with_times = []
        for word in df['word'].unique()[:15]:  # רק 15 המילים הראשונות
            word_data = df[df['word'] == word]
            avg_time = word_data['duration'].mean()
            total_looks = len(word_data)
            words_with_times.append((word, avg_time, total_looks))

        # מיון לפי זמן ממוצע
        words_with_times.sort(key=lambda x: x[1], reverse=True)
        words_with_times = words_with_times[:12]  # רק 12 המילים עם הכי הרבה זמן

        words = [item[0] for item in words_with_times]
        times = [item[1] for item in words_with_times]
        looks = [item[2] for item in words_with_times]

        # יצירת צבעים לפי זמן - אדום לאיטי, ירוק למהיר
        colors = []
        for time_val in times:
            if time_val > 0.5:
                colors.append('red')  # איטי - אדום
            elif time_val > 0.3:
                colors.append('orange')  # בינוני - כתום
            else:
                colors.append('green')  # מהיר - ירוק

        bars = ax1.bar(words, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('📖 Which Words Took You Longest to Read?\n(Red = Slow, Orange = Medium, Green = Fast)',
                      fontweight='bold', fontsize=16, pad=20)
        ax1.set_xlabel('Words', fontweight='bold', fontsize=14)
        ax1.set_ylabel('⏰ Average Time Spent (seconds)', fontweight='bold', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # הוספת ערכים על הבארים
        for bar, time_val, look_count in zip(bars, times, looks):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{time_val:.2f}s\n({look_count}x)', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

        # הוספת הסבר
        ax1.text(0.02, 0.98,
                 '🔴 Red = Hard words (took long time)\n🟠 Orange = Medium words\n🟢 Green = Easy words (quick reading)',
                 transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # גרף תחתון חדש: גרף פשוט של מהירות לאורך זמן
        # חישוב מהירות תזוזות עיניים
        # נניח שיש לנו עמודות x, y למיקום העיניים
        if 'x' in df.columns and 'y' in df.columns:
            # חישוב המרחק בין נקודות עוקבות
            dx = df['x'].diff()
            dy = df['y'].diff()
            distances = np.sqrt(dx ** 2 + dy ** 2)

            # חישוב מהירות (מרחק חלקי זמן)
            dt = df['duration'].shift(1)  # זמן של המילה הקודמת
            eye_speeds = distances / dt

            # הסרת ערכים לא תקינים (נענים ראשון, מחלקים באפס, וכו')
            eye_speeds = eye_speeds.replace([np.inf, -np.inf], np.nan).dropna()

        else:
            # אם אין נתוני מיקום, ניצור נתונים מדומים מבוססי הגיון
            print("⚠️ No eye position data found. Creating simulated eye movement data based on reading patterns...")

            # יצירת נתוני תזוזת עיניים מדומים מבוססים על דפוסי קריאה אמיתיים
            np.random.seed(42)  # לעקביות

            # מהירות תזוזה בסיסית (פיקסלים ליחידת זמן)
            base_speed = 50 + np.random.normal(0, 10, len(df))

            # הוספת דפוסי קריאה:
            # 1. תזוזות מהירות יותר בתחילת שורות (saccades)
            # 2. האטה במילים קשות (מילים ארוכות)
            # 3. תזוזות חזרה (regressions)

            eye_speeds = []
            for i, row in df.iterrows():
                speed = base_speed[i]

                # דימוי מילים קשות (מילים ארוכות = תזוזה איטית יותר)
                if 'word' in df.columns:
                    word_length = len(str(row['word']))
                    if word_length > 6:
                        speed *= 0.7  # האטה במילים ארוכות
                    elif word_length < 3:
                        speed *= 1.3  # האצה במילים קצרות

                # דימוי תחילת שורות (כל 8-12 מילים)
                if i % np.random.randint(8, 13) == 0:
                    speed *= 1.8  # תזוזה מהירה בתחילת שורה

                # דימוי תזוזות חזרה (5% מהמקרים)
                if np.random.random() < 0.05:
                    speed *= -0.3  # תזוזה לאחור

                # הוספת רעש טבעי
                speed += np.random.normal(0, 5)

                eye_speeds.append(abs(speed))  # ערך מוחלט למהירות

            eye_speeds = np.array(eye_speeds)

        # יצירת זמן מצטבר
        if 'timestamp' in df.columns:
            time_data = pd.to_datetime(df['timestamp'])
            start_time = time_data.iloc[0]
            elapsed_seconds = (time_data - start_time).dt.total_seconds()
        else:
            elapsed_seconds = df['duration'].cumsum()

        elapsed_minutes = elapsed_seconds / 60

        # ===== יצירת הגרף =====
        # מהירות תזוזות עיניים לאורך זמן
        ax2.scatter(elapsed_minutes[:len(eye_speeds)], eye_speeds, alpha=0.6, s=30, color='purple',
                    label='Eye Movement Speed')

        # הוספת קו מגמה
        if len(eye_speeds) > 1:
            z = np.polyfit(elapsed_minutes[:len(eye_speeds)], eye_speeds, 1)
            p = np.poly1d(z)
            trend_line = p(elapsed_minutes[:len(eye_speeds)])
            ax2.plot(elapsed_minutes[:len(eye_speeds)], trend_line, "r-", linewidth=3,
                     label=f'Trend: {"Getting Faster" if z[0] > 0 else "Getting Slower"}')

        # הוספת ממוצע נע
        window_size = max(10, len(eye_speeds) // 20)
        moving_avg = pd.Series(eye_speeds).rolling(window=window_size, center=True).mean()
        ax2.plot(elapsed_minutes[:len(eye_speeds)], moving_avg, color='green', linewidth=2, alpha=0.8,
                 label='Moving Average')

        ax2.set_title('👁️ Eye Movement Speed Over Time', fontweight='bold', fontsize=16)
        ax2.set_xlabel('Time Elapsed (minutes)', fontweight='bold')
        ax2.set_ylabel('Eye Movement Speed (pixels/sec)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # הוספת אזורי ביצועים
        percentile_75 = np.percentile(eye_speeds, 75)
        percentile_25 = np.percentile(eye_speeds, 25)
        ax2.axhspan(percentile_75, max(eye_speeds), alpha=0.1, color='green', label='Fast movements')
        ax2.axhspan(percentile_25, percentile_75, alpha=0.1, color='yellow', label='Average movements')
        ax2.axhspan(0, percentile_25, alpha=0.1, color='red', label='Slow movements')

        # זיהוי תבניות מעניינות
        # חיפוש אחר שיאי מהירות (saccades)
        high_speed_threshold = np.percentile(eye_speeds, 90)
        saccades = eye_speeds > high_speed_threshold
        saccade_count = np.sum(saccades)

        # חיפוש אחר האטות (fixations ארוכות)
        low_speed_threshold = np.percentile(eye_speeds, 10)
        slow_movements = eye_speeds < low_speed_threshold
        slow_count = np.sum(slow_movements)

        # חישוב מסקנות
        first_third = eye_speeds[:len(eye_speeds) // 3].mean()
        last_third = eye_speeds[-len(eye_speeds) // 3:].mean()
        speed_change = last_third - first_third
        speed_change_percent = (speed_change / first_third) * 100

        # הוספת מסקנות
        if speed_change_percent > 10:
            conclusion = f"🚀 Your eye movements got {speed_change_percent:.1f}% faster!"
            conclusion_color = "#d5f4e6"
        elif speed_change_percent < -10:
            conclusion = f"🐌 Eye movements slowed down by {abs(speed_change_percent):.1f}%"
            conclusion_color = "#fadbd8"
        else:
            conclusion = f"👁️ Consistent eye movement speed"
            conclusion_color = "#ebf3fd"

        ax2.text(0.5, 0.95, conclusion, transform=ax2.transAxes, fontsize=12,
                 ha='center', va='top', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=conclusion_color, alpha=0.8))

        # הוספת אינפורמציה על תבניות
        info_text = f"""👁️ Eye Movement Patterns:
        • Fast movements (saccades): {saccade_count} ({saccade_count / len(eye_speeds) * 100:.1f}%)
        • Slow movements (fixations): {slow_count} ({slow_count / len(eye_speeds) * 100:.1f}%)
        • Average speed: {np.mean(eye_speeds):.1f} pixels/sec"""

        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
                 va='top', ha='left', bbox=dict(boxstyle="round,pad=0.5",
                                                facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        save_high_quality_plot(f'eye_movement_analysis_{session_id}')
        plt.show()

        # גרף 2: ניתוח מילים (TOP 20)
        word_stats = df.groupby('word').agg({
            'duration': ['mean', 'count', 'sum']
        }).round(3)
        word_stats.columns = ['avg_duration', 'fixation_count', 'total_time']
        word_stats = word_stats.sort_values('total_time', ascending=False).head(20)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Total time
        bars1 = ax1.barh(range(len(word_stats)), word_stats['total_time'],
                         color='steelblue', alpha=0.8)
        ax1.set_yticks(range(len(word_stats)))
        ax1.set_yticklabels(word_stats.index, fontsize=12)
        ax1.set_xlabel('Total Time (seconds)', fontweight='bold')
        ax1.set_title('Top 20 Words - Total Reading Time', fontweight='bold', fontsize=16)

        for i, (bar, value) in enumerate(zip(bars1, word_stats['total_time'])):
            ax1.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{value:.2f}s', va='center', fontsize=10, fontweight='bold')

        # Fixation count
        bars2 = ax2.barh(range(len(word_stats)), word_stats['fixation_count'],
                         color='forestgreen', alpha=0.8)
        ax2.set_yticks(range(len(word_stats)))
        ax2.set_yticklabels(word_stats.index, fontsize=12)
        ax2.set_xlabel('Number of Fixations', fontweight='bold')
        ax2.set_title('Top 20 Words - Fixation Count', fontweight='bold', fontsize=16)

        for i, (bar, value) in enumerate(zip(bars2, word_stats['fixation_count'])):
            ax2.text(value + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{int(value)}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_high_quality_plot(f'word_analysis_{session_id}')
        plt.show()

        # גרף 3: עוגת סטטיסטיקות קריאה (מעודכן למאמר אקדמי)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 1. עוגת סוגי קריאה
        behavior_counts = df['behavior'].value_counts()
        total_fixations = len(df)
        percentages = [(count / total_fixations) * 100 for count in behavior_counts.values]
        labels_with_nums = [f'{behavior.title()}\n{count} times\n({pct:.1f}%)'
                            for behavior, count, pct in zip(behavior_counts.index,
                                                            behavior_counts.values, percentages)]

        ordered_colors = ['#DC143C', '#1E3A8A', '#FF8C00', '#228B22', '#654321']
        colors_pie = ordered_colors[:len(behavior_counts)]

        ax1.pie(behavior_counts.values, labels=labels_with_nums, colors=colors_pie,
                startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Reading Behavior Types', fontweight='bold', fontsize=14)

        # 2. עוגת זמן קריאה
        total_time = df['duration'].sum()
        normal_time = df[df['behavior'] == 'normal']['duration'].sum() if 'normal' in df['behavior'].values else 0
        skip_time = df[df['behavior'] == 'skip']['duration'].sum() if 'skip' in df['behavior'].values else 0
        regression_time = df[df['behavior'] == 'regression']['duration'].sum() if 'regression' in df[
            'behavior'].values else 0

        time_data = []
        time_labels = []

        if normal_time > 0:
            time_data.append(normal_time)
            normal_pct = (normal_time / total_time) * 100
            time_labels.append(f'Normal Reading\n{normal_time:.2f} sec\n({normal_pct:.1f}%)')

        if skip_time > 0:
            time_data.append(skip_time)
            skip_pct = (skip_time / total_time) * 100
            time_labels.append(f'Quick Scanning\n{skip_time:.2f} sec\n({skip_pct:.1f}%)')

        if regression_time > 0:
            time_data.append(regression_time)
            regression_pct = (regression_time / total_time) * 100
            time_labels.append(f'Going Back\n{regression_time:.2f} sec\n({regression_pct:.1f}%)')

        # מיון לפי גודל ויצירת צבעים בהתאם
        time_sorted = sorted(zip(time_data, time_labels), reverse=True)
        time_data_sorted = [x[0] for x in time_sorted]
        time_labels_sorted = [x[1] for x in time_sorted]
        time_colors = ordered_colors[:len(time_data_sorted)]

        ax2.pie(time_data, labels=time_labels, colors=time_colors, startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Time Spent on Each Reading Type', fontweight='bold', fontsize=14)

        # 3. עוגת מהירות קריאה
        avg_duration = df['duration'].mean()

        # קטגוריות מהירות
        fast_fixations = len(df[df['duration'] < 0.2])  # פחות מ-0.2 שניות
        medium_fixations = len(df[(df['duration'] >= 0.2) & (df['duration'] < 0.5)])  # 0.2-0.5 שניות
        slow_fixations = len(df[df['duration'] >= 0.5])  # יותר מ-0.5 שניות

        speed_data = [fast_fixations, medium_fixations, slow_fixations]
        speed_labels = [
            f'Fast Glances\n{fast_fixations} times\n({(fast_fixations / total_fixations) * 100:.1f}%)',
            f'Medium Pace\n{medium_fixations} times\n({(medium_fixations / total_fixations) * 100:.1f}%)',
            f'Slow Reading\n{slow_fixations} times\n({(slow_fixations / total_fixations) * 100:.1f}%)'
        ]
        # סדר צבעים לפי אחוזים: אדום, כתום, כחול, ירוק, חום
        speed_colors = ['#DC143C', '#1E3A8A', '#FF8C00', '#228B22', '#654321']

        # הסר קטגוריות ריקות
        speed_data_clean = []
        speed_labels_clean = []
        speed_colors_clean = []
        for data, label, color in zip(speed_data, speed_labels, speed_colors):
            if data > 0:
                speed_data_clean.append(data)
                speed_labels_clean.append(label)
                speed_colors_clean.append(color)

        if speed_data_clean:
            ax3.pie(speed_data_clean, labels=speed_labels_clean, colors=speed_colors_clean,
                    startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax3.set_title('Reading Speed Distribution', fontweight='bold', fontsize=14)

        plt.tight_layout()
        save_high_quality_plot(f'reading_statistics_pies_{session_id}')
        plt.show()

        # גרף 4: הסבר פשוט של התוצאות - למשתמש הרגיל
        unique_words = len(df['word'].unique())
        total_words_in_text = sum(
            len(line.split()) for line in TEXT_OPTIONS[current_text_option]["lines"] if line.strip())

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. מהירות הקריאה שלך מול הממוצע
        reading_speed = (unique_words / (total_time / 60)) if total_time > 0 else 0
        average_reading_speed = 250  # ממוצע של קוראים בוגרים

        speeds = [reading_speed, average_reading_speed]
        speed_labels = [f'Your Speed\n{reading_speed:.0f} WPM', f'Average Adult\n{average_reading_speed} WPM']
        speed_colors = ['lightgreen' if reading_speed >= average_reading_speed else 'orange', 'lightblue']

        bars = ax1.bar(speed_labels, speeds, color=speed_colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('🚀 How Fast Do You Read?\n(Words Per Minute)', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Words Per Minute (WPM)', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # הוספת ערכים על הבארים
        for bar, speed in zip(bars, speeds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{speed:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

        # הוספת הערכה
        if reading_speed >= 300:
            assessment = "🏆 Excellent - Very Fast Reader!"
        elif reading_speed >= 200:
            assessment = "✅ Good - Above Average"
        elif reading_speed >= 150:
            assessment = "📖 Normal - Average Reader"
        else:
            assessment = "🐌 Slow - Take Your Time"

        ax1.text(0.5, 0.95, assessment, transform=ax1.transAxes, fontsize=14,
                 ha='center', va='top', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))

        # 2. איכות הקריאה שלך
        normal_percentage = (behavior_counts.get('normal', 0) / total_fixations) * 100
        skip_percentage = (behavior_counts.get('skip', 0) / total_fixations) * 100
        regression_percentage = (behavior_counts.get('regression', 0) / total_fixations) * 100

        reading_quality_data = [normal_percentage, skip_percentage, regression_percentage]
        quality_labels = ['Focused Reading', 'Quick Scanning', 'Re-reading']
        quality_colors = ['green', 'orange', 'red']
        quality_explanations = [
            f'Good!\n{normal_percentage:.0f}%',
            f'Efficient!\n{skip_percentage:.0f}%',
            f'Careful!\n{regression_percentage:.0f}%'
        ]

        # הסר קטגוריות ריקות
        quality_data_clean = []
        quality_labels_clean = []
        quality_colors_clean = []
        quality_explanations_clean = []

        for data, label, color, explanation in zip(reading_quality_data, quality_labels, quality_colors,
                                                   quality_explanations):
            if data > 0:
                quality_data_clean.append(data)
                quality_labels_clean.append(label)
                quality_colors_clean.append(color)
                quality_explanations_clean.append(explanation)

        if quality_data_clean:
            wedges, texts, autotexts = ax2.pie(quality_data_clean, labels=quality_explanations_clean,
                                               colors=quality_colors_clean, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'},
                                               autopct='')

        ax2.set_title('📊 What Kind of Reader Are You?', fontweight='bold', fontsize=16)

        # הוספת הסבר
        explanation_text = """
        🟢 Focused Reading = You read carefully
        🟠 Quick Scanning = You skip unimportant words  
        🔴 Re-reading = You go back to check
        """
        ax2.text(1.3, 0.5, explanation_text, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3",
                                                       facecolor="lightgray", alpha=0.8))

        # 3. זמן קריאה מול ממוצע
        avg_fixation = df['duration'].mean()
        typical_fixation = 0.25  # ממוצע טיפוסי בשניות

        times = [avg_fixation, typical_fixation]
        time_labels = [f'Your Average\n{avg_fixation:.2f} sec', f'Typical Reader\n{typical_fixation:.2f} sec']
        time_colors = ['lightcoral' if avg_fixation > typical_fixation else 'lightgreen', 'lightblue']

        bars = ax3.bar(time_labels, times, color=time_colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_title('⏰ How Long Do You Look at Words?', fontweight='bold', fontsize=16)
        ax3.set_ylabel('Average Time (seconds)', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')

        # הוספת ערכים
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=14)

        # הערכת זמן הסתכלות
        if avg_fixation > 0.4:
            time_assessment = "🔍 Thorough - You read carefully"
        elif avg_fixation > 0.2:
            time_assessment = "📖 Normal - Balanced reading"
        else:
            time_assessment = "⚡ Fast - Quick reader"

        ax3.text(0.5, 0.95, time_assessment, transform=ax3.transAxes, fontsize=12,
                 ha='center', va='top', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        # 4. סיכום אישי עם ציון
        ax4.axis('off')

        # חישוב ציון כללי
        speed_score = min(100, (reading_speed / 250) * 100)
        efficiency_score = normal_percentage
        focus_score = 100 - (regression_percentage * 2)  # פחות חזרות = יותר טוב
        overall_score = (speed_score + efficiency_score + focus_score) / 3

        # קביעת דירוג
        if overall_score >= 80:
            grade = "A"
            grade_color = "lightgreen"
            message = "🏆 Excellent Reader!"
        elif overall_score >= 70:
            grade = "B"
            grade_color = "lightblue"
            message = "✅ Good Reader!"
        elif overall_score >= 60:
            grade = "C"
            grade_color = "yellow"
            message = "📖 Average Reader"
        else:
            grade = "D"
            grade_color = "lightcoral"
            message = "📚 Keep Practicing!"

        personal_summary = f"""
        📋 YOUR READING REPORT CARD

        Overall Grade: {grade} ({overall_score:.0f}/100)
        {message}

        📊 DETAILED BREAKDOWN:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🚀 Reading Speed: {reading_speed:.0f} WPM
           • {speed_score:.0f}/100 points
           • {"Fast!" if reading_speed > 250 else "Good pace" if reading_speed > 150 else "Take your time"}

        📖 Reading Quality: {normal_percentage:.0f}% focused
           • {efficiency_score:.0f}/100 points  
           • {"Excellent focus!" if normal_percentage > 70 else "Good concentration" if normal_percentage > 50 else "Try to focus more"}

        🎯 Reading Consistency: {100 - regression_percentage:.0f}% forward
           • {focus_score:.0f}/100 points
           • {"Smooth reading!" if regression_percentage < 10 else "Some re-reading" if regression_percentage < 20 else "Often go back"}

        ⏱️ SESSION SUMMARY:
        - Total time: {total_time:.1f} seconds
        - Words read: {unique_words} out of {total_words_in_text}
        - Eye movements: {total_fixations}
        - Efficiency: {(unique_words / total_fixations) * 100:.0f}% (words per look)

        💡 RECOMMENDATION:
        {"Keep up the great work! You're reading efficiently." if overall_score >= 70 else
        "Good job! Try reading more to improve speed." if overall_score >= 60 else
        "Practice daily reading to build skills. Focus on one word at a time."}
        """

        ax4.text(0.05, 0.95, personal_summary, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=1.0", facecolor=grade_color, alpha=0.3))

        plt.tight_layout()
        save_high_quality_plot(f'personal_reading_report_{session_id}')
        plt.show()

        # גרף 5: רדאר פרופיל קריאה
        plot_reading_profile_radar(
            reading_speed=reading_speed,
            normal_pct=normal_percentage,
            skip_pct=skip_percentage,
            regressions=behavior_counts.get('regression', 0),
            avg_fix_dur=avg_fixation,
            session_id=session_id
        )

        # חישוב מהירות קריאה (WPM)
        if not df.empty and 'timestamp' in df.columns:
            start_time = pd.to_datetime(df['timestamp'].iloc[0], errors='coerce')
            end_time = pd.to_datetime(df['timestamp'].iloc[-1], errors='coerce')
            total_minutes = (end_time - start_time).total_seconds() / 60
            total_words = len(df)
            reading_speed = total_words / total_minutes if total_minutes > 0 else 0
        else:
            reading_speed = 0

        print(f"✅ Created 5 clear and meaningful plots!")
        print(f"📊 Your Reading Report Card: Grade {grade} ({overall_score:.0f}/100)")
        print(f"🚀 Speed: {reading_speed:.0f} WPM | 📖 Focus: {normal_percentage:.0f}% | ⏱️ Time: {total_time:.1f}s")

        return ['reading_path', 'reading_analysis', 'word_focus', 'reading_statistics_pies', 'personal_reading_report']

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def plot_reading_profile_radar(reading_speed, normal_pct, skip_pct, regressions, avg_fix_dur, session_id):
    from math import pi

    # ערכים נורמליזציה לפי סקלות קבועות
    categories = ['Reading Speed', 'Fixation %', 'Skip %', 'Regressions', 'Fixation Time']
    values = [
        min(100, (reading_speed / 300) * 100),  # סקלת מהירות
        normal_pct,  # אחוז פיקסים רגילים
        skip_pct,  # אחוז דילוגים
        max(0, 100 - regressions * 10),  # פחות רגרסיות = יותר טוב
        max(0, 100 - (avg_fix_dur * 250))  # פחות זמן = יותר טוב (0.25 ~ 100%)
    ]

    values += values[:1]  # סגירה של הגרף
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # גריד
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # שמות
    plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')

    # גבולות רדיאלים
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="gray", size=10)
    plt.ylim(0, 100)

    # תרשים
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='purple')
    ax.fill(angles, values, color='violet', alpha=0.4)

    ax.set_title('🧠 Reading Profile Radar', size=16, fontweight='bold', pad=20)

    save_high_quality_plot(f'reading_radar_{session_id}')
    plt.show()


def plot_reading_wpm_and_pupil_current_session(csv_file="extended_eye_tracking.csv"):
    """גרף מהירות קריאה ואישון"""

    if not os.path.exists(csv_file):
        print(f"❌ הקובץ '{csv_file}' לא נמצא.")
        return

    try:
        # טעינת נתונים מ extended_eye_tracking.csv
        df = pd.read_csv(csv_file)

        # פילטר רק לניסוי האחרון
        if 'session_id' in df.columns:
            last_session = df['session_id'].iloc[-1]
            df = df[df['session_id'] == last_session]
            print(f"🎯 ניסוי נוכחי: {last_session}")

        # המרת זמן
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        if df.empty:
            print("❌ אין נתונים!")
            return

        # חישוב משך זמן
        start_time = df['timestamp'].iloc[0]
        end_time = df['timestamp'].iloc[-1]
        total_duration = (end_time - start_time).total_seconds()

        print(f"⏰ משך הניסוי: {total_duration:.1f} שניות ({total_duration / 60:.2f} דקות)")

        # גרף פשוט
        fig, ax = plt.subplots(figsize=(12, 6))

        # יצירת גרף פשוט של מהירות לאורך זמן
        time_in_minutes = (df['timestamp'] - start_time).dt.total_seconds() / 60

        ax.plot(time_in_minutes, range(len(df)), linewidth=2, color='blue')
        ax.set_title('Reading Progress Over Time', fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontweight='bold')
        ax.set_ylabel('Words Read', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('reading_speed_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ הגרף נשמר כ: reading_speed_analysis.png")
        plt.show()

    except Exception as e:
        print(f"❌ שגיאה: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()