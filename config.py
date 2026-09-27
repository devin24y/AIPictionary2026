from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = PROJECT_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Camera configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
MIRROR_CAMERA = True

# Drawing configuration
DRAWING_LINE_THICKNESS = 5
DRAWING_LINE_TYPE = 16  # cv2.LINE_AA
MIN_DRAWING_DISTANCE_PIXELS = 2.0
MAX_DRAWING_DISTANCE_PIXELS = 120.0

# Canvas classifier preprocessing
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_PADDING_RATIO = 0.08

# The internal canvas uses white strokes on a black background.
CANVAS_BACKGROUND_VALUE = 0
CANVAS_STROKE_VALUE = 255

# The classifier receives black strokes on a white background, matching a
# conventional drawing on white paper.
CLASSIFIER_BACKGROUND_VALUE = 255
CLASSIFIER_STROKE_VALUE = 0

# Exponential moving average.
# Higher values follow the finger more quickly; lower values are smoother.
POINTER_EMA_ALPHA = 0.32

# Hand tracking
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.60
MODEL_COMPLEXITY = 1

# Finger extension detection
FINGER_STRAIGHT_ANGLE_DEGREES = 155.0
FINGER_DISTANCE_MARGIN = 0.015
OPEN_HAND_MIN_EXTENDED_FINGERS = 4
OPEN_HAND_STABLE_FRAMES = 4
_GESTURE_COOLDOWN_SECONDS = 1.0

# Guesser
INFERENCE_FPS = 1.5
INFERENCE_INTERVAL_SECONDS = 1.0 / INFERENCE_FPS
TOP_K_PREDICTIONS = 3

# OpenCLIP provides zero-shot classification against the QuickDraw labels.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED_WEIGHTS = "openai"
CLIP_PROMPT_TEMPLATES = (
    "a simple black line drawing of a {}",
    "a hand-drawn sketch of a {}",
    "a doodle of a {}",
)

# Constrain CPU inference so that it does not starve OpenCV or MediaPipe.
TORCH_CPU_THREADS = 2

# UI configuration
WINDOW_NAME = "AI Pictionary"
CANVAS_OVERLAY_COLOR_BGR = (40, 225, 255)
CANVAS_OVERLAY_ALPHA = 0.88

POINTER_DRAW_COLOR_BGR = (40, 40, 255)
POINTER_HOVER_COLOR_BGR = (0, 220, 255)
POINTER_IDLE_COLOR_BGR = (180, 180, 180)
POINTER_RADIUS = 10

PREDICTION_PANEL_WIDTH = 390
PREDICTION_PANEL_HEIGHT = 170
PANEL_MARGIN = 15
PANEL_BACKGROUND_BGR = (20, 20, 20)
PANEL_BORDER_BGR = (90, 90, 90)

CLEAR_BUTTON_WIDTH = 150
CLEAR_BUTTON_HEIGHT = 48
CLEAR_BUTTON_MARGIN = 18
CLEAR_BUTTON_BACKGROUND_BGR = (35, 35, 180)
CLEAR_BUTTON_HOVER_BGR = (45, 45, 230)
CLEAR_BUTTON_TEXT_BGR = (255, 255, 255)

FONT_SCALE = 0.65
FONT_THICKNESS = 2

# Keyboard controls
QUIT_KEYS = (ord("q"), ord("Q"), 27)
CLEAR_KEYS = (ord("c"), ord("C"))