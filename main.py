from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from canvas_manager import CanvasManager
from guesser_model import DrawingGuesser
from hand_tracker import GestureState, HandTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("ai_pictionary")


@dataclass
class PredictionState:
    predictions: list[tuple[str, float]] = field(default_factory=list)
    status: str = "Loading drawing model..."
    backend: str = ""
    updated_at: float = 0.0
    lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    def set_status(self, status: str, backend: str = "") -> None:
        with self.lock:
            self.status = status
            if backend:
                self.backend = backend

    def set_predictions(
        self,
        predictions: list[tuple[str, float]],
        backend: str,
    ) -> None:
        with self.lock:
            self.predictions = list(predictions)
            self.backend = backend
            self.status = "Watching your drawing..."
            self.updated_at = time.monotonic()

    def clear(self) -> None:
        with self.lock:
            self.predictions = []
            self.status = "Draw something..."
            self.updated_at = time.monotonic()

    def snapshot(
        self,
    ) -> tuple[list[tuple[str, float]], str, str]:
        with self.lock:
            return (
                list(self.predictions),
                self.status,
                self.backend,
            )


class GuesserWorker(threading.Thread):
    def __init__(
        self,
        canvas: CanvasManager,
        prediction_state: PredictionState,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            name="drawing-guesser",
            daemon=True,
        )
        self.canvas = canvas
        self.prediction_state = prediction_state
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            self.prediction_state.set_status(
                "Loading AI model in background..."
            )
            guesser = DrawingGuesser()

            status = "Draw something..."
            if guesser.initialization_warning:
                status = guesser.initialization_warning

            self.prediction_state.set_status(
                status,
                guesser.backend,
            )

            while not self.stop_event.is_set():
                iteration_started = time.monotonic()

                classifier_image = self.canvas.get_classifier_image()

                if classifier_image is None:
                    self.prediction_state.clear()
                else:
                    try:
                        predictions = guesser.predict(classifier_image)
                        self.prediction_state.set_predictions(
                            predictions,
                            guesser.backend,
                        )
                    except Exception as inference_error:
                        LOGGER.exception("Drawing inference failed")
                        self.prediction_state.set_status(
                            "Inference error: "
                            f"{type(inference_error).__name__}",
                            guesser.backend,
                        )

                elapsed = time.monotonic() - iteration_started
                remaining = max(
                    0.0,
                    config.INFERENCE_INTERVAL_SECONDS - elapsed,
                )
                self.stop_event.wait(remaining)

        except Exception as initialization_error:
            LOGGER.exception("Guesser worker initialization failed")
            self.prediction_state.set_status(
                "AI unavailable: "
                f"{type(initialization_error).__name__}"
            )


class MouseState:
    def __init__(self) -> None:
        self.x = -1
        self.y = -1
        self.clear_requested = threading.Event()
        self.button_rect = (0, 0, 0, 0)

    def update_button_rect(
        self,
        frame_width: int,
        frame_height: int,
    ) -> None:
        x2 = frame_width - config.CLEAR_BUTTON_MARGIN
        y2 = frame_height - config.CLEAR_BUTTON_MARGIN
        x1 = x2 - config.CLEAR_BUTTON_WIDTH
        y1 = y2 - config.CLEAR_BUTTON_HEIGHT
        self.button_rect = (x1, y1, x2, y2)

    def is_over_button(self) -> bool:
        x1, y1, x2, y2 = self.button_rect
        return x1 <= self.x <= x2 and y1 <= self.y <= y2


def mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    userdata: MouseState,
) -> None:
    del flags

    userdata.x = x
    userdata.y = y

    if event == cv2.EVENT_LBUTTONDOWN and userdata.is_over_button():
        userdata.clear_requested.set()


def create_camera() -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(
            config.CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(config.CAMERA_INDEX)
    else:
        capture = cv2.VideoCapture(config.CAMERA_INDEX)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return capture


def overlay_canvas(
    frame: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Overlays the drawing mask onto the BGR camera frame with transparency.
    """
    if mask is None or not np.any(mask):
        return frame

    colored_canvas = np.zeros_like(frame)
    colored_canvas[mask > 0] = config.CANVAS_OVERLAY_COLOR_BGR

    overlay_pixels = mask > 0
    blended = frame.copy()
    blended[overlay_pixels] = cv2.addWeighted(
        frame[overlay_pixels],
        1.0 - config.CANVAS_OVERLAY_ALPHA,
        colored_canvas[overlay_pixels],
        config.CANVAS_OVERLAY_ALPHA,
        0,
    )
    return blended


def draw_ui(
    frame: np.ndarray,
    gesture: GestureState,
    pointer: tuple[int, int] | None,
    prediction_state: PredictionState,
    mouse_state: MouseState,
) -> None:
    height, width = frame.shape[:2]
    mouse_state.update_button_rect(width, height)

    # Pointer indicator
    if pointer is not None:
        color = config.POINTER_IDLE_COLOR_BGR
        if gesture == GestureState.DRAW:
            color = config.POINTER_DRAW_COLOR_BGR
        elif gesture == GestureState.HOVER:
            color = config.POINTER_HOVER_COLOR_BGR

        cv2.circle(frame, pointer, config.POINTER_RADIUS, color, -1)
        cv2.circle(frame, pointer, config.POINTER_RADIUS + 2, (255, 255, 255), 1)

    # Top-Left Prediction Box
    predictions, status, backend = prediction_state.snapshot()
    p_x1 = config.PANEL_MARGIN
    p_y1 = config.PANEL_MARGIN
    p_x2 = p_x1 + config.PREDICTION_PANEL_WIDTH
    p_y2 = p_y1 + config.PREDICTION_PANEL_HEIGHT

    cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), config.PANEL_BACKGROUND_BGR, -1)
    cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), config.PANEL_BORDER_BGR, 2)

    cv2.putText(
        frame,
        f"State: {gesture.value}",
        (p_x1 + 10, p_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        (200, 200, 200),
        config.FONT_THICKNESS,
    )

    y_offset = p_y1 + 55
    if predictions:
        for label, prob in predictions:
            text = f"{label.title()}: {prob:.1f}%"
            cv2.putText(
                frame,
                text,
                (p_x1 + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )
            y_offset += 30
    else:
        cv2.putText(
            frame,
            status,
            (p_x1 + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # Bottom-Right Clear Button
    x1, y1, x2, y2 = mouse_state.button_rect
    btn_color = (
        config.CLEAR_BUTTON_HOVER_BGR
        if mouse_state.is_over_button()
        else config.CLEAR_BUTTON_BACKGROUND_BGR
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), btn_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.putText(
        frame,
        "Clear (C)",
        (x1 + 22, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        config.CLEAR_BUTTON_TEXT_BGR,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    LOGGER.info("Starting AI Pictionary...")
    capture = create_camera()
    if not capture.isOpened():
        LOGGER.error("Failed to open webcam capture.")
        return

    ret, frame = capture.read()
    if not ret:
        LOGGER.error("Failed to read frame from webcam.")
        capture.release()
        return

    height, width = frame.shape[:2]
    canvas = CanvasManager(width=width, height=height)
    hand_tracker = HandTracker()
    prediction_state = PredictionState()
    mouse_state = MouseState()

    stop_event = threading.Event()
    worker = GuesserWorker(canvas, prediction_state, stop_event)
    worker.start()

    cv2.namedWindow(config.WINDOW_NAME)
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback, mouse_state)

    try:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            if config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            gesture = hand_tracker.process_frame(frame)
            pointer = hand_tracker.get_pointer_position()

            if gesture == GestureState.CLEAR or mouse_state.clear_requested.is_set():
                canvas.clear()
                prediction_state.clear()
                mouse_state.clear_requested.clear()
            else:
                is_drawing = (gesture == GestureState.DRAW)
                canvas.update_pointer(pointer, drawing=is_drawing)

            output_frame = overlay_canvas(frame, canvas.get_mask_snapshot())
            draw_ui(output_frame, gesture, pointer, prediction_state, mouse_state)

            cv2.imshow(config.WINDOW_NAME, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            elif key in config.CLEAR_KEYS:
                canvas.clear()
                prediction_state.clear()

    finally:
        LOGGER.info("Shutting down...")
        stop_event.set()
        worker.join(timeout=2.0)
        hand_tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()