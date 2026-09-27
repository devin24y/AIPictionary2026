from __future__ import annotations

import math
import threading
from typing import Optional

import cv2
import numpy as np

import config


class CanvasManager:
    """
    Manages a full-resolution binary drawing mask.

    The internal canvas stores white strokes on a black background. Classifier
    images are dynamically cropped around active strokes, padded by 8%, resized
    with preserved aspect ratio, and converted to black strokes on white.
    """

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive")

        self.width = width
        self.height = height

        self._canvas = np.full(
            (height, width),
            config.CANVAS_BACKGROUND_VALUE,
            dtype=np.uint8,
        )
        self._last_point: Optional[tuple[int, int]] = None
        self._lock = threading.RLock()
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def clear(self) -> None:
        with self._lock:
            self._canvas.fill(config.CANVAS_BACKGROUND_VALUE)
            self._last_point = None
            self._version += 1

    def reset_stroke(self) -> None:
        with self._lock:
            self._last_point = None

    def update_pointer(
        self,
        point: Optional[tuple[int, int]],
        drawing: bool,
    ) -> None:
        """
        Adds a smoothed drawing segment when drawing is enabled.

        Resetting the prior point while hovering or when the hand disappears
        prevents accidental lines from being drawn across the screen.
        """

        with self._lock:
            if point is None or not drawing:
                self._last_point = None
                return

            x = int(np.clip(point[0], 0, self.width - 1))
            y = int(np.clip(point[1], 0, self.height - 1))
            current_point = (x, y)

            if self._last_point is None:
                cv2.circle(
                    self._canvas,
                    current_point,
                    max(1, config.DRAWING_LINE_THICKNESS // 2),
                    config.CANVAS_STROKE_VALUE,
                    thickness=-1,
                    lineType=config.DRAWING_LINE_TYPE,
                )
                self._last_point = current_point
                self._version += 1
                return

            distance = math.hypot(
                current_point[0] - self._last_point[0],
                current_point[1] - self._last_point[1],
            )

            if distance < config.MIN_DRAWING_DISTANCE_PIXELS:
                return

            if distance > config.MAX_DRAWING_DISTANCE_PIXELS:
                self._last_point = current_point
                return

            cv2.line(
                self._canvas,
                self._last_point,
                current_point,
                config.CANVAS_STROKE_VALUE,
                thickness=config.DRAWING_LINE_THICKNESS,
                lineType=config.DRAWING_LINE_TYPE,
            )
            self._last_point = current_point
            self._version += 1

    def has_drawing(self) -> bool:
        with self._lock:
            return bool(np.any(self._canvas > 0))

    def get_mask_snapshot(self) -> np.ndarray:
        with self._lock:
            return self._canvas.copy()

    def get_classifier_image(self) -> Optional[np.ndarray]:
        """
        Creates a 224x224 RGB classifier image without shrinking the complete
        camera-sized canvas.

        Processing:
          1. Find the active-stroke bounding box.
          2. Add an 8% margin.
          3. Crop only the relevant region.
          4. Resize while preserving aspect ratio.
          5. Letterbox to 224x224.
          6. Produce binary black strokes on a white background.
        """

        with self._lock:
            snapshot = self._canvas.copy()

        stroke_points = cv2.findNonZero(snapshot)
        if stroke_points is None:
            return None

        x, y, width, height = cv2.boundingRect(stroke_points)

        horizontal_padding = max(
            1,
            int(round(width * config.CLASSIFIER_PADDING_RATIO)),
        )
        vertical_padding = max(
            1,
            int(round(height * config.CLASSIFIER_PADDING_RATIO)),
        )

        x1 = max(0, x - horizontal_padding)
        y1 = max(0, y - vertical_padding)
        x2 = min(self.width, x + width + horizontal_padding)
        y2 = min(self.height, y + height + vertical_padding)

        cropped = snapshot[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        crop_height, crop_width = cropped.shape[:2]
        target_size = config.CLASSIFIER_INPUT_SIZE
        scale = min(
            target_size / float(crop_width),
            target_size / float(crop_height),
        )

        resized_width = max(1, int(round(crop_width * scale)))
        resized_height = max(1, int(round(crop_height * scale)))

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        resized = cv2.resize(
            cropped,
            (resized_width, resized_height),
            interpolation=interpolation,
        )

        # Re-binarize after anti-aliased resizing. A low threshold retains
        # thin edge pixels instead of erasing stroke details.
        binary_strokes = np.where(resized >= 24, 255, 0).astype(np.uint8)

        letterboxed_mask = np.zeros(
            (target_size, target_size),
            dtype=np.uint8,
        )

        offset_x = (target_size - resized_width) // 2
        offset_y = (target_size - resized_height) // 2

        letterboxed_mask[
            offset_y:offset_y + resized_height,
            offset_x:offset_x + resized_width,
        ] = binary_strokes

        classifier_gray = np.full(
            (target_size, target_size),
            config.CLASSIFIER_BACKGROUND_VALUE,
            dtype=np.uint8,
        )
        classifier_gray[letterboxed_mask > 0] = (
            config.CLASSIFIER_STROKE_VALUE
        )

        classifier_rgb = cv2.cvtColor(
            classifier_gray,
            cv2.COLOR_GRAY2RGB,
        )
        return classifier_rgb