from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import config


class GestureState(str, Enum):
    NO_HAND = "NO HAND"
    IDLE = "IDLE"
    DRAW = "DRAW"
    HOVER = "HOVER"
    CLEAR = "CLEAR"


class HandTracker:
    """
    Tracks one hand, classifies drawing gestures, and smooths the index-finger
    tip using an exponential moving average.
    """

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

        self._pointer_position: Optional[tuple[int, int]] = None
        self._smoothed_pointer: Optional[np.ndarray] = None
        self._gesture_state = GestureState.NO_HAND
        self._landmarks = None
        self._frame_width = 0
        self._frame_height = 0

        self._open_hand_frames = 0

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _joint_angle(a, b, c) -> float:
        """
        Returns angle ABC in degrees.
        """

        ba = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
        bc = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)

        denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denominator < 1e-8:
            return 0.0

        cosine = float(np.dot(ba, bc) / denominator)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def _is_finger_extended(
        self,
        landmarks,
        mcp_index: int,
        pip_index: int,
        dip_index: int,
        tip_index: int,
    ) -> bool:
        wrist = landmarks[0]
        mcp = landmarks[mcp_index]
        pip = landmarks[pip_index]
        dip = landmarks[dip_index]
        tip = landmarks[tip_index]

        pip_angle = self._joint_angle(mcp, pip, dip)
        dip_angle = self._joint_angle(pip, dip, tip)

        tip_distance = self._distance(wrist, tip)
        pip_distance = self._distance(wrist, pip)

        straight = (
            pip_angle >= config.FINGER_STRAIGHT_ANGLE_DEGREES
            and dip_angle >= config.FINGER_STRAIGHT_ANGLE_DEGREES
        )
        farther_from_wrist = (
            tip_distance > pip_distance + config.FINGER_DISTANCE_MARGIN
        )

        return straight and farther_from_wrist

    def _classify_gesture(self, landmarks) -> GestureState:
        index_extended = self._is_finger_extended(
            landmarks, 5, 6, 7, 8
        )
        middle_extended = self._is_finger_extended(
            landmarks, 9, 10, 11, 12
        )
        ring_extended = self._is_finger_extended(
            landmarks, 13, 14, 15, 16
        )
        pinky_extended = self._is_finger_extended(
            landmarks, 17, 18, 19, 20
        )

        extended_count = sum(
            (
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended,
            )
        )

        if extended_count >= config.OPEN_HAND_MIN_EXTENDED_FINGERS:
            self._open_hand_frames += 1
        else:
            self._open_hand_frames = 0

        if self._open_hand_frames >= config.OPEN_HAND_STABLE_FRAMES:
            return GestureState.CLEAR

        if index_extended and middle_extended:
            return GestureState.HOVER

        if index_extended and not middle_extended:
            return GestureState.DRAW

        return GestureState.IDLE

    def _update_smoothed_pointer(self, x: int, y: int) -> None:
        raw = np.array([float(x), float(y)], dtype=np.float32)

        if self._smoothed_pointer is None:
            self._smoothed_pointer = raw
        else:
            alpha = config.POINTER_EMA_ALPHA
            self._smoothed_pointer = (
                alpha * raw + (1.0 - alpha) * self._smoothed_pointer
            )

        smoothed_x = int(
            np.clip(round(self._smoothed_pointer[0]), 0, self._frame_width - 1)
        )
        smoothed_y = int(
            np.clip(round(self._smoothed_pointer[1]), 0, self._frame_height - 1)
        )
        self._pointer_position = (smoothed_x, smoothed_y)

    def process_frame(self, frame: np.ndarray) -> GestureState:
        """
        Processes a BGR OpenCV frame and updates the pointer and gesture state.
        """

        if frame is None or frame.size == 0:
            self._gesture_state = GestureState.NO_HAND
            self._pointer_position = None
            self._smoothed_pointer = None
            self._landmarks = None
            return self._gesture_state

        self._frame_height, self._frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            self._gesture_state = GestureState.NO_HAND
            self._pointer_position = None
            self._smoothed_pointer = None
            self._landmarks = None
            self._open_hand_frames = 0
            return self._gesture_state

        self._landmarks = result.multi_hand_landmarks[0].landmark

        index_tip = self._landmarks[8]
        pointer_x = int(
            np.clip(index_tip.x * self._frame_width, 0, self._frame_width - 1)
        )
        pointer_y = int(
            np.clip(index_tip.y * self._frame_height, 0, self._frame_height - 1)
        )

        self._update_smoothed_pointer(pointer_x, pointer_y)
        self._gesture_state = self._classify_gesture(self._landmarks)

        return self._gesture_state

    def get_pointer_position(self) -> Optional[tuple[int, int]]:
        return self._pointer_position

    def get_gesture_state(self) -> GestureState:
        return self._gesture_state

    def close(self) -> None:
        self._hands.close()