from __future__ import annotations

import random
import re
import threading
import time


class GameState:
    def __init__(self) -> None:
        self.target = "Loading model..."
        self.won_at: float | None = None
        self.round_id = 0
        self.labels: tuple[str, ...] = ()
        self._lock = threading.Lock()

    def start(self, labels: tuple[str, ...] | list[str]) -> None:
        available_labels = tuple(labels)
        if not available_labels:
            return

        with self._lock:
            choices = [label for label in available_labels if label != self.target]
            self.target = random.choice(choices or available_labels)
            self.labels = available_labels
            self.won_at = None
            self.round_id += 1

    def snapshot(self) -> tuple[str, bool, int]:
        with self._lock:
            return self.target, self.won_at is not None, self.round_id

    def should_flash_green(self, duration_seconds: float) -> bool:
        with self._lock:
            return (
                self.won_at is not None
                and time.monotonic() - self.won_at < duration_seconds
            )

    def check_guess(self, round_id: int, guess: str) -> bool:
        normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        with self._lock:
            if round_id != self.round_id or self.won_at is not None:
                return False
            if normalize(guess) != normalize(self.target):
                return False
            self.won_at = time.monotonic()
            return True

    def skip_round(self) -> bool:
        with self._lock:
            return self._start_next_round()

    def advance_if_won(self, delay_seconds: float) -> bool:
        with self._lock:
            if self.won_at is None or time.monotonic() - self.won_at < delay_seconds:
                return False
            return self._start_next_round()

    def _start_next_round(self) -> bool:
        if not self.labels:
            return False
        choices = [label for label in self.labels if label != self.target]
        self.target = random.choice(choices or self.labels)
        self.won_at = None
        self.round_id += 1
        return True