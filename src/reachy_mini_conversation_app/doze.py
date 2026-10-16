"""Doze mode: after a long silence Reachy tucks its head and holds still until someone calls its name.

While dozing the microphone stays open and speech is still transcribed, but the backend is told not to answer
(``turn_detection.create_response = false``). Only a transcript containing a wake word wakes Reachy up and gets
answered; everything else is dropped from the conversation.

A loud enough sound (sustained for a moment) also wakes Reachy, but lazily: it slowly lifts its head, stretches
its antennas and looks left and right for whoever is there. If nobody talks to it within a short "drowsy" window it
dozes off again, like a child who was only half woken by a noise.
"""

from __future__ import annotations
import re
import math
import time
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def _normalize(text: str) -> str:
    """Lowercase and drop whitespace and punctuation so "小 白！" and "Xiao Bai" still match."""
    return re.sub(r"[\W_]+", "", text.lower())


class DozeState:
    """Track whether Reachy is dozing and decide when to doze or wake."""

    def __init__(self, doze_after_s: float | None, wake_words: Iterable[str]) -> None:
        """Doze after ``doze_after_s`` seconds without conversation; None disables dozing."""
        self.doze_after_s = doze_after_s
        self.wake_words = tuple(word for word in (_normalize(w) for w in wake_words) if word)
        self.dozing = False
        self.since: float | None = None

    @property
    def enabled(self) -> bool:
        """Return whether dozing is enabled at all."""
        return self.doze_after_s is not None

    def should_doze(self, idle_for_s: float) -> bool:
        """Return whether Reachy should start dozing after ``idle_for_s`` seconds without conversation."""
        return self.enabled and not self.dozing and idle_for_s >= float(self.doze_after_s or 0)

    def is_wake_call(self, transcript: str) -> bool:
        """Return whether a transcript calls Reachy by name."""
        text = _normalize(transcript)
        return bool(text) and any(word in text for word in self.wake_words)

    def doze(self) -> bool:
        """Enter doze; return False when already dozing."""
        if self.dozing:
            return False
        self.dozing = True
        self.since = time.monotonic()
        return True

    def wake(self) -> bool:
        """Leave doze; return False when already awake."""
        if not self.dozing:
            return False
        self.dozing = False
        self.since = None
        return True


def level_dbfs(frame: NDArray[np.int16]) -> float:
    """Return the RMS level of an int16 frame in dBFS (0 = full scale, silence about -96)."""
    if frame.size == 0:
        return -96.0
    rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))
    return 20.0 * math.log10(max(rms, 1.0) / 32768.0)


class LoudnessTrigger:
    """Decide when the room got loud enough, for long enough, to wake a dozing Reachy.

    Loud audio fills a small bucket and quiet audio drains it at half speed, so a sustained noise (talking, a door,
    someone moving around) wakes Reachy while a single click does not. After firing it stays quiet for a cooldown.
    The loudest level heard is kept so it can be logged to tune the threshold for the room.
    """

    def __init__(self, threshold_dbfs: float, hold_s: float, cooldown_s: float = 60.0) -> None:
        """Fire once ``hold_s`` seconds of audio at or above ``threshold_dbfs`` have built up."""
        self.threshold_dbfs = threshold_dbfs
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self._loud_s = 0.0
        self._quiet_until = 0.0
        self.peak_dbfs = -96.0

    def reset(self) -> None:
        """Forget accumulated loudness (e.g. when Reachy falls asleep again)."""
        self._loud_s = 0.0
        self.peak_dbfs = -96.0

    def feed(self, frame: NDArray[np.int16], sample_rate: int, now: float | None = None) -> bool:
        """Account one microphone frame; return True when Reachy should wake up."""
        now = time.monotonic() if now is None else now
        level = level_dbfs(frame)
        self.peak_dbfs = max(self.peak_dbfs, level)
        if now < self._quiet_until:
            return False
        duration = frame.shape[0] / float(sample_rate or 16000)
        if level >= self.threshold_dbfs:
            self._loud_s += duration
        else:
            self._loud_s = max(0.0, self._loud_s - duration / 2)
        if self._loud_s >= self.hold_s:
            self._loud_s = 0.0
            self._quiet_until = now + self.cooldown_s
            return True
        return False
