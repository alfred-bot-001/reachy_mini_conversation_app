"""Doze mode: after a long silence Reachy tucks its head and holds still until someone calls its name.

While dozing the microphone stays open and speech is still transcribed, but the backend is told not to answer
(``turn_detection.create_response = false``). Only a transcript containing a wake word wakes Reachy up and gets
answered; everything else is dropped from the conversation.
"""

from __future__ import annotations
import re
import time
from collections.abc import Iterable


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
