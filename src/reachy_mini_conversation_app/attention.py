"""Speaker attention: turn the head toward whoever is talking, then let face tracking hold on them.

The daemon's face tracker follows one face at a time, so with several people around it would stay on whoever
it locked first. While the user speaks, the microphone array's direction of arrival (DoA) is read from the
daemon; once a few readings agree, the movement manager compares that direction with the tracked face and,
when the voice comes from somewhere else, turns the head that way so the tracker re-acquires the person now
in front of it.
"""

from __future__ import annotations
import math
import logging
import threading
from typing import Any

import httpx


logger = logging.getLogger(__name__)

# ReSpeaker DoA angles: 0 = left, π/2 = front, π = right.
_FRONT_RAD = math.pi / 2
POLL_INTERVAL_S = 0.1  # the daemon refreshes its DoA cache at 10 Hz
MIN_SPEECH_READINGS = 3  # ~0.3 s of speech before trusting the direction


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def doa_to_yaw(angle: float) -> float:
    """Convert a DoA angle to a yaw relative to the head (positive = left)."""
    return wrap_angle(_FRONT_RAD - angle)


def mean_angle(angles: list[float]) -> float:
    """Circular mean, so readings either side of the wrap-around don't average to the opposite direction."""
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


class SpeakerAttention:
    """Poll the daemon's DoA while the user speaks and ask the movement manager to look that way."""

    def __init__(self, movement_manager: Any, doa_url: str, client: httpx.Client | None = None) -> None:
        """Read DoA from ``doa_url`` (the daemon's ``/api/state/doa``) and drive ``movement_manager``."""
        self._movement_manager = movement_manager
        self._doa_url = doa_url
        self._client = client or httpx.Client(timeout=1.0, trust_env=False)
        self._speaking = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_yaw: float | None = None
        self._online: bool | None = None

    def set_user_speaking(self, speaking: bool) -> None:
        """Mark the start or end of the user's speech; DoA is only sampled in between."""
        if speaking:
            self._speaking.set()
        else:
            self._speaking.clear()

    def status(self) -> dict[str, object]:
        """Return the latest decision for the status page."""
        with self._lock:
            last_yaw = self._last_yaw
        return {
            "user_speaking": self._speaking.is_set(),
            "doa_online": self._online,
            "last_voice_yaw_deg": None if last_yaw is None else round(math.degrees(last_yaw)),
        }

    def read_doa(self) -> tuple[float, bool] | None:
        """Return ``(angle, speech_detected)`` from the daemon, or None when unavailable."""
        try:
            response = self._client.get(self._doa_url)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as e:
            self._set_online(False, e)
            return None
        self._set_online(True)
        if not isinstance(payload, dict):
            return None  # the daemon answers null until the microphone array is up
        try:
            return float(payload["angle"]), bool(payload["speech_detected"])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Unexpected DoA payload %r: %s", payload, e)
            return None

    def _set_online(self, online: bool, error: Exception | None = None) -> None:
        if online != self._online:
            if online:
                logger.info("Sound direction available from the daemon")
            else:
                logger.warning("Sound direction unavailable, cannot turn toward voices: %s", error)
        self._online = online

    def listen_once(self) -> float | None:
        """Follow one utterance: sample DoA until speech ends or enough readings agree; return the yaw chosen."""
        with_speech: list[float] = []
        without_speech: list[float] = []
        while self._speaking.is_set() and not self._stop.is_set():
            doa = self.read_doa()
            if doa is None:
                return None
            angle, speech_detected = doa
            (with_speech if speech_detected else without_speech).append(angle)
            if len(with_speech) >= MIN_SPEECH_READINGS:
                break
            self._stop.wait(POLL_INTERVAL_S)
        # The microphone's own VAD may miss soft voices; the backend already confirmed someone spoke.
        readings = with_speech or (without_speech if len(without_speech) >= 2 else [])
        if not readings:
            return None
        yaw = doa_to_yaw(mean_angle(readings))
        with self._lock:
            self._last_yaw = yaw
        self._movement_manager.look_toward(yaw)
        return yaw

    def start(self) -> None:
        """Start the polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speaker-attention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._speaking.wait(0.2):
                continue
            self.listen_once()
            # One decision per utterance: wait for the speech to end before listening again.
            while self._speaking.is_set() and not self._stop.wait(POLL_INTERVAL_S):
                pass
