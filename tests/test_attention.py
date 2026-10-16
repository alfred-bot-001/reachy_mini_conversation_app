"""Speaker attention: while the user talks, the sound direction decides where the head should turn."""

import math
import threading
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app.config import (
    DAEMON_URL_ENV,
    FACE_FOLLOW_ENV,
    face_follow_enabled,
    resolve_daemon_doa_url,
)
from reachy_mini_conversation_app.attention import (
    SpeakerAttention,
    doa_to_yaw,
    mean_angle,
    wrap_angle,
)


def _attention(readings: list[Any], movement_manager: MagicMock | None = None) -> tuple[SpeakerAttention, list[str]]:
    """Attention backed by a fake daemon that serves ``readings`` in order (the last one repeats)."""
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(request.url.path)
        reading = readings[min(len(served), len(readings)) - 1]
        if isinstance(reading, int):
            return httpx.Response(reading)
        return httpx.Response(200, json=reading)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    attention = SpeakerAttention(movement_manager or MagicMock(), "http://daemon.invalid/api/state/doa", client=client)
    return attention, served


def test_doa_angles_map_to_head_relative_yaw() -> None:
    """ReSpeaker angles: 0 = left, π/2 = front, π = right; yaw is positive to the left."""
    assert doa_to_yaw(math.pi / 2) == pytest.approx(0.0)
    assert doa_to_yaw(0.0) == pytest.approx(math.pi / 2)
    assert doa_to_yaw(math.pi) == pytest.approx(-math.pi / 2)
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    # Readings either side of the wrap-around average to the front, not the back.
    assert mean_angle([0.1, 2 * math.pi - 0.1]) == pytest.approx(0.0, abs=1e-9)


def test_face_follow_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Face follow is on by default; the daemon URL can point at a robot on the LAN."""
    monkeypatch.delenv(FACE_FOLLOW_ENV, raising=False)
    monkeypatch.delenv(DAEMON_URL_ENV, raising=False)
    assert face_follow_enabled() is True
    assert resolve_daemon_doa_url() == "http://localhost:8000/api/state/doa"
    monkeypatch.setenv(FACE_FOLLOW_ENV, "0")
    monkeypatch.setenv(DAEMON_URL_ENV, "http://192.168.1.31:8000/")
    assert face_follow_enabled() is False
    assert resolve_daemon_doa_url() == "http://192.168.1.31:8000/api/state/doa"


def test_listen_once_turns_toward_the_voice_after_a_few_speech_readings() -> None:
    """Three readings with speech from the left ask the movement manager to look left."""
    movement_manager = MagicMock()
    attention, served = _attention(
        [
            {"angle": math.pi / 2, "speech_detected": False},  # stale direction before the voice registers
            {"angle": 0.2, "speech_detected": True},
            {"angle": 0.3, "speech_detected": True},
            {"angle": 0.25, "speech_detected": True},
        ],
        movement_manager,
    )
    attention.set_user_speaking(True)

    yaw = attention.listen_once()

    assert yaw == pytest.approx(doa_to_yaw(0.25), abs=0.01)
    movement_manager.look_toward.assert_called_once_with(pytest.approx(yaw))
    assert len(served) == 4
    assert attention.status()["last_voice_yaw_deg"] == round(math.degrees(yaw))


def test_listen_once_falls_back_to_unflagged_readings_when_speech_ends() -> None:
    """A soft voice the microphone's own VAD missed still counts once the backend says the user spoke."""
    movement_manager = MagicMock()
    attention, served = _attention([{"angle": 2.5, "speech_detected": False}], movement_manager)
    attention.set_user_speaking(True)

    def stop_after_two_polls() -> None:
        while len(served) < 2:
            pass
        attention.set_user_speaking(False)

    threading.Thread(target=stop_after_two_polls, daemon=True).start()
    yaw = attention.listen_once()

    assert yaw == pytest.approx(doa_to_yaw(2.5))
    movement_manager.look_toward.assert_called_once()


def test_listen_once_gives_up_without_the_daemon_or_while_silent() -> None:
    """No DoA (daemon down or audio not up) or a single stale reading never moves the head."""
    movement_manager = MagicMock()
    attention, _ = _attention([503], movement_manager)
    attention.set_user_speaking(True)
    assert attention.listen_once() is None
    assert attention.status()["doa_online"] is False

    attention, _ = _attention([None], movement_manager)
    attention.set_user_speaking(True)
    attention.set_user_speaking(False)
    assert attention.listen_once() is None
    movement_manager.look_toward.assert_not_called()
