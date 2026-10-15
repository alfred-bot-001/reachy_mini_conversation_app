"""Doze mode: tuck and hold still after a long silence, wake only when called by name."""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE, SLEEP_ANTENNAS_JOINT_POSITIONS
from tests.test_huggingface_realtime import _FakeEvent, _make_fake_realtime_client
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.doze import DozeState
from reachy_mini_conversation_app.moves import DozeMove, BreathingMove, MovementManager
from reachy_mini_conversation_app.config import (
    DEFAULT_WAKE_WORDS,
    resolve_wake_words,
    resolve_doze_minutes,
    speech_wobble_enabled,
)
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


def _plain_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def _session_handler(monkeypatch: Any, events: tuple[_FakeEvent, ...]) -> HuggingFaceRealtimeHandler:
    """Build a handler wired to a fake realtime client yielding `events`, ready to run a session."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=None: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    handler = _plain_handler()
    handler.client = _make_fake_realtime_client(events=events)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    return handler


def _drain(handler: HuggingFaceRealtimeHandler) -> list[Any]:
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def _messages(items: list[Any]) -> list[dict[str, Any]]:
    return [item.args[0] for item in items if isinstance(item, AdditionalOutputs)]


def test_doze_settings_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doze after 30 minutes by default; 0 disables it; wake words can be overridden."""
    monkeypatch.delenv("REACHY_MINI_DOZE_MINUTES", raising=False)
    monkeypatch.delenv("REACHY_MINI_WAKE_WORDS", raising=False)
    monkeypatch.delenv("REACHY_MINI_SPEECH_WOBBLE", raising=False)
    assert resolve_doze_minutes() == 30.0
    assert resolve_wake_words() == DEFAULT_WAKE_WORDS
    assert speech_wobble_enabled() is False

    monkeypatch.setenv("REACHY_MINI_DOZE_MINUTES", "0")
    assert resolve_doze_minutes() is None
    monkeypatch.setenv("REACHY_MINI_DOZE_MINUTES", "abc")
    assert resolve_doze_minutes() == 30.0
    monkeypatch.setenv("REACHY_MINI_WAKE_WORDS", "小黑，Robot")
    assert resolve_wake_words() == ("小黑", "Robot")


def test_wake_call_matches_name_and_homophones_only() -> None:
    """The name (or a common mis-transcription of it) wakes Reachy; other speech does not."""
    doze = DozeState(1800, DEFAULT_WAKE_WORDS)
    for text in ("风风", "风风，起床啦", "嘿 风 风！", "峰峰你醒醒", "丰风快醒醒", "Feng Feng wake up"):
        assert doze.is_wake_call(text), text
    for text in ("今天吃什么", "外面风好大", "小白快来", "丰富多彩", "", "   "):
        assert not doze.is_wake_call(text), text


def test_doze_state_timing_and_transitions() -> None:
    """Doze only once the silence is long enough, and never when disabled."""
    doze = DozeState(1800, DEFAULT_WAKE_WORDS)
    assert not doze.should_doze(1799)
    assert doze.should_doze(1800)
    assert doze.doze() and doze.dozing and not doze.doze()
    assert not doze.should_doze(99999)  # already dozing
    assert doze.wake() and not doze.dozing and not doze.wake()
    assert not DozeState(None, DEFAULT_WAKE_WORDS).should_doze(10**9)


def test_doze_move_eases_into_sleep_pose_and_holds() -> None:
    """The doze move starts at the current pose, ends in the sleep pose, and then never moves."""
    start = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    move = DozeMove(start_pose=start, start_antennas=(-0.17, 0.17), transition_duration=3.0)
    assert move.duration == float("inf")

    head0, antennas0, _ = move.evaluate(0.0)
    np.testing.assert_allclose(head0, start, atol=1e-6)
    np.testing.assert_allclose(antennas0, [-0.17, 0.17], atol=1e-6)

    held = [move.evaluate(t) for t in (3.0, 10.0, 600.0)]
    for head, antennas, body_yaw in held:
        np.testing.assert_allclose(head, SLEEP_HEAD_POSE, atol=1e-9)
        np.testing.assert_allclose(antennas, SLEEP_ANTENNAS_JOINT_POSITIONS, atol=1e-9)
        assert body_yaw == 0.0


def test_movement_manager_dozing_rejects_moves_and_wakes_to_breathing() -> None:
    """While dozing, queued moves and clears cannot disturb the pose; waking rises back through breathing."""
    robot = MagicMock()
    manager = MovementManager(robot)
    now = manager._now()
    manager._head_tracking = True

    manager._handle_command("set_dozing", True, now)
    assert isinstance(manager.state.current_move, DozeMove)
    robot.stop_head_tracking.assert_called_once()

    manager._handle_command(
        "queue_move", GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)), now
    )
    assert len(manager.move_queue) == 0
    manager._handle_command("clear_queue", None, now)
    assert isinstance(manager.state.current_move, DozeMove)

    # Breathing never replaces the doze pose, however long it lasts.
    manager.state.last_activity_time = now - 3600
    manager._manage_breathing(now)
    assert isinstance(manager.state.current_move, DozeMove)
    assert manager.get_status()["dozing"] is True

    manager._handle_command("set_dozing", False, now)
    assert isinstance(manager.state.current_move, BreathingMove)
    assert manager.state.current_move.interpolation_duration == 2.0
    robot.start_head_tracking.assert_called_with(weight=1.0)
    assert manager.get_status()["dozing"] is False


def _dozing_handler(monkeypatch: Any, transcript: str) -> Any:
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("conversation.item.input_audio_transcription.completed", transcript=transcript, item_id="it1"),),
    )
    handler._doze.doze()
    return handler


@pytest.mark.asyncio
async def test_dozing_ignores_speech_without_the_name(monkeypatch: Any) -> None:
    """Speech without the wake word is dropped: no transcript, no response, still dozing."""
    handler = _dozing_handler(monkeypatch, "今天吃什么呀")

    await handler._run_realtime_session()

    assert handler.is_dozing
    assert [m for m in _messages(_drain(handler)) if m.get("role") == "user"] == []
    assert handler._pending_responses.empty()
    handler.deps.movement_manager.set_dozing.assert_not_called()


@pytest.mark.asyncio
async def test_dozing_wakes_on_name_and_answers(monkeypatch: Any) -> None:
    """Calling the name wakes Reachy, re-enables auto responses, and answers this very turn."""
    handler = _dozing_handler(monkeypatch, "风风，醒醒")

    await handler._run_realtime_session()

    assert not handler.is_dozing
    assert {"role": "user", "content": "风风，醒醒"} in _messages(_drain(handler))
    assert not handler._pending_responses.empty()
    handler.deps.movement_manager.set_dozing.assert_called_once_with(False)
    assert handler._turn_detection()["create_response"] is True


@pytest.mark.asyncio
async def test_long_silence_dozes_and_turns_off_auto_response(monkeypatch: Any) -> None:
    """After the doze delay with nothing pending, Reachy dozes and the backend stops auto-answering."""
    handler = _plain_handler()
    updates: list[dict[str, Any]] = []

    async def capture(**kwargs: Any) -> None:
        updates.append(kwargs)

    handler.connection = MagicMock()
    handler.connection.session.update = capture
    handler.last_activity_time = time.monotonic() - 31 * 60

    await handler._maybe_doze()

    assert handler.is_dozing
    handler.deps.movement_manager.set_dozing.assert_called_once_with(True)
    assert updates[-1]["session"]["audio"]["input"]["turn_detection"]["create_response"] is False
    assert handler._get_session_config([])["audio"]["input"]["turn_detection"]["create_response"] is False
    assert not handler._idle_behavior_ready()


@pytest.mark.asyncio
async def test_no_doze_while_recently_active_or_responding(monkeypatch: Any) -> None:
    """Recent activity or an active response keeps Reachy awake."""
    handler = _plain_handler()
    handler.connection = MagicMock()
    handler.last_activity_time = time.monotonic() - 60
    await handler._maybe_doze()
    assert not handler.is_dozing

    handler.last_activity_time = time.monotonic() - 31 * 60
    handler._response_done_event.clear()
    await handler._maybe_doze()
    assert not handler.is_dozing


@pytest.mark.asyncio
async def test_say_wakes_before_speaking(monkeypatch: Any) -> None:
    """A proactive reminder (conversation.say) wakes Reachy first."""
    handler = _plain_handler()
    handler.connection = MagicMock()

    async def noop(**_kw: Any) -> None:
        return None

    handler.connection.session.update = noop
    handler.connection.conversation.item.create = noop
    handler._doze.doze()

    await handler.say("提醒爸爸开会")

    assert not handler.is_dozing
    handler.deps.movement_manager.set_dozing.assert_called_once_with(False)


def test_doze_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_DOZE_MINUTES=0 turns dozing off for new handlers."""
    monkeypatch.setenv("REACHY_MINI_DOZE_MINUTES", "0")
    handler = _plain_handler()
    assert not handler._doze.enabled
    assert hf_mod.HuggingFaceRealtimeHandler is type(handler)
