"""Doze mode: tuck and hold still after a long silence, wake only when called by name."""

import time
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE, SLEEP_ANTENNAS_JOINT_POSITIONS
from tests.test_huggingface_realtime import _FakeEvent, _make_fake_realtime_client
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.doze import DozeState, LoudnessTrigger, level_dbfs
from reachy_mini_conversation_app.moves import (
    LAZY_WAKE_KEYFRAMES,
    DozeMove,
    LazyWakeMove,
    BreathingMove,
    MovementManager,
)
from reachy_mini_conversation_app.config import (
    DEFAULT_WAKE_WORDS,
    resolve_loud_wake,
    resolve_wake_words,
    resolve_doze_minutes,
    speech_wobble_enabled,
    resolve_drowsy_minutes,
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
    for text in ("小白", "小白，起床啦", "嘿 小 白！", "晓白你醒醒", "小百快醒醒", "Xiao Bai wake up"):
        assert doze.is_wake_call(text), text
    for text in ("今天吃什么", "外面风好大", "风风快来", "白白浪费", "", "   "):
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
    handler = _dozing_handler(monkeypatch, "小白，醒醒")

    await handler._run_realtime_session()

    assert not handler.is_dozing
    assert {"role": "user", "content": "小白，醒醒"} in _messages(_drain(handler))
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


# ── Lazy wake on a loud noise ───────────────────────────────────────────


def _tone(dbfs: float, seconds: float = 0.1, rate: int = 16000) -> np.ndarray:
    amplitude = 32768.0 * 10 ** (dbfs / 20) * np.sqrt(2)
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def test_level_dbfs_measures_rms() -> None:
    """A sine at -20 dBFS RMS reads about -20; silence reads very low."""
    assert abs(level_dbfs(_tone(-20.0)) - (-20.0)) < 0.5
    assert level_dbfs(np.zeros(1600, dtype=np.int16)) <= -90


def test_loudness_needs_a_sustained_noise_then_cools_down() -> None:
    """A short click does not wake; ~0.8 s of loud sound does; then it stays quiet for the cooldown."""
    trigger = LoudnessTrigger(threshold_dbfs=-30.0, hold_s=0.8, cooldown_s=60.0)
    assert not trigger.feed(_tone(-10.0), 16000, now=0.0)  # 0.1 s click
    for i in range(10):  # quiet drains it
        assert not trigger.feed(_tone(-60.0), 16000, now=0.1 + i * 0.1)
    fired = [trigger.feed(_tone(-20.0), 16000, now=2.0 + i * 0.1) for i in range(9)]
    assert fired.count(True) == 1 and fired[-1] is True
    assert not any(trigger.feed(_tone(-10.0), 16000, now=3.0 + i * 0.1) for i in range(20))  # cooldown
    assert trigger.peak_dbfs > -11


def test_lazy_wake_move_rises_from_sleep_looks_both_ways_and_ends_neutral() -> None:
    """Starts at the doze pose, eases smoothly (no jumps), looks left and right, ends at neutral."""
    sleep = np.array(SLEEP_HEAD_POSE)
    move = LazyWakeMove(start_pose=sleep, start_antennas=tuple(SLEEP_ANTENNAS_JOINT_POSITIONS))
    assert move.duration == LAZY_WAKE_KEYFRAMES[-1][0] >= 12
    head, antennas, _ = move.evaluate(0.0)
    assert np.allclose(head, sleep, atol=2e-3) and np.allclose(antennas, SLEEP_ANTENNAS_JOINT_POSITIONS)
    previous = None
    yaws = []
    for t in np.arange(0.0, move.duration + 0.5, 0.01):
        head, antennas, _ = move.evaluate(float(t))
        yaws.append(np.degrees(np.arctan2(head[1, 0], head[0, 0])))
        if previous is not None:
            assert np.abs(head - previous[0]).max() < 0.02 and np.abs(antennas - previous[1]).max() < 0.05
        previous = (head, antennas)
    assert max(yaws) > 25 and min(yaws) < -25
    assert np.allclose(head, create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), atol=1e-6)
    assert np.allclose(antennas, (-0.1745, 0.1745))


def test_manager_lazy_wake_plays_move_then_breathes() -> None:
    """The lazy wake replaces the doze pose with the wake-up move; breathing follows once it ends."""
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [-0.17, 0.17])
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    manager = MovementManager(current_robot=robot)
    now = time.monotonic()
    manager._handle_command("set_dozing", True, now)
    manager._handle_command("lazy_wake", None, now)
    assert isinstance(manager.state.current_move, LazyWakeMove)
    assert not manager._dozing and not manager._breathing_active
    later = now + LAZY_WAKE_KEYFRAMES[-1][0] + 0.1
    manager._manage_move_queue(later)
    assert manager.state.current_move is None
    manager.state.last_activity_time = later - 1
    manager._manage_breathing(later)
    manager._manage_move_queue(later)
    assert isinstance(manager.state.current_move, BreathingMove)


@pytest.mark.asyncio
async def test_loud_noise_wakes_lazily_and_dozes_again_if_ignored(monkeypatch: Any) -> None:
    """A loud noise while dozing wakes lazily; with nobody engaging, Reachy dozes off after the drowsy window."""
    handler = _plain_handler()
    handler.connection = MagicMock()
    handler.connection.session.update = AsyncMock()
    handler.connection.input_audio_buffer.append = AsyncMock()
    handler._loud = LoudnessTrigger(threshold_dbfs=-30.0, hold_s=0.3)
    handler._drowsy_s = 120.0
    await handler.fall_asleep()
    for _ in range(4):
        await handler.receive((16000, _tone(-15.0)))
    await asyncio.sleep(0)
    assert not handler.is_dozing
    handler.deps.movement_manager.wake_lazily.assert_called_once_with()
    assert handler._turn_detection()["create_response"] is True
    await handler._maybe_doze()
    assert not handler.is_dozing  # still within the drowsy window
    handler._drowsy_until = time.monotonic() - 1
    await handler._maybe_doze()
    assert handler.is_dozing


@pytest.mark.asyncio
async def test_lazily_woken_reachy_stays_up_once_someone_talks() -> None:
    """Any conversation after the lazy wake cancels the early doze."""
    handler = _plain_handler()
    handler.connection = MagicMock()
    handler.connection.session.update = AsyncMock()
    await handler.fall_asleep()
    await handler.wake_up("loud_sound", lazy=True)
    handler._drowsy_until = time.monotonic() - 1
    handler.last_activity_time = handler._drowsy_activity + 5  # someone spoke
    await handler._maybe_doze()
    assert not handler.is_dozing and handler._drowsy_until is None


def test_quiet_room_and_awake_robot_do_not_trigger() -> None:
    """Quiet audio never wakes; nothing is measured while awake."""
    handler = _plain_handler()
    handler._loud = LoudnessTrigger(threshold_dbfs=-30.0, hold_s=0.3)
    handler._listen_while_dozing(16000, _tone(-10.0, seconds=1.0))  # awake: ignored
    assert handler._loud.peak_dbfs == -96.0
    handler._doze.doze()
    for _ in range(30):
        handler._listen_while_dozing(16000, _tone(-50.0))
    assert handler.is_dozing


def test_loud_wake_settings(monkeypatch: Any) -> None:
    """Defaults, overrides and switching the loud wake off."""
    monkeypatch.delenv("REACHY_MINI_LOUD_WAKE_DBFS", raising=False)
    monkeypatch.delenv("REACHY_MINI_LOUD_WAKE_SECONDS", raising=False)
    monkeypatch.delenv("REACHY_MINI_DROWSY_MINUTES", raising=False)
    assert resolve_loud_wake() == (-30.0, 0.8) and resolve_drowsy_minutes() == 2.0
    monkeypatch.setenv("REACHY_MINI_LOUD_WAKE_DBFS", "-25")
    monkeypatch.setenv("REACHY_MINI_DROWSY_MINUTES", "5")
    assert resolve_loud_wake() == (-25.0, 0.8) and resolve_drowsy_minutes() == 5.0
    monkeypatch.setenv("REACHY_MINI_LOUD_WAKE_DBFS", "off")
    assert resolve_loud_wake() is None
