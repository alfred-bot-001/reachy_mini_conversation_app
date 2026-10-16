import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import (
    BreathingMove,
    MovementManager,
    LoopFrequencyStats,
    clone_full_body_pose,
)
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove, EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_clone_full_body_pose_is_a_deep_copy() -> None:
    """Cloning a pose must not alias the head-pose array of the original."""
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    original = (head, (1.0, 2.0), 3.0)
    clone = clone_full_body_pose(original)
    head[0, 0] = 999.0
    assert clone[0][0, 0] != 999.0
    assert clone[1] == (1.0, 2.0)
    assert clone[2] == 3.0


def test_loop_frequency_stats_reset_keeps_last_potential() -> None:
    """Reset clears accumulators but preserves last/potential frequency."""
    stats = LoopFrequencyStats(mean=5.0, m2=2.0, min_freq=1.0, count=10, last_freq=59.0, potential_freq=61.0)
    stats.reset()
    assert stats.mean == 0.0
    assert stats.m2 == 0.0
    assert stats.count == 0
    assert stats.min_freq == float("inf")
    assert stats.last_freq == 59.0
    assert stats.potential_freq == 61.0


def test_breathing_move_interpolates_then_breathes() -> None:
    """Phase 1 starts at the given antennas; phase 2 keeps body yaw neutral."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(0.3, -0.3),
        interpolation_duration=1.0,
    )
    head_start, antennas_start, body_yaw_start = move.evaluate(0.0)
    assert head_start is not None
    np.testing.assert_allclose(antennas_start, [0.3, -0.3])
    assert body_yaw_start == 0.0

    head_breathe, antennas_breathe, body_yaw_breathe = move.evaluate(5.0)
    assert head_breathe is not None
    assert antennas_breathe is not None and antennas_breathe.shape == (2,)
    assert body_yaw_breathe == 0.0


def test_breathing_antennas_do_not_jump_when_breathing_starts() -> None:
    """Phase 2 continues from the neutral antennas reached at the end of phase 1."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(0.3, -0.3),
        interpolation_duration=1.0,
    )
    _, antennas_end_of_interpolation, _ = move.evaluate(1.0 - 1e-6)
    _, antennas_start_of_breathing, _ = move.evaluate(1.0)
    assert antennas_end_of_interpolation is not None and antennas_start_of_breathing is not None

    np.testing.assert_allclose(antennas_start_of_breathing, antennas_end_of_interpolation, atol=1e-4)


def test_breathing_stays_small_and_antennas_never_cross_vertical() -> None:
    """Idle breathing keeps antennas within a few degrees of neutral and the head nearly still."""
    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    move = BreathingMove(
        interpolation_start_pose=neutral,
        interpolation_start_antennas=(-0.1745, 0.1745),
        interpolation_duration=1.0,
    )

    for t in np.linspace(1.0, 31.0, 601):
        head, antennas, _ = move.evaluate(float(t))
        assert head is not None and antennas is not None
        right, left = antennas
        assert right < 0.0 < left
        np.testing.assert_array_less(np.abs(antennas - np.array([-0.1745, 0.1745])), np.deg2rad(5))
        assert abs(head[2, 3]) <= 0.003


def test_is_idle_reflects_listening_and_activity() -> None:
    """is_idle is False while listening and True once past the inactivity delay."""
    manager = MovementManager(MagicMock())

    manager._shared_is_listening = True
    assert manager.is_idle() is False

    manager._shared_is_listening = False
    manager._shared_last_activity_time = manager._now()
    assert manager.is_idle() is False

    manager._shared_last_activity_time = manager._now() - 10.0
    assert manager.is_idle() is True


def test_handle_command_queue_and_clear() -> None:
    """queue_move appends real moves, ignores bad payloads, and clear empties the queue."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    move = GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))

    manager._handle_command("queue_move", move, now)
    assert list(manager.move_queue) == [move]

    manager._handle_command("queue_move", "not-a-move", now)
    assert list(manager.move_queue) == [move]

    manager._handle_command("clear_queue", None, now)
    assert len(manager.move_queue) == 0
    assert manager.state.current_move is None


class _Face:
    def __init__(self, detected: bool, x: float | None = None) -> None:
        self.detected = detected
        self.x = x
        self.y = 0.0 if detected else None


def _tracking_manager(face: _Face, head_yaw_deg: float = 0.0) -> tuple[MovementManager, MagicMock]:
    robot = MagicMock()
    robot.get_tracked_face.return_value = face
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, head_yaw_deg, degrees=True)
    manager = MovementManager(robot)
    manager._head_tracking = True
    return manager, robot


def _commanded_yaw_deg(manager: MovementManager, t: float) -> float:
    head, _, _ = manager.state.current_move.evaluate(t)
    return float(np.degrees(np.arctan2(head[1, 0], head[0, 0])))


def test_look_toward_turns_to_a_voice_away_from_the_tracked_face_then_resumes_tracking() -> None:
    """A voice 40° left of a centred face pauses tracking, turns the head there and hands it back."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
    now = manager._now()

    manager._handle_command("look_toward", np.radians(40), now)

    robot.start_head_tracking.assert_called_once_with(weight=0.0)
    assert manager.get_status()["turning"] is True
    assert isinstance(manager.state.current_move, GotoQueueMove)
    assert _commanded_yaw_deg(manager, 0.0) == pytest.approx(0.0, abs=0.5)
    assert _commanded_yaw_deg(manager, 10.0) == pytest.approx(40.0, abs=0.5)

    manager._finish_turn(now + 0.1)
    assert manager.get_status()["turning"] is True
    manager._finish_turn(now + 1.0)
    assert manager.get_status()["turning"] is False
    robot.start_head_tracking.assert_called_with(weight=1.0)


def test_look_toward_ignores_a_voice_from_the_tracked_face() -> None:
    """The person already being looked at keeps talking: no turn, tracking stays on."""
    # Face 20° to the right of the image centre (x > 0) and a voice ~25° right of the head.
    manager, robot = _tracking_manager(_Face(detected=True, x=0.38))
    manager._handle_command("look_toward", np.radians(-25), manager._now())
    robot.start_head_tracking.assert_not_called()
    assert manager.state.current_move is None


def test_look_toward_without_a_face_turns_only_for_clear_offsets_and_clamps_the_neck() -> None:
    """No face locked: small offsets are left to the tracker; large ones turn as far as the neck allows."""
    manager, robot = _tracking_manager(_Face(detected=False))
    manager._handle_command("look_toward", np.radians(8), manager._now())
    robot.start_head_tracking.assert_not_called()

    manager, robot = _tracking_manager(_Face(detected=False), head_yaw_deg=30)
    manager._handle_command("look_toward", np.radians(-100), manager._now())
    assert _commanded_yaw_deg(manager, 10.0) == pytest.approx(-60.0, abs=0.5)


def test_look_toward_is_blocked_while_speaking_dozing_untracked_or_cooling_down() -> None:
    """Reachy keeps still while it talks or dozes, never turns with tracking off, and not twice in a row."""
    manager, robot = _tracking_manager(_Face(detected=False))
    yaw = np.radians(45)
    now = manager._now()

    manager._is_speaking = True
    manager._handle_command("look_toward", yaw, now)
    manager._is_speaking = False
    manager._dozing = True
    manager._handle_command("look_toward", yaw, now)
    manager._dozing = False
    manager._head_tracking = False
    manager._handle_command("look_toward", yaw, now)
    robot.start_head_tracking.assert_not_called()

    manager._head_tracking = True
    manager._handle_command("look_toward", yaw, now)
    robot.start_head_tracking.assert_called_once_with(weight=0.0)
    manager._finish_turn(now + 1.0)
    manager._handle_command("look_toward", yaw, now + 1.0)
    assert manager.get_status()["turning"] is False  # within the cooldown
    manager._handle_command("look_toward", yaw, now + 3.0)
    assert manager.get_status()["turning"] is True


def test_dozing_cancels_a_turn_and_listening_notifies_attention() -> None:
    """Dozing mid-turn must not resume tracking afterwards; listening toggles reach speaker attention."""
    manager, robot = _tracking_manager(_Face(detected=False))
    now = manager._now()
    manager._handle_command("look_toward", np.radians(45), now)
    manager._handle_command("set_dozing", True, now)
    robot.start_head_tracking.reset_mock()
    manager._finish_turn(now + 1.0)
    robot.start_head_tracking.assert_not_called()

    heard: list[bool] = []
    manager.on_listening = heard.append
    manager.set_listening(True)
    assert heard == [True]
