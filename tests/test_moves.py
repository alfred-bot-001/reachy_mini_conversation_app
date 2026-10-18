import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import (
    LIMIT_HOLD_S,
    TURN_COOLDOWN_S,
    TURN_DURATION_S,
    FACE_LOST_HOLD_S,
    RECENTER_QUIET_S,
    QUIET_DIRECTION_S,
    VOICE_FACE_WAIT_S,
    BODY_FOLLOW_DELAY_S,
    RECENTER_DURATION_S,
    TURN_MAX_DURATION_S,
    BODY_FOLLOW_COOLDOWN_S,
    BODY_FOLLOW_DURATION_S,
    BODY_REVERSE_COOLDOWN_S,
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
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    robot.get_tracked_face.return_value = _Face(detected=True, x=0.0)
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
    def __init__(self, detected: bool, x: float | None = None, y: float | None = None) -> None:
        self.detected = detected
        self.x = x
        self.y = (0.0 if y is None else y) if detected else None


def _tracking_manager(
    face: _Face, head_yaw_deg: float = 0.0, body_yaw_deg: float = 0.0
) -> tuple[MovementManager, MagicMock]:
    robot = MagicMock()
    robot.get_tracked_face.return_value = face
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, head_yaw_deg, degrees=True)
    robot.get_current_joint_positions.return_value = ([np.radians(body_yaw_deg)] + [0.0] * 6, [0.0, 0.0])
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
    manager._finish_turn(now + TURN_DURATION_S)
    assert manager.get_status()["turning"] is False
    robot.start_head_tracking.assert_called_with(weight=1.0)
    # Until the tracker sees a face again the head stays where the turn ended, breathing or not.
    assert manager._watch_anchor is not None


def test_turns_ease_in_and_out() -> None:
    """A voice turn starts and ends slowly instead of jerking off at full speed."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
    manager._handle_command("look_toward", np.radians(40), manager._now())
    assert _commanded_yaw_deg(manager, 0.1 * TURN_DURATION_S) < 4.0 * 0.5  # linear would already be at 4°
    assert _commanded_yaw_deg(manager, 0.5 * TURN_DURATION_S) == pytest.approx(20.0, abs=0.5)


def test_big_voice_turns_take_longer_than_small_ones() -> None:
    """Turn time grows with the angle, so a swing across the room is never a snap."""
    manager, _ = _tracking_manager(_Face(detected=True, x=0.0))
    now = manager._now()
    manager._handle_command("look_toward", np.radians(40), now)
    assert manager._turn_until - now == pytest.approx(TURN_DURATION_S)

    manager, _ = _tracking_manager(_Face(detected=True, x=0.0))
    manager._handle_command("look_toward", np.radians(60), now)
    assert manager._turn_until - now == pytest.approx(2.0)

    manager, _ = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=-50)
    manager._handle_command("look_toward", np.radians(100), now)
    assert manager._turn_until - now == pytest.approx(TURN_MAX_DURATION_S)
    assert _commanded_yaw_deg(manager, TURN_MAX_DURATION_S) == pytest.approx(50.0, abs=0.5)


def _turn_and_settle(
    manager: MovementManager, robot: MagicMock, yaw_deg: float, start: float, face_after: _Face
) -> float:
    """Turn toward a voice, then report what the tracker sees there; return when the check is over."""
    manager._handle_command("look_toward", np.radians(yaw_deg), start)
    end = manager._turn_until
    manager._finish_turn(end)
    robot.get_tracked_face.return_value = face_after
    manager._check_voice_turn(end + 0.5)
    manager._check_voice_turn(end + VOICE_FACE_WAIT_S)
    return end + VOICE_FACE_WAIT_S


def test_a_voice_turn_that_finds_nobody_stops_voices_from_that_way_turning_the_head() -> None:
    """A TV: once a turn toward it finds no face, its voice is ignored for a while; other directions still turn."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
    now = _turn_and_settle(manager, robot, 40, manager._now(), _Face(detected=False))

    robot.get_tracked_face.return_value = _Face(detected=True, x=0.0)
    later = now + TURN_COOLDOWN_S
    manager._handle_command("look_toward", np.radians(50), later)
    assert manager.get_status()["turning"] is False

    manager._handle_command("look_toward", np.radians(-40), later)
    assert manager.get_status()["turning"] is True

    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
    now = _turn_and_settle(manager, robot, 40, manager._now(), _Face(detected=False))
    robot.get_tracked_face.return_value = _Face(detected=True, x=0.0)
    manager._handle_command("look_toward", np.radians(40), now + QUIET_DIRECTION_S)
    assert manager.get_status()["turning"] is True


def test_a_voice_turn_that_finds_a_face_keeps_turning_that_way() -> None:
    """A turn that finds someone leaves that direction open to later voices."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
    now = _turn_and_settle(manager, robot, 40, manager._now(), _Face(detected=True, x=0.1))
    now = _turn_and_settle(manager, robot, -40, now + TURN_COOLDOWN_S, _Face(detected=True, x=0.1))
    manager._handle_command("look_toward", np.radians(40), now + TURN_COOLDOWN_S)
    assert manager.get_status()["turning"] is True


def test_look_toward_ignores_a_voice_from_the_tracked_face() -> None:
    """The person already being looked at keeps talking: no turn, tracking stays on."""
    # Face 20° to the right of the image centre (x > 0) and a voice ~25° right of the head.
    manager, robot = _tracking_manager(_Face(detected=True, x=0.38))
    manager._handle_command("look_toward", np.radians(-25), manager._now())
    robot.start_head_tracking.assert_not_called()
    assert manager.state.current_move is None


def test_look_toward_without_a_face_stays_put_and_a_far_voice_turns_only_as_far_as_the_neck_allows() -> None:
    """Nobody in view: voices (TV, echoes, talk elsewhere) never swing the head. With a face, the neck is clamped."""
    manager, robot = _tracking_manager(_Face(detected=False))
    for degrees in (8, 40, -100):
        manager._handle_command("look_toward", np.radians(degrees), manager._now())
    robot.start_head_tracking.assert_not_called()
    assert manager.state.current_move is None

    manager, robot = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=30)
    manager._handle_command("look_toward", np.radians(-100), manager._now())
    assert _commanded_yaw_deg(manager, 10.0) == pytest.approx(-60.0, abs=0.5)


def test_look_toward_is_blocked_while_speaking_dozing_untracked_or_cooling_down() -> None:
    """Reachy keeps still while it talks or dozes, never turns with tracking off, and not twice in a row."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
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
    manager._finish_turn(now + TURN_DURATION_S)
    manager._handle_command("look_toward", yaw, now + TURN_DURATION_S)
    assert manager.get_status()["turning"] is False  # within the cooldown
    manager._handle_command("look_toward", yaw, now + 3.5)
    assert manager.get_status()["turning"] is True


def test_look_toward_clamps_the_neck_relative_to_the_body_and_keeps_the_body_still() -> None:
    """With the body already turned 40° left, a voice far left takes the head to body + 60°, body untouched."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=40, body_yaw_deg=40)
    manager._handle_command("look_toward", np.radians(100), manager._now())
    assert _commanded_yaw_deg(manager, 10.0) == pytest.approx(100.0, abs=0.5)
    assert manager.state.current_move.evaluate(10.0)[2] == pytest.approx(np.radians(40))


def test_a_neck_held_round_turns_the_body_under_the_head_and_idling_keeps_it_there() -> None:
    """Looking 50° over the shoulder for two seconds swings the body there; breathing doesn't undo it."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=20)
    now = manager._now()
    manager._follow_with_body(now)
    manager._follow_with_body(now + 1.0)
    assert manager.state.current_move is None  # 20° is a glance, not a twisted neck

    manager, robot = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=50)
    manager._follow_with_body(now)
    manager._follow_with_body(now + 1.0)
    assert manager.state.current_move is None
    manager._follow_with_body(now + BODY_FOLLOW_DELAY_S + 0.1)
    move = manager.state.current_move
    assert isinstance(move, GotoQueueMove)
    head, _, body_yaw = move.evaluate(BODY_FOLLOW_DURATION_S)
    assert np.allclose(head, robot.get_current_head_pose.return_value)
    assert body_yaw == pytest.approx(np.radians(50))

    manager.state.move_start_time = now + 0.6
    manager._get_primary_pose(now + 0.6 + BODY_FOLLOW_DURATION_S)
    assert move.evaluate(0.1 * BODY_FOLLOW_DURATION_S)[2] < np.radians(5)  # eases off, no jerk
    manager.state.current_move = BreathingMove(np.eye(4), (0.0, 0.0))
    assert manager._get_primary_pose(now + 5.0)[2] == pytest.approx(np.radians(50))
    manager.state.current_move = None
    assert manager._get_primary_pose(now + 6.0)[2] == pytest.approx(np.radians(50))


def test_a_face_the_head_cannot_reach_pauses_tracking_holds_the_pose_and_retries_later() -> None:
    """A face far below centre with the head not moving is a joint limit: stop pushing, hold, try again later."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0, y=0.7))
    held_pose = create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
    robot.get_current_head_pose.return_value = held_pose
    now = manager._now()
    manager._guard_limits(now)
    manager._guard_limits(now + 0.3)
    robot.start_head_tracking.assert_not_called()
    manager._guard_limits(now + 0.7)
    robot.start_head_tracking.assert_called_once_with(weight=0.0)
    assert manager.get_status()["at_limit"] is True

    # Breathing would drift the head back to neutral; the held pose wins.
    manager.state.current_move = BreathingMove(held_pose, (0.0, 0.0))
    manager.state.move_start_time = now
    head, _, _ = manager._get_primary_pose(now + 2.0)
    assert np.allclose(head, held_pose)

    robot.start_head_tracking.reset_mock()
    resumed_at = now + 0.7 + LIMIT_HOLD_S
    manager._guard_limits(resumed_at)
    robot.start_head_tracking.assert_called_once_with(weight=1.0)
    assert manager.get_status()["at_limit"] is False

    # Stuck again straight away: the next hold is twice as long.
    manager._guard_limits(resumed_at + 0.1)
    manager._guard_limits(resumed_at + 0.8)
    assert manager._limit_hold_until == pytest.approx(resumed_at + 0.8 + 2 * LIMIT_HOLD_S)


def test_tracking_keeps_going_while_the_head_still_moves_or_is_not_over_to_the_face_side() -> None:
    """A head still moving, settled near neutral (faces sit a little high), or pitched the other way is no limit."""
    face = _Face(detected=True, x=0.0, y=0.7)
    manager, robot = _tracking_manager(face)
    now = manager._now()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 32, 0, degrees=True)
    manager._guard_limits(now)
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 36, 0, degrees=True)
    manager._guard_limits(now + 0.7)
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 40, 0, degrees=True)
    manager._guard_limits(now + 1.4)
    robot.start_head_tracking.assert_not_called()

    # Converged on the real robot: face 0.38 above centre, head level and still for ages.
    face.y = -0.38
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, -2, 0, degrees=True)
    for seconds in (2.0, 3.0, 4.0, 10.0):
        manager._guard_limits(now + seconds)
    robot.start_head_tracking.assert_not_called()

    # Pitched well down but the face is above: the head can come back up, nothing is blocked.
    face.y = -0.5
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 35, 0, degrees=True)
    for seconds in (11.0, 12.0, 13.0):
        manager._guard_limits(now + seconds)
    robot.start_head_tracking.assert_not_called()


def test_a_neck_at_its_limit_with_the_face_further_round_is_a_limit_too() -> None:
    """Looking 55° left of the body with the face still further left (image x < 0) and the head still."""
    manager, robot = _tracking_manager(_Face(detected=True, x=-0.4), head_yaw_deg=55)
    now = manager._now()
    manager._guard_limits(now)
    manager._guard_limits(now + 0.7)
    robot.start_head_tracking.assert_called_once_with(weight=0.0)


def test_dozing_cancels_a_turn_and_listening_notifies_attention() -> None:
    """Dozing mid-turn must not resume tracking afterwards; listening toggles reach speaker attention."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0))
    now = manager._now()
    manager._handle_command("look_toward", np.radians(45), now)
    assert manager.get_status()["turning"] is True
    manager._handle_command("set_dozing", True, now)
    robot.start_head_tracking.reset_mock()
    manager._finish_turn(now + 1.0)
    robot.start_head_tracking.assert_not_called()

    heard: list[bool] = []
    manager.on_listening = heard.append
    manager.set_listening(True)
    assert heard == [True]


def test_a_thermal_hold_dozes_ignores_wake_ups_and_restores_the_requested_state() -> None:
    """Hot motors force the doze pose; waking waits for the release, which restores what was last asked for."""
    manager, _ = _tracking_manager(_Face(detected=False))
    now = manager._now()

    manager._handle_command("thermal_hold", True, now)
    assert manager.get_status()["dozing"] is True
    assert manager.get_status()["thermal_hold"] is True

    manager._handle_command("set_dozing", False, now)
    manager._handle_command("lazy_wake", None, now)
    assert manager.get_status()["dozing"] is True

    manager._handle_command("thermal_hold", False, now)
    assert manager.get_status()["dozing"] is False
    assert manager.get_status()["thermal_hold"] is False

    manager._handle_command("set_dozing", True, now)
    manager._handle_command("thermal_hold", True, now)
    manager._handle_command("thermal_hold", False, now)
    assert manager.get_status()["dozing"] is True


def _hold_neck(manager: MovementManager, start: float, seconds: float) -> None:
    for step in range(int(seconds * 10) + 1):
        manager._follow_with_body(start + step * 0.1)


def test_the_body_follows_only_a_settled_centred_face() -> None:
    """A twisted neck still swinging toward the face (face off-centre) or a face flicker never turns the body."""
    face = _Face(detected=True, x=0.5)
    manager, robot = _tracking_manager(face, head_yaw_deg=50)
    now = manager._now()
    _hold_neck(manager, now, 3.0)
    assert manager.state.current_move is None

    # Centred, but the face drops out halfway through the wait: the wait starts over.
    face.x = 0.0
    manager._follow_with_body(now + 10.0)
    manager._follow_with_body(now + 11.0)
    face.detected = False
    manager._follow_with_body(now + 11.5)
    face.detected = True
    manager._follow_with_body(now + 12.1)
    assert manager.state.current_move is None
    manager._follow_with_body(now + 12.1 + BODY_FOLLOW_DELAY_S)
    assert isinstance(manager.state.current_move, GotoQueueMove)


def test_the_body_rests_after_a_turn_and_never_swings_straight_back() -> None:
    """The overnight ping-pong: body to -60°, head back to the world front, body back to 0 three seconds later."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=-60)
    now = manager._now()
    _hold_neck(manager, now, BODY_FOLLOW_DELAY_S + 0.1)
    assert manager.state.current_move.evaluate(BODY_FOLLOW_DURATION_S)[2] == pytest.approx(np.radians(-60))
    turned_at = now + BODY_FOLLOW_DELAY_S
    manager.state.current_move = None
    manager._body_turn_until = None

    # Same way again, but still resting.
    robot.get_current_joint_positions.return_value = ([np.radians(-60)] + [0.0] * 6, [0.0, 0.0])
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, -105, degrees=True)
    _hold_neck(manager, turned_at + 1.0, BODY_FOLLOW_COOLDOWN_S - 2.0)
    assert manager.state.current_move is None

    # The other way: not for half a minute.
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, -5, degrees=True)
    _hold_neck(manager, turned_at + BODY_FOLLOW_COOLDOWN_S, BODY_REVERSE_COOLDOWN_S - BODY_FOLLOW_COOLDOWN_S - 1.0)
    assert manager.state.current_move is None
    _hold_neck(manager, turned_at + BODY_REVERSE_COOLDOWN_S, BODY_FOLLOW_DELAY_S + 0.2)
    assert manager.state.current_move.evaluate(BODY_FOLLOW_DURATION_S)[2] == pytest.approx(np.radians(-5))


def test_a_lost_face_leaves_the_head_where_it_is_until_a_face_comes_back() -> None:
    """Before the daemon would recentre on the world front, re-arm tracking and hold the current pose."""
    face = _Face(detected=True, x=0.0)
    manager, robot = _tracking_manager(face, head_yaw_deg=50, body_yaw_deg=50)
    now = manager._now()
    manager._watch_for_face(now)
    face.detected = False
    manager._watch_for_face(now + 0.5)
    robot.start_head_tracking.assert_not_called()  # a short gap: the daemon holds its last target itself

    manager._watch_for_face(now + FACE_LOST_HOLD_S)
    assert robot.start_head_tracking.call_args_list == [call(weight=0.0), call(weight=1.0)]
    held = robot.get_current_head_pose.return_value
    manager.state.current_move = BreathingMove(np.eye(4), (0.0, 0.0))
    manager.state.move_start_time = now
    assert np.allclose(manager._get_primary_pose(now + 5.0)[0], held)

    # Nobody for ages: no further re-arming, the pose just holds.
    manager._watch_for_face(now + 10.0)
    assert robot.start_head_tracking.call_count == 2

    face.detected = True
    manager._watch_for_face(now + 11.0)
    assert manager._watch_anchor is None


def test_speech_end_holds_the_head_until_the_tracker_has_the_face_again() -> None:
    """Handing back after speaking must not drop the head to the breathing pose for a frame or two."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0), head_yaw_deg=30)
    now = manager._now()
    manager._handle_command("set_speaking", True, now)
    anchor = manager._track_anchor
    manager._handle_command("set_speaking", False, now + 2.0)
    assert manager._track_anchor is None
    assert manager._watch_anchor is anchor
    robot.start_head_tracking.assert_called_with(weight=1.0)


def test_a_limit_hold_hands_back_without_dropping_the_pose() -> None:
    """When a limit hold ends, the held pose stays until the tracker sees the face again."""
    manager, robot = _tracking_manager(_Face(detected=True, x=0.0, y=0.7))
    held_pose = create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
    robot.get_current_head_pose.return_value = held_pose
    now = manager._now()
    manager._guard_limits(now)
    manager._guard_limits(now + 0.7)
    manager._guard_limits(now + 0.7 + LIMIT_HOLD_S)
    assert manager._limit_anchor is None
    assert np.allclose(manager._watch_anchor, held_pose)


def _quiet_manager(face: _Face) -> tuple[MovementManager, MagicMock, float]:
    """Body turned 50° left, head on it, face lost and held; returns the moment the last face was seen."""
    manager, robot = _tracking_manager(face, head_yaw_deg=50, body_yaw_deg=50)
    now = manager._now()
    manager._watch_for_face(now)
    face.detected = False
    manager._watch_for_face(now + FACE_LOST_HOLD_S)
    assert manager._watch_anchor is not None
    return manager, robot, now


def test_after_a_quiet_spell_head_and_body_ease_back_to_the_front_together() -> None:
    """Hold first; once nobody has been seen or heard for a while, return slowly to the front, neck untwisted."""
    face = _Face(detected=True, x=0.0)
    manager, robot, seen_at = _quiet_manager(face)
    manager._watch_for_face(seen_at + RECENTER_QUIET_S - 1.0)
    assert manager.state.current_move is None  # still holding where the person was

    # Someone talks (no face in view): the quiet spell starts over.
    manager._handle_command("set_listening", True, seen_at + RECENTER_QUIET_S - 0.5)
    manager._handle_command("set_listening", False, seen_at + RECENTER_QUIET_S)
    manager._watch_for_face(seen_at + RECENTER_QUIET_S + 1.0)
    assert manager.state.current_move is None

    start = seen_at + 2 * RECENTER_QUIET_S + 0.1
    manager._watch_for_face(start)
    move = manager.state.current_move
    assert isinstance(move, GotoQueueMove)
    assert manager._watch_anchor is None
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        head, _, body_yaw = move.evaluate(fraction * RECENTER_DURATION_S)
        head_deg = float(np.degrees(np.arctan2(head[1, 0], head[0, 0])))
        assert head_deg == pytest.approx(np.degrees(body_yaw), abs=1.0)  # the neck stays straight all the way
    assert move.evaluate(0.1 * RECENTER_DURATION_S)[2] > np.radians(48)  # starts gently
    assert move.evaluate(RECENTER_DURATION_S)[2] == pytest.approx(0.0)

    # Once is enough: no new recentre while it stays quiet.
    manager._watch_for_face(start + RECENTER_DURATION_S)
    manager.state.current_move = None
    manager._watch_for_face(start + RECENTER_DURATION_S + 60.0)
    assert manager.state.current_move is None


def test_a_face_showing_up_while_recentring_stops_it_and_tracking_takes_over() -> None:
    """Someone walks up mid-recentre: stop where the head is and let the tracker ease on from there."""
    face = _Face(detected=True, x=0.0)
    manager, robot, seen_at = _quiet_manager(face)
    start = seen_at + RECENTER_QUIET_S + 0.1
    manager._watch_for_face(start)
    assert isinstance(manager.state.current_move, GotoQueueMove)
    face.detected = True
    manager._watch_for_face(start + 1.0)
    assert manager.state.current_move is None
    assert manager._recenter_until is None


def test_no_recentring_while_a_face_is_in_view_or_something_is_playing() -> None:
    """A person sitting there quietly is still company; a move in progress is never cut short."""
    face = _Face(detected=True, x=0.0)
    manager, robot = _tracking_manager(face, head_yaw_deg=50, body_yaw_deg=50)
    now = manager._now()
    for seconds in (0.0, RECENTER_QUIET_S + 1.0, 2 * RECENTER_QUIET_S):
        manager._watch_for_face(now + seconds)
    assert manager.state.current_move is None  # someone is sitting there quietly: keep looking at them

    manager, robot, seen_at = _quiet_manager(_Face(detected=True, x=0.0))
    playing = _FakeMove(np.eye(4))
    manager.state.current_move = playing
    manager._watch_for_face(seen_at + RECENTER_QUIET_S + 1.0)
    assert manager.state.current_move is playing
