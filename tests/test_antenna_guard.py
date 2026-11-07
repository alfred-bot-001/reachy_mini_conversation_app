"""A blocked antenna is yielded before its motor overloads, retried with backoff, and released when the pose changes."""

import math
import logging
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.antenna_guard import (
    FREE,
    STALL_S,
    YIELD_S,
    GIVEN_UP,
    YIELDING,
    BACKOFF_RAD,
    YIELD_MAX_S,
    MAX_ATTEMPTS,
    STALL_ERROR_RAD,
    AntennaStallGuard,
)


TICK = 0.01  # the movement loop runs at 100 Hz


def _run(guard: AntennaStallGuard, target: float, present: float, start: float, seconds: float) -> tuple[float, float]:
    """Feed a constant target and reported position for ``seconds``; return (last command, time after)."""
    now, command = start, target
    for _ in range(int(round(seconds / TICK))):
        command = guard.command(target, present, now)
        now += TICK
    return command, now


def test_an_antenna_that_follows_its_target_is_never_held_back() -> None:
    """An antenna closing in on its target, even with some lag, is passed straight through."""
    guard = AntennaStallGuard("left")
    now = 100.0
    # Moving 1 rad/s towards the target with a 0.1 rad lag: off target, but closing in and moving.
    for i in range(300):
        target = -0.17 - 0.01 * i
        present = target + 0.1
        assert guard.command(target, present, now) == target
        now += TICK
    assert guard.state == FREE and guard.attempts == 0


def test_a_fast_move_with_steady_lag_is_not_a_stall() -> None:
    """A big lag during a fast sweep is normal as long as the antenna itself keeps moving."""
    guard = AntennaStallGuard("right")
    now = 0.0
    for i in range(200):
        target = 0.02 * i
        assert guard.command(target, target - 0.3, now) == target  # 0.3 rad behind but moving 2 rad/s
        now += TICK
    assert guard.state == FREE


def test_a_blocked_antenna_yields_backed_off_from_the_obstacle_and_logs_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Half a second stuck short of the target: hold just back from the obstacle and say so."""
    guard = AntennaStallGuard("left")
    target, stuck = -2.9, -2.6  # folding into the doze pose, caught 17° short
    with caplog.at_level(logging.WARNING, logger="reachy_mini_conversation_app.antenna_guard"):
        command, now = _run(guard, target, stuck, start=10.0, seconds=STALL_S + 2 * TICK)
    assert guard.state == YIELDING and guard.attempts == 1 and guard.blocked
    # Held just short of where it stopped, on the side it came from (away from the obstacle).
    assert command == pytest.approx(stuck + BACKOFF_RAD)
    assert "Antenna left blocked at -2.60 rad on the way to -2.90 rad; yielding for 3s (attempt 1/3)" in caplog.text
    # It keeps holding there until the retry is due.
    assert guard.command(target, stuck + BACKOFF_RAD, now + YIELD_S / 2) == pytest.approx(stuck + BACKOFF_RAD)


def test_stall_detection_waits_half_a_second_of_no_progress() -> None:
    """Being off target for less than the stall window changes nothing."""
    guard = AntennaStallGuard("left")
    command, _ = _run(guard, -2.9, -2.6, start=0.0, seconds=STALL_S - 2 * TICK)
    assert command == -2.9 and guard.state == FREE


def test_retries_with_doubling_waits_then_gives_up_until_the_pose_changes(caplog: pytest.LogCaptureFixture) -> None:
    """Retry after 3 s, then 6 s; after the third stall hold until the commanded pose moves on."""
    guard = AntennaStallGuard("left")
    target, stuck = -2.9, -2.6
    now = 0.0
    waits = []
    with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.antenna_guard"):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            _, now = _run(guard, target, stuck, start=now, seconds=STALL_S + 2 * TICK)
            assert guard.attempts == attempt
            if attempt < MAX_ATTEMPTS:
                assert guard.state == YIELDING
                waits.append(guard._retry_at - (now - TICK))
                now = guard._retry_at + TICK  # the retry passes the target through again
            else:
                assert guard.state == GIVEN_UP
        assert [round(w, 1) for w in waits] == [YIELD_S, min(2 * YIELD_S, YIELD_MAX_S)]
        assert "giving up until the pose changes (attempt 3/3)" in caplog.text
        # Given up: still held an hour later with the same target.
        assert guard.command(target, stuck, now + 3600) == pytest.approx(stuck + BACKOFF_RAD)
        # Waking moves the target far from the blocked one: released at once, counters cleared.
        assert guard.command(-0.9, stuck, now + 3601) == -0.9
    assert guard.state == FREE and guard.attempts == 0
    assert "Antenna left released: the pose changed" in caplog.text


def test_reaching_the_target_on_a_retry_clears_the_stall(caplog: pytest.LogCaptureFixture) -> None:
    """Arriving on a retry clears the attempt counter and the blocked flag."""
    guard = AntennaStallGuard("right")
    target = 2.9
    _, now = _run(guard, target, 2.6, start=0.0, seconds=STALL_S + 2 * TICK)
    assert guard.state == YIELDING
    with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.antenna_guard"):
        now = guard._retry_at + TICK
        assert guard.command(target, 2.6, now) == target  # retry: target handed back
        assert guard.command(target, target - STALL_ERROR_RAD / 2, now + 0.3) == target  # and this time it arrives
    assert guard.state == FREE and guard.attempts == 0 and not guard.blocked
    assert "Antenna right back on target after 1 stall(s)" in caplog.text


def test_a_hand_holding_the_antenna_makes_it_yield_then_it_returns_when_let_go() -> None:
    """A hand holding the antenna gets a compliant antenna; letting go brings it home on the next retry."""
    guard = AntennaStallGuard("left")
    target = -0.1745  # awake neutral
    held_at = -0.6  # someone pulled it back and holds it
    command, now = _run(guard, target, held_at, start=0.0, seconds=STALL_S + 2 * TICK)
    assert guard.state == YIELDING and command == pytest.approx(held_at - BACKOFF_RAD)
    # Let go before the retry: the guard still waits for the retry, then the antenna follows the target home.
    now = guard._retry_at + TICK
    assert guard.command(target, held_at, now) == target
    assert guard.command(target, target + 0.05, now + 0.2) == target
    assert guard.state == FREE


def test_movement_manager_runs_the_guard_on_both_antennas_and_reports_them() -> None:
    """The movement loop guards each antenna against the daemon's reported position and lists the held ones."""
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [-2.6, 2.9])  # left stuck short, right arrived
    manager = MovementManager(current_robot=robot)
    now = 50.0
    for _ in range(int(STALL_S / TICK) + 2):
        left, right = manager._guard_antennas((-2.9, 2.9), now)
        now += TICK
    assert left == pytest.approx(-2.6 + BACKOFF_RAD) and right == 2.9
    assert manager.get_status()["antenna_blocked"] == ["left"]


def test_movement_manager_passes_antennas_through_without_positions_or_with_the_guard_off() -> None:
    """No reported positions, or the guard switched off: the commanded antennas are untouched."""
    robot = MagicMock()
    robot.get_current_joint_positions.side_effect = RuntimeError("daemon away")
    manager = MovementManager(current_robot=robot)
    assert manager._guard_antennas((-2.9, 2.9), 1.0) == (-2.9, 2.9)
    off = MovementManager(current_robot=MagicMock(), antenna_guard=False)
    assert off._guard_antennas((-2.9, 2.9), 1.0) == (-2.9, 2.9)
    assert off.get_status()["antenna_blocked"] == []


def test_backoff_is_a_couple_of_degrees() -> None:
    """The yield backs off by a couple of degrees, not a visible swing."""
    assert math.degrees(BACKOFF_RAD) == pytest.approx(2.5)
