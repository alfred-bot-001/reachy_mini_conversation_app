import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.motor_health import (
    HOT_C,
    COOL_C,
    CRITICAL_C,
    OVERHEATING_ERROR,
    MotorReading,
    MotorHealthMonitor,
    read_packet,
    parse_status,
)


class _FakeBus:
    def __init__(self) -> None:
        self.temperatures = {"stewart_1": 40, "stewart_2": 38}
        self.hardware_errors: dict[str, int] = {}
        self.moving = False
        self.fail = False
        self.reads = 0

    def read_all(self) -> dict[str, MotorReading]:
        if self.fail:
            raise OSError("daemon unreachable")
        self.reads += 1
        return {
            name: MotorReading(
                temperature_c=t, current_ma=20, voltage_v=7.4, hardware_error=self.hardware_errors.get(name, 0)
            )
            for name, t in self.temperatures.items()
        }

    def is_moving(self) -> bool:
        return self.moving

    def close(self) -> None:
        pass


def _monitor(tmp_path: Path) -> tuple[MotorHealthMonitor, _FakeBus, MagicMock, MagicMock]:
    bus, robot, movement_manager = _FakeBus(), MagicMock(), MagicMock()
    movement_manager.get_status.return_value = {"dozing": False, "head_tracking": True}
    return MotorHealthMonitor(bus, robot, movement_manager, tmp_path), bus, robot, movement_manager


def _log_lines(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "motor_health.jsonl").read_text().splitlines()]


def test_read_packet_matches_the_protocol_2_reference_example() -> None:
    """Robotis manual example: read 4 bytes at address 132 from ID 1."""
    assert read_packet(1, 132, 4) == bytes.fromhex("FFFFFD0001070002840004001D15")


def test_parse_status_returns_the_error_byte_and_data_and_rejects_bad_packets() -> None:
    """A status packet yields its error byte and data; truncated or corrupted ones are rejected."""
    status = bytes.fromhex("FFFFFD000107005580A600000ECD")
    assert parse_status(status, 3) == (0x80, bytes.fromhex("A60000"))
    with pytest.raises(ValueError):
        parse_status(status[:8], 3)
    with pytest.raises(ValueError):
        parse_status(status[:-1] + b"\x00", 3)


def test_each_sample_is_logged_with_the_motion_state(tmp_path: Path) -> None:
    """Every sample lands in the JSON-lines log with what the head was doing."""
    monitor, _, _, _ = _monitor(tmp_path)
    monitor.poll_once()
    monitor.close()

    (sample,) = _log_lines(tmp_path)
    assert sample["temperature_c"] == {"stewart_1": 40, "stewart_2": 38}
    assert sample["current_ma"]["stewart_1"] == 20
    assert sample["motion"]["head_tracking"] is True


def test_no_bus_reads_while_the_daemon_plays_a_move(tmp_path: Path) -> None:
    """Raw reads racing a daemon-side move have hung the daemon, so a running move skips the sample."""
    monitor, bus, _, _ = _monitor(tmp_path)
    bus.moving = True
    monitor.poll_once()
    assert bus.reads == 0


def test_hot_motors_hold_the_doze_pose_until_every_motor_has_cooled(tmp_path: Path) -> None:
    """A hot motor rests the head until every motor is back at or below the cool threshold."""
    monitor, bus, robot, movement_manager = _monitor(tmp_path)

    bus.temperatures["stewart_1"] = HOT_C
    monitor.poll_once()
    movement_manager.set_thermal_hold.assert_called_once_with(True)
    robot.disable_motors.assert_not_called()

    bus.temperatures["stewart_1"] = COOL_C + 1
    monitor.poll_once()
    movement_manager.set_thermal_hold.assert_called_once_with(True)

    bus.temperatures["stewart_1"] = COOL_C
    monitor.poll_once()
    movement_manager.set_thermal_hold.assert_called_with(False)
    monitor.close()
    assert [line["event"] for line in _log_lines(tmp_path) if "event" in line] == ["hot", "cooled"]


def test_critical_heat_turns_the_motors_off_and_back_on_once_cooled(tmp_path: Path) -> None:
    """Critical heat cuts the torque, which comes back once the motors cool."""
    monitor, bus, robot, movement_manager = _monitor(tmp_path)

    bus.temperatures["stewart_1"] = CRITICAL_C
    monitor.poll_once()
    movement_manager.set_thermal_hold.assert_called_once_with(True)
    robot.disable_motors.assert_called_once()

    bus.temperatures["stewart_1"] = COOL_C
    monitor.poll_once()
    robot.enable_motors.assert_called_once()
    movement_manager.set_thermal_hold.assert_called_with(False)


def test_an_overheating_error_keeps_the_motors_off_even_once_cool(tmp_path: Path) -> None:
    """The motor only clears its overheating flag on a power cycle, so torque stays off."""
    monitor, bus, robot, movement_manager = _monitor(tmp_path)
    bus.hardware_errors["stewart_1"] = OVERHEATING_ERROR

    monitor.poll_once()
    robot.disable_motors.assert_called_once()

    bus.temperatures["stewart_1"] = COOL_C - 5
    monitor.poll_once()
    robot.enable_motors.assert_not_called()
    movement_manager.set_thermal_hold.assert_called_once_with(True)


def test_a_failed_read_is_survived_and_retried(tmp_path: Path) -> None:
    """A failed bus read is skipped and the next poll reads again."""
    monitor, bus, _, movement_manager = _monitor(tmp_path)
    bus.fail = True
    monitor.poll_once()
    bus.fail = False
    monitor.poll_once()
    assert bus.reads == 1
    movement_manager.set_thermal_hold.assert_not_called()
