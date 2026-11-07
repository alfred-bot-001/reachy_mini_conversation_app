import math
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.get_antennas import GetAntennas, fold_degrees


def _deps(status: dict | None = None) -> ToolDependencies:
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [-0.1745, 0.1745])
    manager = MagicMock()
    manager.get_status.return_value = status or {"antenna_hold": None, "antenna_blocked": []}
    return ToolDependencies(reachy_mini=robot, movement_manager=manager)


def test_fold_degrees_reads_both_motors_as_backward_folds() -> None:
    """Left negative and right positive both read as the same backward fold in degrees."""
    assert fold_degrees(-math.pi / 2, math.pi / 2) == {"left_deg": 90.0, "right_deg": 90.0}
    assert fold_degrees(0.0, 0.0) == {"left_deg": 0.0, "right_deg": 0.0}


@pytest.mark.asyncio
async def test_get_antennas_reports_measured_angles() -> None:
    """The answer comes from the robot's reported joints, not from whatever was last asked for."""
    result = await GetAntennas()(_deps())
    assert result["left_deg"] == 10.0 and result["right_deg"] == 10.0
    assert "target" not in result and "blocked" not in result


@pytest.mark.asyncio
async def test_get_antennas_includes_hold_target_and_stall() -> None:
    """The held target and a stalled antenna are reported alongside the measured angles."""
    deps = _deps({"antenna_hold": [-2.88, 2.88], "antenna_blocked": ["left"]})
    result = await GetAntennas()(deps)
    assert result["target"] == {"left_deg": 165.0, "right_deg": 165.0}
    assert result["blocked"] == ["left"] and "overload" in result["note"]


@pytest.mark.asyncio
async def test_get_antennas_reports_robot_errors() -> None:
    """A robot that cannot be read yields an error, not an invented angle."""
    deps = _deps()
    deps.reachy_mini.get_current_joint_positions.side_effect = RuntimeError("offline")
    assert await GetAntennas()(deps) == {"error": "get_antennas failed: RuntimeError: offline"}
