import math
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.set_antennas import MAX_FOLD_DEG, SetAntennas, joint_targets


def _deps() -> ToolDependencies:
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [-0.17, 0.17])
    return ToolDependencies(reachy_mini=robot, movement_manager=MagicMock(), motion_duration_s=1.5)


def test_joint_targets_mirror_the_two_motors() -> None:
    """Folding goes negative on the left motor and positive on the right; one side alone keeps the other."""
    assert joint_targets(90, "both", (-0.17, 0.17)) == pytest.approx((-math.pi / 2, math.pi / 2))
    assert joint_targets(0, "both", (-0.17, 0.17)) == (0.0, 0.0)
    assert joint_targets(90, "left", (-0.17, 0.17)) == pytest.approx((-math.pi / 2, 0.17))
    assert joint_targets(90, "right", (-0.17, 0.17)) == pytest.approx((-0.17, math.pi / 2))


@pytest.mark.asyncio
async def test_set_antennas_holds_both_flat() -> None:
    """Flat backwards is a hold on the movement manager, never a raw goto, so the stall guard stays in the loop."""
    deps = _deps()

    result = await SetAntennas()(deps, angle_deg=160)

    assert result == {"status": "antennas both moving to 160 degrees", "angle_deg": 160.0, "side": "both"}
    deps.movement_manager.set_antennas.assert_called_once()
    left, right = deps.movement_manager.set_antennas.call_args.args[0]
    assert (left, right) == pytest.approx((-math.radians(160), math.radians(160)))
    deps.movement_manager.set_moving_state.assert_called_once_with(1.5)


@pytest.mark.asyncio
async def test_set_antennas_clamps_and_rejects_bad_input() -> None:
    """Angles stop short of the mechanical stop; nonsense never reaches the robot."""
    deps = _deps()
    result = await SetAntennas()(deps, angle_deg=400, side="left")
    assert result["angle_deg"] == MAX_FOLD_DEG and result["note"] == f"angle clamped to {MAX_FOLD_DEG:.0f} degrees"
    left, right = deps.movement_manager.set_antennas.call_args.args[0]
    assert left == pytest.approx(-math.radians(MAX_FOLD_DEG)) and right == pytest.approx(0.17)

    deps = _deps()
    assert await SetAntennas()(deps, angle_deg="flat") == {"error": "angle_deg must be a number"}
    assert await SetAntennas()(deps) == {"error": "angle_deg must be a number"}
    assert await SetAntennas()(deps, angle_deg=10, side="middle") == {"error": "side must be one of both, left, right"}
    deps.movement_manager.set_antennas.assert_not_called()


@pytest.mark.asyncio
async def test_set_antennas_reports_robot_errors() -> None:
    """A robot that cannot be read yields an error instead of a hold."""
    deps = _deps()
    deps.reachy_mini.get_current_joint_positions.side_effect = RuntimeError("offline")
    result = await SetAntennas()(deps, angle_deg=90)
    assert result == {"error": "set_antennas failed: RuntimeError: offline"}
