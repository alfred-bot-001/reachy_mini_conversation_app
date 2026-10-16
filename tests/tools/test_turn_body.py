import math
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.turn_body import TurnBody
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    robot = MagicMock()
    robot.get_current_head_pose.return_value = "head-pose"
    robot.get_current_joint_positions.return_value = ([0.1, 0, 0, 0, 0, 0, 0], [0.2, 0.3])
    return ToolDependencies(reachy_mini=robot, movement_manager=MagicMock(), motion_duration_s=1.5)


@pytest.mark.asyncio
async def test_turn_body_queues_a_body_yaw_move_and_keeps_the_head() -> None:
    """Turning rotates only the body: the head and antennas interpolate to where they already are."""
    deps = _deps()

    result = await TurnBody()(deps, direction="left")

    assert result == {"status": "turned body left"}
    move = deps.movement_manager.queue_move.call_args.args[0]
    assert move.target_body_yaw == pytest.approx(math.radians(60))
    assert move.start_body_yaw == pytest.approx(0.1)
    assert move.target_head_pose == "head-pose" and move.start_head_pose == "head-pose"
    assert move.target_antennas == (0.2, 0.3)
    deps.movement_manager.set_moving_state.assert_called_once_with(1.5)


@pytest.mark.asyncio
async def test_turn_body_center_and_bad_direction() -> None:
    """Center resets the yaw; unknown or missing directions are rejected without moving."""
    deps = _deps()
    result = await TurnBody()(deps, direction="center")
    assert result == {"status": "turned body center"}
    assert deps.movement_manager.queue_move.call_args.args[0].target_body_yaw == 0

    assert await TurnBody()(deps, direction="backwards") == {"error": "direction must be one of left, right, center"}
    assert await TurnBody()(deps) == {"error": "direction must be one of left, right, center"}
