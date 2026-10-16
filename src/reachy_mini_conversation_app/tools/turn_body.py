import math
import logging
from typing import Any, Dict, Literal

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "center"]


class TurnBody(Tool):
    """Rotate the whole body left or right (the head keeps its pose), or back to center."""

    name = "turn_body"
    description = (
        "Turn your whole body (not just the head) to the left or right, or back to the center. "
        "Use when someone asks you to turn your body, turn around, or face another direction."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "center"],
            },
        },
        "required": ["direction"],
    }

    # direction -> body yaw in degrees; positive is left (counter-clockwise seen from above)
    YAW_DEGREES: Dict[str, float] = {"left": 60.0, "right": -60.0, "center": 0.0}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Rotate the body to the requested side."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str) or direction_raw not in self.YAW_DEGREES:
            return {"error": "direction must be one of left, right, center"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: turn_body direction=%s", direction)

        try:
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
            current_body_yaw = float(head_joints[0])  # body yaw is the first head joint
            goto_move = GotoQueueMove(
                target_head_pose=current_head_pose,  # keep the head where it is
                start_head_pose=current_head_pose,
                target_antennas=(current_antennas[0], current_antennas[1]),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=math.radians(self.YAW_DEGREES[direction]),
                start_body_yaw=current_body_yaw,
                duration=deps.motion_duration_s,
            )
            deps.movement_manager.queue_move(goto_move)
            deps.movement_manager.set_moving_state(deps.motion_duration_s)
            return {"status": f"turned body {direction}"}
        except Exception as e:
            logger.error("turn_body failed")
            return {"error": f"turn_body failed: {type(e).__name__}: {e}"}
