import math
import logging
from typing import Any, Dict, Tuple

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# 0° is upright; folding goes backwards. The doze pose stops at 2.9 rad (166°) short of the mechanical stop that
# overloaded the left antenna motor on 2026-11-05, so requests stop a little earlier still.
MAX_FOLD_DEG = 165.0
SIDES = ("both", "left", "right")


def joint_targets(angle_deg: float, side: str, present: Tuple[float, float]) -> Tuple[float, float]:
    """Joint angles in radians for ``angle_deg`` on ``side``; the other antenna keeps its present angle.

    The two antennas fold in mirror image: the left motor goes negative, the right one positive.
    """
    rad = math.radians(angle_deg)
    left = -rad if side in ("both", "left") else float(present[0])
    right = rad if side in ("both", "right") else float(present[1])
    return (left, right)


class SetAntennas(Tool):
    """Put the antennas at an angle and keep them there."""

    name = "set_antennas"
    description = (
        "Move your two antennas to an angle and hold them there: 0 = standing straight up, 90 = horizontal, "
        f"{MAX_FOLD_DEG:.0f} = folded flat backwards. Use when someone asks you to fold, flatten, raise, "
        "straighten or angle your antennas. Call get_antennas afterwards if asked whether it worked."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "angle_deg": {
                "type": "number",
                "minimum": 0,
                "maximum": MAX_FOLD_DEG,
                "description": f"Angle from upright in degrees, 0 (up) to {MAX_FOLD_DEG:.0f} (flat backwards).",
            },
            "side": {
                "type": "string",
                "enum": list(SIDES),
                "description": "Which antenna to move; default both.",
            },
        },
        "required": ["angle_deg"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Hold the antennas at the requested angle (the movement manager eases them there)."""
        angle_raw = kwargs.get("angle_deg")
        if isinstance(angle_raw, bool) or not isinstance(angle_raw, (int, float)) or math.isnan(float(angle_raw)):
            return {"error": "angle_deg must be a number"}
        side = kwargs.get("side") or "both"
        if side not in SIDES:
            return {"error": "side must be one of both, left, right"}
        angle = min(MAX_FOLD_DEG, max(0.0, float(angle_raw)))
        logger.info("Tool call: set_antennas angle_deg=%.0f side=%s", angle, side)
        try:
            _, present = deps.reachy_mini.get_current_joint_positions()
            targets = joint_targets(angle, side, (float(present[0]), float(present[1])))
            deps.movement_manager.set_antennas(targets)
            deps.movement_manager.set_moving_state(deps.motion_duration_s)
        except Exception as e:
            logger.error("set_antennas failed")
            return {"error": f"set_antennas failed: {type(e).__name__}: {e}"}
        result: Dict[str, Any] = {
            "status": f"antennas {side} moving to {angle:.0f} degrees",
            "angle_deg": angle,
            "side": side,
        }
        if angle != float(angle_raw):
            result["note"] = f"angle clamped to {angle:.0f} degrees"
        return result
