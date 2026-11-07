import math
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def fold_degrees(left_rad: float, right_rad: float) -> Dict[str, float]:
    """Present antenna angles as degrees from upright (positive = folded backwards), left and right."""
    return {"left_deg": round(-math.degrees(left_rad), 1), "right_deg": round(math.degrees(right_rad), 1)}


class GetAntennas(Tool):
    """Read where the antennas really are."""

    name = "get_antennas"
    description = (
        "Read the actual measured angle of each antenna: 0 = straight up, 90 = horizontal, about 165 = flat "
        "backwards. Use when asked what angle the antennas are at, whether they are flat or up, or whether a "
        "set_antennas call worked. Report the numbers; never guess an angle without calling this."
    )
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return measured antenna angles plus the held target and any stall the guard is yielding to."""
        logger.info("Tool call: get_antennas")
        try:
            _, present = deps.reachy_mini.get_current_joint_positions()
            result: Dict[str, Any] = {
                **fold_degrees(float(present[0]), float(present[1])),
                "note": "degrees from upright; larger means folded further backwards",
            }
        except Exception as e:
            logger.error("get_antennas failed")
            return {"error": f"get_antennas failed: {type(e).__name__}: {e}"}
        try:
            status = deps.movement_manager.get_status()
        except Exception:
            status = {}
        if isinstance(status, dict):
            hold = status.get("antenna_hold")
            if hold:
                result["target"] = fold_degrees(float(hold[0]), float(hold[1]))
            blocked = status.get("antenna_blocked") or []
            if blocked:
                result["blocked"] = list(blocked)
                result["note"] += (
                    "; a blocked antenna is being held back from its target so its motor does not overload"
                )
        return result
