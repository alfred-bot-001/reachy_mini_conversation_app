from __future__ import annotations
from typing import Any
from dataclasses import replace

import reachy_mini_conversation_app.idle_policy as idle_policy_mod
import reachy_mini_conversation_app.tools.play_emotion as play_emotion_mod
from reachy_mini_conversation_app.tools.dance import Dance
from reachy_mini_conversation_app.tools.move_head import MoveHead
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion
from reachy_mini_conversation_app.tools.idle_do_nothing import IdleDoNothing


class FakeIdleTool(Tool):
    """Tool with an idle tool name but not the idle tool class."""

    _auto_register = False
    name = "idle_do_nothing"
    description = "Fake idle tool with the same public name."
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return a fake result."""
        return {"status": "fake"}


def test_choose_idle_tool_call_uses_registered_name_for_matching_class() -> None:
    """A matching idle class should use the registered tool instance name."""
    tool = IdleDoNothing()
    tool.name = "profile_idle_do_nothing"

    selected = idle_policy_mod.choose_idle_tool_call(
        ["profile_idle_do_nothing"],
        tool_registry={"profile_idle_do_nothing": tool},
    )

    assert selected == ("profile_idle_do_nothing", {"reason": "random idle policy selected stillness"})


def test_choose_idle_tool_call_rejects_same_name_with_unmatched_class() -> None:
    """A matching public name alone should not make a tool eligible."""
    selected = idle_policy_mod.choose_idle_tool_call(
        ["idle_do_nothing"],
        tool_registry={"idle_do_nothing": FakeIdleTool()},
    )

    assert selected is None


def test_choose_idle_tool_call_keeps_args_with_weighted_candidate(monkeypatch) -> None:
    """Argument generation should stay attached to the weighted candidate."""
    monkeypatch.setattr(
        idle_policy_mod,
        "_IDLE_TOOL_CANDIDATES",
        tuple(
            replace(candidate, weight=1.0) if candidate.tool_type is MoveHead else candidate
            for candidate in idle_policy_mod._IDLE_TOOL_CANDIDATES
        ),
    )
    monkeypatch.setattr(idle_policy_mod.random, "choice", lambda _choices: "right")

    selected = idle_policy_mod.choose_idle_tool_call(
        ["move_head"],
        tool_registry={"move_head": MoveHead()},
    )

    assert selected == ("move_head", {"direction": "right"})


def test_default_idle_policy_only_selects_stillness() -> None:
    """Without interaction the default policy keeps the robot still even when motion tools are available."""
    registry: dict[str, Tool] = {
        "idle_do_nothing": IdleDoNothing(),
        "dance": Dance(),
        "play_emotion": PlayEmotion(),
        "move_head": MoveHead(),
    }

    for _ in range(50):
        selected = idle_policy_mod.choose_idle_tool_call(registry.keys(), tool_registry=registry)
        assert selected == ("idle_do_nothing", {"reason": "random idle policy selected stillness"})


def test_choose_idle_tool_call_skips_zero_weight_candidates() -> None:
    """Zero-weight idle tools are never selected, and selecting among none of them does not raise."""
    selected = idle_policy_mod.choose_idle_tool_call(
        ["dance", "play_emotion", "move_head"],
        tool_registry={"dance": Dance(), "play_emotion": PlayEmotion(), "move_head": MoveHead()},
    )

    assert selected is None


def test_idle_policy_never_dances_even_when_every_candidate_is_enabled(monkeypatch) -> None:
    """Dance is never an idle action, even if all idle weights are raised or a candidate type would match it."""
    monkeypatch.setattr(
        idle_policy_mod,
        "_IDLE_TOOL_CANDIDATES",
        tuple(replace(candidate, weight=1.0) for candidate in idle_policy_mod._IDLE_TOOL_CANDIDATES)
        + (idle_policy_mod.IdleToolCandidate(Tool, 1.0, lambda: {}),),
    )
    registry: dict[str, Tool] = {
        "idle_do_nothing": IdleDoNothing(),
        "dance": Dance(),
        "play_emotion": PlayEmotion(),
        "move_head": MoveHead(),
    }

    for _ in range(200):
        selected = idle_policy_mod.choose_idle_tool_call(registry.keys(), tool_registry=registry)
        assert selected is not None
        assert selected[0] != "dance"

    assert idle_policy_mod.choose_idle_tool_call(["dance"], tool_registry={"dance": Dance()}) is None


def test_idle_emotion_only_requests_calm_intents(monkeypatch) -> None:
    """A re-enabled idle emotion never asks for an intent that resolves to a dance emotion."""
    monkeypatch.setattr(
        idle_policy_mod,
        "_IDLE_TOOL_CANDIDATES",
        tuple(
            replace(candidate, weight=1.0) if candidate.tool_type is PlayEmotion else candidate
            for candidate in idle_policy_mod._IDLE_TOOL_CANDIDATES
        ),
    )

    for _ in range(50):
        selected = idle_policy_mod.choose_idle_tool_call(
            ["play_emotion"],
            tool_registry={"play_emotion": PlayEmotion()},
        )
        assert selected is not None
        tool_name, arguments = selected
        assert tool_name == "play_emotion"
        assert arguments["emotion"] in idle_policy_mod._IDLE_EMOTION_INTENTS
        for move in play_emotion_mod._INTENT_TO_MOVES[arguments["emotion"]]:
            assert "dance" not in move
