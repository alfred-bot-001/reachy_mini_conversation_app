"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


def test_motor_log_dir_defaults_outside_the_app_and_can_be_overridden(monkeypatch, tmp_path) -> None:
    """The log lives under agent-data unless REACHY_MINI_MOTOR_LOG_DIR points elsewhere."""
    monkeypatch.delenv(config.MOTOR_LOG_DIR_ENV, raising=False)
    assert config.resolve_motor_log_dir().parts[-3:] == ("agent-data", "reachy-mini", "motor-health")
    monkeypatch.setenv(config.MOTOR_LOG_DIR_ENV, str(tmp_path))
    assert config.resolve_motor_log_dir() == tmp_path


def test_motor_watchdog_is_off_unless_asked_for(monkeypatch) -> None:
    """Its raw bus reads can freeze the daemon, so it only runs when explicitly enabled."""
    monkeypatch.delenv(config.MOTOR_WATCHDOG_ENV, raising=False)
    assert config.motor_watchdog_enabled() is False
    monkeypatch.setenv(config.MOTOR_WATCHDOG_ENV, "1")
    assert config.motor_watchdog_enabled() is True
