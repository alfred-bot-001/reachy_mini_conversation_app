"""Motor temperature log and watchdog.

The SDK exposes no motor temperatures, so the Dynamixel registers are read through the daemon's raw-packet
websocket. Every sample goes to a rotating JSON-lines log together with what the head was doing, so a later
overheat can be traced back to the motion behind it. Hot motors rest the head in the doze pose; critically hot
ones (or a motor that has already latched its overheating error) get their torque turned off.
"""

import json
import math
import time
import struct
import logging
import threading
from typing import Protocol
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

import httpx
from websockets.sync.client import ClientConnection, connect

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager, rotation_between


logger = logging.getLogger(__name__)

# Motor IDs on the Reachy Mini bus.
MOTOR_IDS = {
    "body_rotation": 10,
    "stewart_1": 11,
    "stewart_2": 12,
    "stewart_3": 13,
    "stewart_4": 14,
    "stewart_5": 15,
    "stewart_6": 16,
    "left_antenna": 17,
    "right_antenna": 18,
}

# XL330 control table: present current (126) through present temperature (146) in one read.
_TELEMETRY_ADDRESS = 126
_TELEMETRY_LENGTH = 21
_HARDWARE_ERROR_ADDRESS = 70
_ALERT_BIT = 0x80  # set in a status packet's error byte while the motor has a hardware error
OVERHEATING_ERROR = 0x04
INPUT_VOLTAGE_ERROR = 0x01
# The motors latch the input-voltage flag spuriously; like the daemon, trust it only above this voltage.
_MAX_PLAUSIBLE_VOLTAGE_V = 7.8

# The XL330 shuts itself down at 70 °C.
WARM_C = 50  # sample faster from here
HOT_C = 55  # rest in the doze pose
CRITICAL_C = 65  # turn the torque off
COOL_C = 48  # every motor at or below this ends the hold

POLL_S = 30.0
WARM_POLL_S = 10.0
# Bus reads can collide with the daemon's control loop and drop a frame, so they wait for the head to be still.
IDLE_RETRY_S = 5.0
STILL_RAD = math.radians(1.5)
MAX_UNSAMPLED_S = 300.0  # read anyway after this long without a still moment
WARM_MAX_UNSAMPLED_S = 60.0
_READ_FAILURE_LOG_S = 600.0

LOG_FILENAME = "motor_health.jsonl"
_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUPS = 3


@dataclass(frozen=True)
class MotorReading:
    """One motor's telemetry sample."""

    temperature_c: int
    current_ma: int
    voltage_v: float
    hardware_error: int


class MotorBus(Protocol):
    """Where motor telemetry comes from."""

    def read_all(self) -> dict[str, MotorReading]:
        """Read every motor."""
        ...

    def is_moving(self) -> bool:
        """Whether the daemon itself is playing a move."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
    return crc


def read_packet(motor_id: int, address: int, length: int) -> bytes:
    """Build a Dynamixel protocol 2.0 READ instruction packet."""
    params = struct.pack("<HH", address, length)
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, motor_id]) + struct.pack("<H", len(params) + 3) + b"\x02" + params
    return body + struct.pack("<H", _crc16(body))


def parse_status(packet: bytes, motor_id: int, length: int) -> tuple[int, bytes]:
    """Return the error byte and ``length`` data bytes of ``motor_id``'s status packet; ValueError otherwise."""
    if len(packet) < 11 + length or packet[:4] != b"\xff\xff\xfd\x00" or packet[7] != 0x55:
        raise ValueError(f"not a status packet: {packet.hex()}")
    if packet[4] != motor_id:
        raise ValueError(f"reply from motor {packet[4]}, expected motor {motor_id}")
    end = 9 + length
    if struct.unpack("<H", packet[end : end + 2])[0] != _crc16(packet[:end]):
        raise ValueError(f"bad status packet CRC: {packet.hex()}")
    return packet[8], packet[9:end]


class DaemonMotorBus:
    """Read motor registers through the daemon's ``/api/move/ws/raw/write`` websocket."""

    def __init__(self, daemon_url: str) -> None:
        """Talk to the daemon at ``daemon_url`` (http base URL)."""
        self._daemon_url = daemon_url.rstrip("/")
        self._raw_url = "ws" + self._daemon_url.removeprefix("http") + "/api/move/ws/raw/write"
        self._http = httpx.Client(timeout=2.0, trust_env=False)
        self._ws: ClientConnection | None = None

    def _read(self, motor_id: int, address: int, length: int) -> tuple[int, bytes]:
        if self._ws is None:
            self._ws = connect(self._raw_url, open_timeout=3.0, max_size=None, ping_interval=None)
        self._ws.send(read_packet(motor_id, address, length))
        reply = self._ws.recv(timeout=2.0)
        if isinstance(reply, str):
            raise ValueError("expected a binary status packet")
        return parse_status(reply, motor_id, length)

    def read_all(self) -> dict[str, MotorReading]:
        """Read every motor; any failure drops the connection so the next call reconnects."""
        try:
            readings = {}
            for name, motor_id in MOTOR_IDS.items():
                error, data = self._read(motor_id, _TELEMETRY_ADDRESS, _TELEMETRY_LENGTH)
                hardware_error = self._read(motor_id, _HARDWARE_ERROR_ADDRESS, 1)[1][0] if error & _ALERT_BIT else 0
                voltage_v = struct.unpack("<H", data[18:20])[0] / 10.0
                if voltage_v <= _MAX_PLAUSIBLE_VOLTAGE_V:
                    hardware_error &= ~INPUT_VOLTAGE_ERROR
                readings[name] = MotorReading(
                    temperature_c=data[20],
                    current_ma=struct.unpack("<h", data[0:2])[0],
                    voltage_v=voltage_v,
                    hardware_error=hardware_error,
                )
            return readings
        except Exception:
            self.close()
            raise

    def is_moving(self) -> bool:
        """Whether the daemon itself is playing a move."""
        response = self._http.get(self._daemon_url + "/api/move/running")
        response.raise_for_status()
        return bool(response.json())

    def close(self) -> None:
        """Drop the websocket."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class MotorHealthMonitor:
    """Log motor telemetry and protect hot motors."""

    def __init__(self, bus: MotorBus, robot: ReachyMini, movement_manager: MovementManager, log_dir: Path) -> None:
        """Sample ``bus``, log to ``log_dir`` and act through ``robot`` and ``movement_manager``."""
        self._bus = bus
        self._robot = robot
        self._movement_manager = movement_manager
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._log_handler = RotatingFileHandler(
            log_dir / LOG_FILENAME, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
        )
        self._sample_log = logging.getLogger(f"{__name__}.samples.{id(self)}")
        self._sample_log.propagate = False
        self._sample_log.setLevel(logging.INFO)
        self._sample_log.addHandler(self._log_handler)
        self._holding = False
        self._torque_off = False
        self._last_failure_log = -_READ_FAILURE_LOG_S
        self._last_sample_at = time.monotonic()
        self._warm = False
        self._last_pose = (robot.get_current_head_pose(), float(robot.get_current_joint_positions()[0][0]))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write(self, record: dict[str, object]) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        self._sample_log.info(json.dumps(record, separators=(",", ":")))

    def _event(self, event: str, hottest: tuple[str, int], level: int, message: str) -> None:
        logger.log(level, "Motor %s at %d °C: %s", hottest[0], hottest[1], message)
        self._write({"event": event, "motor": hottest[0], "temperature_c": hottest[1]})

    def poll_once(self) -> float:
        """Take one sample, act on it and return the seconds until the next one."""
        try:
            if self._bus.is_moving():
                return IDLE_RETRY_S
            head, body_yaw = (
                self._robot.get_current_head_pose(),
                float(self._robot.get_current_joint_positions()[0][0]),
            )
            last_head, last_body_yaw = self._last_pose
            self._last_pose = (head, body_yaw)
            still = rotation_between(last_head, head) <= STILL_RAD and abs(body_yaw - last_body_yaw) <= STILL_RAD
            overdue = time.monotonic() - self._last_sample_at >= (
                WARM_MAX_UNSAMPLED_S if self._warm else MAX_UNSAMPLED_S
            )
            if not (still or overdue):
                return IDLE_RETRY_S
            readings = self._bus.read_all()
        except Exception as e:
            if time.monotonic() - self._last_failure_log >= _READ_FAILURE_LOG_S:
                logger.warning("Could not read motor temperatures: %s", e)
                self._last_failure_log = time.monotonic()
            return POLL_S
        self._last_sample_at = time.monotonic()

        status = self._movement_manager.get_status()
        motion_keys = ("dozing", "thermal_hold", "head_tracking", "turning", "at_limit", "queue_size")
        record: dict[str, object] = {
            "temperature_c": {name: r.temperature_c for name, r in readings.items()},
            "current_ma": {name: r.current_ma for name, r in readings.items()},
            "voltage_v": {name: r.voltage_v for name, r in readings.items()},
            "motion": {key: status[key] for key in motion_keys if key in status},
        }
        errors = {name: r.hardware_error for name, r in readings.items() if r.hardware_error}
        if errors:
            record["hardware_error"] = errors
        self._write(record)

        hottest = max(((name, r.temperature_c) for name, r in readings.items()), key=lambda item: item[1])
        overheated = [name for name, r in readings.items() if r.hardware_error & OVERHEATING_ERROR]
        if (hottest[1] >= CRITICAL_C or overheated) and not self._torque_off:
            self._event("critical", hottest, logging.ERROR, f"turning the motors off (overheating flag: {overheated})")
            self._movement_manager.set_thermal_hold(True)
            self._holding = True
            self._robot.disable_motors()
            self._torque_off = True
        elif hottest[1] >= HOT_C and not self._holding:
            self._event("hot", hottest, logging.WARNING, "resting in the doze pose until it cools")
            self._movement_manager.set_thermal_hold(True)
            self._holding = True
        elif self._holding and hottest[1] <= COOL_C and not overheated:
            self._event("cooled", hottest, logging.INFO, "resuming")
            if self._torque_off:
                self._robot.enable_motors()
                self._torque_off = False
            self._movement_manager.set_thermal_hold(False)
            self._holding = False
        self._warm = hottest[1] >= WARM_C
        return WARM_POLL_S if self._warm else POLL_S

    def _run(self) -> None:
        delay = IDLE_RETRY_S
        while not self._stop.wait(delay):
            try:
                delay = self.poll_once()
            except Exception as e:
                logger.error("Motor watchdog step failed: %s", e)
                delay = POLL_S

    def start(self) -> None:
        """Start sampling in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="motor-health")
        self._thread.start()

    def close(self) -> None:
        """Stop sampling and close the bus and the log."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._bus.close()
        self._sample_log.removeHandler(self._log_handler)
        self._log_handler.close()
