"""Yield a blocked antenna before its motor overloads.

An XL330 that keeps pushing against something it cannot move trips its Overload protection within a few seconds,
cuts its own torque and latches the error until the robot is power-cycled (the left antenna did exactly that on
2026-11-05 while folding into the doze pose, and hung limp for two days). Motor current cannot be read without the
raw bus reads that have frozen the daemon, so this guard works from what the movement loop already has every tick:
the position it commands and the position the daemon reports.

Per antenna: when the commanded position sits well off the reported one, the gap stops shrinking and the antenna
stands still for half a second, the motor is judged blocked. The guard then commands the antenna to stay where it
stopped (backed off a couple of degrees), logs a WARNING, and retries the original target after 3 s, then 6 s,
then 12 s. After three stalls it holds until the pose changes (waking, dozing), so a cable caught behind the antenna
does not turn into a slow bake. A fixed log wording lets the household console pick the stall up as an alert.
"""

from __future__ import annotations
import math
import logging


logger = logging.getLogger(__name__)

STALL_ERROR_RAD = math.radians(9)  # commanded and reported positions this far apart: not following
STALL_PROGRESS_RAD = math.radians(0.6)  # the gap must shrink by this much to count as still moving in
STALL_STILL_RAD = math.radians(1.5)  # the antenna itself moved less than this: standing still
STALL_S = 0.5  # off target, no progress and still for this long: blocked
BACKOFF_RAD = math.radians(2.5)  # yield to just short of where it stopped, so it is not leaning on the obstacle
YIELD_S = 3.0  # first retry after this long; doubles each stall
YIELD_MAX_S = 12.0
MAX_ATTEMPTS = 3  # stalls before giving the target up until the pose changes
RETARGET_RAD = math.radians(12)  # the target moved this far from the blocked one: a new pose, start afresh

FREE, YIELDING, GIVEN_UP = "free", "yielding", "given_up"


class AntennaStallGuard:
    """Stall detection and yield / retry policy for one antenna. Not thread-safe; owned by the movement loop."""

    def __init__(self, name: str) -> None:
        """Guard the antenna called ``name`` (``left`` / ``right``), as it appears in the logs."""
        self.name = name
        self.state = FREE
        self.attempts = 0  # stalls against the current blocked target
        self.hold_position: float | None = None  # commanded while yielding / given up
        self.blocked_target: float | None = None
        self._retry_at: float | None = None
        self._stall_since: float | None = None
        self._best_error: float = math.inf
        self._still_ref: float = 0.0

    @property
    def blocked(self) -> bool:
        """Whether the antenna is being held back from its target right now."""
        return self.state != FREE

    def command(self, target: float, present: float, now: float) -> float:
        """Return the position to command this tick, given the wanted ``target`` and the reported ``present``."""
        if self.state != FREE:
            assert self.blocked_target is not None and self.hold_position is not None
            if abs(target - self.blocked_target) > RETARGET_RAD:
                logger.info("Antenna %s released: the pose changed", self.name)
                self._release()
            elif self.state == GIVEN_UP or (self._retry_at is not None and now < self._retry_at):
                return self.hold_position
            else:
                logger.info(
                    "Antenna %s retrying its target (attempt %d/%d)", self.name, self.attempts + 1, MAX_ATTEMPTS
                )
                self.state = FREE
                self._reset_window()
        error = target - present
        if abs(error) <= STALL_ERROR_RAD:
            if self.attempts:
                logger.info("Antenna %s back on target after %d stall(s)", self.name, self.attempts)
                self.attempts = 0
            self._reset_window()
            return target
        if self._stall_since is None or abs(error) < self._best_error - STALL_PROGRESS_RAD:
            self._start_window(now, abs(error), present)  # first tick off target, or still closing in
        elif abs(present - self._still_ref) > STALL_STILL_RAD:
            self._start_window(now, abs(error), present)  # the antenna is moving, even if not towards the target
        elif now - self._stall_since >= STALL_S:
            return self._yield(target, present, error, now)
        return target

    def _start_window(self, now: float, error: float, present: float) -> None:
        self._stall_since, self._best_error, self._still_ref = now, error, present

    def _reset_window(self) -> None:
        self._stall_since, self._best_error = None, math.inf

    def _release(self) -> None:
        self.state, self.attempts = FREE, 0
        self.hold_position = self.blocked_target = self._retry_at = None
        self._reset_window()

    def _yield(self, target: float, present: float, error: float, now: float) -> float:
        self.attempts += 1
        self.blocked_target = target
        self.hold_position = present - math.copysign(BACKOFF_RAD, error)
        self._reset_window()
        where = f"Antenna {self.name} blocked at {present:.2f} rad on the way to {target:.2f} rad"
        if self.attempts >= MAX_ATTEMPTS:
            self.state, self._retry_at = GIVEN_UP, None
            logger.warning("%s; giving up until the pose changes (attempt %d/%d)", where, self.attempts, MAX_ATTEMPTS)
        else:
            wait = min(YIELD_S * 2 ** (self.attempts - 1), YIELD_MAX_S)
            self.state, self._retry_at = YIELDING, now + wait
            logger.warning("%s; yielding for %.0fs (attempt %d/%d)", where, wait, self.attempts, MAX_ATTEMPTS)
        return self.hold_position
