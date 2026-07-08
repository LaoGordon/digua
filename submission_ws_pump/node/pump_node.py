#!/usr/bin/env python3
"""Water pump control node for RDK S100P.

Hardware:
- BOARD pin 37 -> MOS IO (40PIN_GPIO5_3V3)
- BOARD pin 39 -> GND
- MOS is active-high: HIGH = pump ON, LOW = pump OFF

This node is a long-running daemon that:
  * holds BOARD pin 37 LOW at startup (pump OFF) and for the whole lifetime,
  * subscribes to /pump/cmd (std_msgs/Bool): True  -> spray (HIGH),
                                               False -> stop immediately (LOW),
  * auto-stops after spray_duration_sec via a watchdog timer (never stuck ON),
  * never calls GPIO.cleanup() so the pin does not float back to input mode,
  * forces the pin LOW on SIGINT/SIGTERM before exit.

Pin stays owned by this single process; nothing else should touch pin 37.
"""

from __future__ import annotations

import signal
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import Hobot.GPIO as GPIO
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] failed to import Hobot.GPIO: {exc}", file=sys.stderr)
    sys.exit(1)

DEFAULT_PIN = 37
DEFAULT_SPRAY_DURATION_SEC = 5.0


class PumpNode(Node):
    def __init__(self) -> None:
        super().__init__("pump_node")

        self.declare_parameter("pin", DEFAULT_PIN)
        self.declare_parameter("spray_duration_sec", DEFAULT_SPRAY_DURATION_SEC)

        self.pin: int = int(self.get_parameter("pin").value)
        self.spray_duration_sec: float = float(self.get_parameter("spray_duration_sec").value)

        # One-shot watchdog timer that forces the pump OFF after spray_duration_sec.
        # Created cancelled; (re)armed on each True command.
        self._watchdog = self.create_timer(self.spray_duration_sec, self._watchdog_cb)
        self._watchdog.cancel()

        self.create_subscription(Bool, "/pump/cmd", self._cmd_cb, 10)

        # Hold the pin LOW for the whole node lifetime.
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._set_low()

        self.get_logger().info(
            f"pump_node ready: BOARD pin {self.pin} held LOW, "
            f"listen /pump/cmd (Bool), spray {self.spray_duration_sec:.1f}s on True"
        )

    # ---- GPIO helpers -------------------------------------------------
    def _set_low(self) -> None:
        with _swallow("drive pin LOW"):
            GPIO.output(self.pin, GPIO.LOW)

    def _set_high(self) -> None:
        with _swallow("drive pin HIGH"):
            GPIO.output(self.pin, GPIO.HIGH)

    # ---- ROS callbacks ------------------------------------------------
    def _cmd_cb(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info(f"SPRAY ON for {self.spray_duration_sec:.1f}s")
            self._set_high()
            # (Re)arm the watchdog: cancel any pending one, then reset.
            self._watchdog.cancel()
            self._watchdog.reset()
        else:
            self.get_logger().info("SPRAY OFF (cmd=False)")
            self._watchdog.cancel()
            self._set_low()

    def _watchdog_cb(self) -> None:
        self.get_logger().info("watchdog: spray duration elapsed, forcing OFF")
        self._set_low()
        self._watchdog.cancel()

    # ---- shutdown -----------------------------------------------------
    def shutdown(self) -> None:
        self.get_logger().info("shutdown: forcing pin LOW")
        self._set_low()
        # Intentionally do NOT call GPIO.cleanup(): keeping the pin driven LOW
        # prevents it from floating back to input mode after exit.


class _swallow:
    """Context manager that logs GPIO errors instead of raising."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            print(f"[WARN] failed to {self.what}: {exc}", file=sys.stderr)
        return True  # suppress


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = PumpNode()

    def _on_signal(signum, _frame):  # noqa: ANN001
        node.get_logger().info(f"received signal {signum}, forcing OFF then exiting")
        node.shutdown()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
