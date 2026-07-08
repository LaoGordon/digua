
#!/usr/bin/env python3
"""One-shot pump spray node for RDK S100P.

On startup, drive BOARD pin 37 HIGH for spray_duration_sec, then LOW.
The pin is intentionally left driven LOW on exit so the pump stays off.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from contextlib import suppress

import rclpy
from rclpy.node import Node

try:
    import Hobot.GPIO as GPIO
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] failed to import Hobot.GPIO: {exc}", file=sys.stderr)
    sys.exit(1)

DEFAULT_PIN = 37
DEFAULT_SPRAY_DURATION_SEC = 5.0


class PumpSprayNode(Node):
    def __init__(self) -> None:
        super().__init__("pump_spray_node")
        self.declare_parameter("pin", DEFAULT_PIN)
        self.declare_parameter("spray_duration_sec", DEFAULT_SPRAY_DURATION_SEC)

        self.pin = int(self.get_parameter("pin").value)
        self.spray_duration_sec = float(self.get_parameter("spray_duration_sec").value)
        self._running = True
        self._holdoff_was_active = False

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._set_low()
        self.get_logger().info(
            f"pump_spray_node ready: BOARD pin {self.pin}, spray {self.spray_duration_sec:.1f}s"
        )

    def _set_low(self) -> None:
        with suppress(Exception):
            GPIO.output(self.pin, GPIO.LOW)

    def _set_high(self) -> None:
        with suppress(Exception):
            GPIO.output(self.pin, GPIO.HIGH)

    def _stop_holdoff_service(self) -> None:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "pump_hold_off.service"],
            check=False,
        )
        self._holdoff_was_active = result.returncode == 0
        if self._holdoff_was_active:
            self.get_logger().info("stopping pump_hold_off.service for spray window")
            subprocess.run(["systemctl", "--user", "stop", "pump_hold_off.service"], check=False)

    def _restore_holdoff_service(self) -> None:
        if self._holdoff_was_active:
            self.get_logger().info("restarting pump_hold_off.service after spray")
            subprocess.run(["systemctl", "--user", "start", "pump_hold_off.service"], check=False)
            self._holdoff_was_active = False

    def spray_once(self) -> None:
        self._stop_holdoff_service()
        start = time.monotonic()
        self.get_logger().info(
            f"spray start: BOARD pin {self.pin} HIGH for {self.spray_duration_sec:.1f}s"
        )
        self._set_high()
        deadline = start + self.spray_duration_sec
        while self._running and rclpy.ok() and time.monotonic() < deadline:
            time.sleep(0.02)
        elapsed = time.monotonic() - start
        self.get_logger().info(f"spray end after {elapsed:.3f}s: forcing LOW")
        self._set_low()
        self._restore_holdoff_service()

    def shutdown(self) -> None:
        self._running = False
        self.get_logger().info("shutdown: forcing pin LOW")
        self._set_low()
        self._restore_holdoff_service()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = PumpSprayNode()

    def _on_signal(signum, _frame):  # noqa: ANN001
        node.get_logger().info(f"received signal {signum}, forcing OFF then exiting")
        node.shutdown()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        node.spray_once()
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
