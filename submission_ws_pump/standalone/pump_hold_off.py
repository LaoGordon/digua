#!/usr/bin/env python3
"""Keep the pump OFF by holding GPIO low on RDK S100P.

Hardware:
- BOARD pin 37 -> MOS IO (40PIN_GPIO5_3V3)
- BOARD pin 39 -> GND
- MOS is active-high: HIGH = pump ON, LOW = pump OFF

This script keeps the GPIO driven LOW and never calls GPIO.cleanup(),
so the pin will not fall back to input mode on exit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

try:
    import Hobot.GPIO as GPIO
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] failed to import Hobot.GPIO: {exc}", file=sys.stderr)
    sys.exit(1)

DEFAULT_PIN = 37
_running = True
_pin = DEFAULT_PIN


def set_low(pin: int) -> None:
    """Drive the MOS control pin low."""
    GPIO.output(pin, GPIO.LOW)


def handle_exit_signal(signum, frame) -> None:  # noqa: D401, ANN001
    """Signal handler that forces the pump OFF before exit."""
    global _running
    print(f"\n[INFO] received signal {signum}, forcing GPIO { _pin } LOW and exiting...")
    try:
        set_low(_pin)
        print(f"[INFO] GPIO {_pin} set LOW, pump should be OFF")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to drive GPIO low on exit: {exc}", file=sys.stderr)
    _running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hold pump OFF by keeping GPIO LOW")
    parser.add_argument("--pin", type=int, default=DEFAULT_PIN, help="BOARD physical pin number, default: 37")
    return parser.parse_args()


def main() -> int:
    global _running, _pin
    args = parse_args()
    _pin = int(args.pin)

    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(_pin, GPIO.LOW)

    print(f"[INFO] using BOARD pin {_pin}")
    print(f"[INFO] GPIO {_pin} initialized LOW")
    print("[INFO] pump should be OFF, holding LOW every 1 second")

    try:
        while _running:
            set_low(_pin)
            print(f"[INFO] GPIO {_pin} held LOW")
            time.sleep(1.0)
    except KeyboardInterrupt:
        handle_exit_signal(signal.SIGINT, None)
    finally:
        # Intentionally do not call GPIO.cleanup().
        print("[INFO] exit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
