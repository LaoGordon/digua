#!/usr/bin/env python3
"""Simple pump control for RDK S100P using Hobot.GPIO.

Hardware:
- BOARD pin 37 -> MOS IO (40PIN_GPIO5_3V3)
- BOARD pin 39 -> GND
- MOS is active-high: HIGH = pump ON, LOW = pump OFF
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from contextlib import suppress
from typing import Optional

try:
    import Hobot.GPIO as GPIO
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] failed to import Hobot.GPIO: {exc}", file=sys.stderr)
    sys.exit(1)

DEFAULT_PIN = 37
_running = True
_pin = DEFAULT_PIN


def setup_gpio(pin: int) -> None:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


def pump_on(pin: int) -> None:
    GPIO.output(pin, GPIO.HIGH)


def pump_off(pin: int) -> None:
    GPIO.output(pin, GPIO.LOW)


def cleanup(pin: Optional[int]) -> None:
    if pin is not None:
        with suppress(Exception):
            GPIO.output(pin, GPIO.LOW)
    with suppress(Exception):
        GPIO.cleanup()


def handle_exit(signum, frame) -> None:  # noqa: D401, ANN001
    global _running
    print(f"\n[INFO] received signal {signum}, turning pump OFF")
    try:
        pump_off(_pin)
        print(f"[INFO] GPIO {_pin} set LOW")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to force GPIO low: {exc}", file=sys.stderr)
    _running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDK S100P pump control")
    parser.add_argument("command", choices=["on", "off", "run"], help="pump command")
    parser.add_argument("seconds", nargs="?", type=float, help="duration for 'run'")
    parser.add_argument("--pin", type=int, default=DEFAULT_PIN, help="BOARD physical pin number, default: 37")
    return parser.parse_args()


def main() -> int:
    global _running, _pin
    args = parse_args()
    _pin = int(args.pin)
    _running = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    setup_gpio(_pin)

    try:
        if args.command == "on":
            pump_on(_pin)
            print(f"[INFO] pump ON on BOARD pin {_pin}")
            return 0

        if args.command == "off":
            pump_off(_pin)
            print(f"[INFO] pump OFF on BOARD pin {_pin}")
            return 0

        if args.seconds is None or args.seconds <= 0:
            print("[ERROR] run requires a positive duration, e.g. run 5", file=sys.stderr)
            return 2

        print(f"[INFO] pump ON for {args.seconds:.2f}s on BOARD pin {_pin}")
        pump_on(_pin)
        end_time = time.monotonic() + float(args.seconds)
        while _running and time.monotonic() < end_time:
            time.sleep(0.1)
        pump_off(_pin)
        print("[INFO] pump OFF")
        return 0

    except KeyboardInterrupt:
        handle_exit(signal.SIGINT, None)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] pump control failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cleanup(_pin)


if __name__ == "__main__":
    raise SystemExit(main())
