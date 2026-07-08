#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Continuously send a fixed Lite3 pitch axis value for observation."
    )
    parser.add_argument("--axis", type=int, default=5000, help="Fixed pitch axis value to send.")
    parser.add_argument("--duration", type=float, default=10.0, help="How long to send the command.")
    parser.add_argument("--rate", type=float, default=20.0, help="Send rate in Hz.")
    parser.add_argument("--dog-ip", default="192.168.137.120", help="Lite3 controller IP.")
    parser.add_argument("--dog-port", type=int, default=43893, help="Lite3 controller UDP port.")
    parser.add_argument("--heartbeat-hz", type=float, default=10.0, help="Heartbeat rate in Hz.")
    parser.add_argument("--send-zero-after", action="store_true", help="Send pitch 0 twice before exit.")
    parser.add_argument("--navi-mode", action="store_true", help="Switch to NAVI_MODE before the test.")
    parser.add_argument("--pose-mode", action="store_true", help="Switch to POSE_MODE before the test.")
    parser.add_argument(
        "--stand",
        action="store_true",
        help="Send STAND_SIT before the test. Use only when the dog is currently sitting.",
    )
    parser.add_argument("--stand-wait", type=float, default=8.0, help="Wait time after --stand.")
    parser.add_argument("--mode-wait", type=float, default=0.5, help="Wait after mode switch commands.")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    pkg_root = Path(__file__).resolve().parent / "src" / "fire_pitch_aim_ros2"
    sys.path.insert(0, str(pkg_root))

    from fire_pitch_aim_ros2.lite3_udp import HeartbeatThread, Lite3UDP

    if args.rate <= 0.0:
        raise SystemExit("--rate must be > 0")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be > 0")

    dog = Lite3UDP(ip=args.dog_ip, port=args.dog_port)
    heartbeat = HeartbeatThread(dog, hz=args.heartbeat_hz)
    heartbeat.start()

    try:
        time.sleep(0.2)
        if args.navi_mode:
            print("[INFO] sending NAVI_MODE")
            dog.switch_navi_mode()
            time.sleep(args.mode_wait)
        if args.stand:
            print("[INFO] sending STAND_SIT (toggle)")
            dog.stand_sit()
            time.sleep(args.stand_wait)
        if args.pose_mode:
            print("[INFO] sending POSE_MODE")
            dog.switch_pose_mode()
            time.sleep(args.mode_wait)

        interval = 1.0 / args.rate
        end_time = time.monotonic() + args.duration
        count = 0
        print(
            f"[INFO] sending fixed pitch axis={args.axis} for {args.duration:.2f}s "
            f"at {args.rate:.1f}Hz to {args.dog_ip}:{args.dog_port}"
        )
        while time.monotonic() < end_time:
            dog.send_pitch_axis(args.axis)
            count += 1
            time.sleep(interval)
        print(f"[INFO] sent {count} pitch packets")

        if args.send_zero_after:
            print("[INFO] sending pitch axis 0 twice")
            dog.send_pitch_axis(0)
            time.sleep(0.05)
            dog.send_pitch_axis(0)
    finally:
        heartbeat.stop()
        dog.close()


if __name__ == "__main__":
    main()
