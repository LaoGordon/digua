#!/usr/bin/env python3
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import cv2
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from .lite3_udp import DOG_IP, DOG_PORT, Lite3UDP


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Patrol2Node(Node):
    def __init__(self):
        super().__init__("patrol2_node")
        self._declare_parameters()

        self.waypoints_file = Path(str(self.get_parameter("waypoints_file").value)).expanduser()
        self.photo_dir = Path(str(self.get_parameter("photo_dir").value)).expanduser()
        self.record_file = Path(str(self.get_parameter("record_file").value)).expanduser()
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.nav_action_name = str(self.get_parameter("nav_action_name").value)
        self.nav_goal_timeout_sec = float(self.get_parameter("nav_goal_timeout_sec").value)
        self.nav_server_timeout_sec = float(self.get_parameter("nav_server_timeout_sec").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.spray_delay_sec = float(self.get_parameter("spray_delay_sec").value)
        self.aim_stop_settle_sec = float(self.get_parameter("aim_stop_settle_sec").value)
        self.inter_waypoint_pause_sec = float(self.get_parameter("inter_waypoint_pause_sec").value)
        self.return_to_start = bool(self.get_parameter("return_to_start").value)
        self.pump_workspace = Path(str(self.get_parameter("pump_workspace").value)).expanduser()
        self.pump_launch_timeout_sec = float(self.get_parameter("pump_launch_timeout_sec").value)
        self.dog_ip = str(self.get_parameter("dog_ip").value or DOG_IP)
        self.dog_port = int(self.get_parameter("dog_port").value or DOG_PORT)

        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.record_file.parent.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)
        self.latest_bgr = None
        self.latest_image_stamp = None
        self.latest_detection_stamp = None
        self._pump_proc = None
        self._dog = Lite3UDP(ip=self.dog_ip, port=self.dog_port)

        self.create_subscription(Image, self.color_topic, self._image_cb, 10)
        self.create_subscription(Detection2DArray, self.detections_topic, self._detections_cb, 10)

    def _declare_parameters(self):
        self.declare_parameter(
            "waypoints_file",
            "/home/sunrise/ws_yolo_patrol/src/yolo_patrol_ros2/config/patrol_waypoints.yaml",
        )
        self.declare_parameter("photo_dir", "/home/sunrise/Desktop/patrol_shot")
        self.declare_parameter("record_file", "/home/sunrise/ws_patrol2/patrol2_records.jsonl")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("detections_topic", "/fire/detections")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("nav_action_name", "/navigate_to_pose")
        self.declare_parameter("nav_goal_timeout_sec", 180.0)
        self.declare_parameter("nav_server_timeout_sec", 20.0)
        self.declare_parameter("tf_timeout_sec", 5.0)
        self.declare_parameter("spray_delay_sec", 2.0)
        self.declare_parameter("aim_stop_settle_sec", 0.8)
        self.declare_parameter("inter_waypoint_pause_sec", 1.0)
        self.declare_parameter("return_to_start", True)
        self.declare_parameter("pump_workspace", "/home/sunrise/ws_pump")
        self.declare_parameter("pump_launch_timeout_sec", 20.0)
        self.declare_parameter("dog_ip", DOG_IP)
        self.declare_parameter("dog_port", DOG_PORT)

    def _image_cb(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_stamp = msg.header.stamp
        except Exception as exc:
            self.get_logger().warn(f"image convert failed: {exc}")

    def _detections_cb(self, msg: Detection2DArray):
        if msg.detections:
            self.latest_detection_stamp = time.time()

    def _spin_until(self, predicate, timeout_sec, sleep_sec=0.1):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=sleep_sec)
            if predicate():
                return True
        return predicate()

    def _spin_for(self, duration_sec, sleep_sec=0.1):
        deadline = time.time() + duration_sec
        while rclpy.ok() and time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            rclpy.spin_once(self, timeout_sec=min(sleep_sec, remaining))

    def load_waypoints(self):
        if not self.waypoints_file.exists():
            raise FileNotFoundError(f"waypoints file not found: {self.waypoints_file}")
        data = yaml.safe_load(self.waypoints_file.read_text(encoding="utf-8")) or {}
        waypoints = data.get("waypoints") or []
        if not waypoints:
            raise RuntimeError(f"no waypoints in: {self.waypoints_file}")
        return waypoints

    def wait_for_nav_server(self):
        self.get_logger().info(f"waiting for nav action server {self.nav_action_name}")
        last_log = 0.0
        deadline = None if self.nav_server_timeout_sec <= 0 else (time.time() + self.nav_server_timeout_sec)
        while rclpy.ok():
            if deadline is not None and time.time() > deadline:
                return False
            try:
                if self.nav_client.wait_for_server(timeout_sec=2.0):
                    return True
            except Exception as exc:
                now = time.time()
                if now - last_log >= 5.0:
                    self.get_logger().warn(f"waiting for nav server failed temporarily: {exc}")
                    last_log = now
                time.sleep(1.0)
                continue
            now = time.time()
            if now - last_log >= 5.0:
                self.get_logger().info(f"still waiting for nav action server {self.nav_action_name}")
                last_log = now
        return False

    def current_map_pose(self):
        deadline = time.time() + self.tf_timeout_sec
        while rclpy.ok() and time.time() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    Time(),
                )
                pose = PoseStamped()
                pose.header.frame_id = self.map_frame
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = transform.transform.translation.x
                pose.pose.position.y = transform.transform.translation.y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                return pose
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(f"failed to get transform {self.map_frame}->{self.base_frame}")

    def make_goal_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        qx, qy, qz, qw = quaternion_from_yaw(float(yaw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def navigate_to(self, pose: PoseStamped, label: str):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(
            f"navigating to {label}: x={pose.pose.position.x:.3f} y={pose.pose.position.y:.3f}"
        )
        send_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None:
            raise RuntimeError(f"navigate_to_pose send goal failed: {label}")
        if not goal_handle.accepted:
            raise RuntimeError(f"navigate_to_pose goal rejected: {label}")

        result_future = goal_handle.get_result_async()
        deadline = time.time() + self.nav_goal_timeout_sec
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
            if result_future.done():
                break
            if time.time() > deadline:
                self.get_logger().warn(f"navigation timeout for {label}, canceling goal")
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
                raise RuntimeError(f"navigation timeout: {label}")

        result = result_future.result()
        if result is None:
            raise RuntimeError(f"navigation returned no result: {label}")
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"navigation failed for {label}, status={result.status}")
        self.get_logger().info(f"reached {label}")

    def _pump_command(self):
        return (
            f"source /opt/ros/humble/setup.bash && "
            f"source {self.pump_workspace}/install/setup.bash && "
            f"ros2 launch pump_ros2 pump_spray.launch.py"
        )

    def start_pump_spray(self):
        if self._pump_proc is not None and self._pump_proc.poll() is None:
            return
        cmd = self._pump_command()
        self.get_logger().info("starting pump spray launcher")
        self._pump_proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            preexec_fn=os.setsid,
        )

    def wait_for_pump_spray(self):
        proc = self._pump_proc
        if proc is None:
            raise RuntimeError("pump spray was not started")
        deadline = time.time() + self.pump_launch_timeout_sec
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if proc.poll() is not None:
                break
            if time.time() > deadline:
                self.get_logger().warn("pump spray launcher timeout, terminating")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                self._pump_proc = None
                raise RuntimeError("pump spray launcher timeout")
        rc = proc.poll()
        self._pump_proc = None
        if rc not in (0, None):
            raise RuntimeError(f"pump spray launcher failed with code {rc}")

    def capture_photo(self, waypoint_name):
        if self.latest_bgr is None:
            raise RuntimeError("no color image received yet, cannot capture photo")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = waypoint_name.replace("/", "_")
        output_path = self.photo_dir / f"{timestamp}_{safe_name}.jpg"
        ok = cv2.imwrite(str(output_path), self.latest_bgr)
        if not ok:
            raise RuntimeError(f"failed to write photo: {output_path}")
        return output_path

    def append_record(self, payload):
        with self.record_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run_patrol(self):
        waypoints = self.load_waypoints()
        if not self.wait_for_nav_server():
            raise RuntimeError(f"nav action server not ready: {self.nav_action_name}")

        start_pose = self.current_map_pose()
        start_yaw = yaw_from_quaternion(
            start_pose.pose.orientation.x,
            start_pose.pose.orientation.y,
            start_pose.pose.orientation.z,
            start_pose.pose.orientation.w,
        )
        self.get_logger().info(
            f"saved start pose x={start_pose.pose.position.x:.3f} "
            f"y={start_pose.pose.position.y:.3f} yaw={start_yaw:.3f}"
        )

        for index, waypoint in enumerate(waypoints, start=1):
            name = str(waypoint.get("name", f"observe_{index}"))
            goal_pose = self.make_goal_pose(waypoint["x"], waypoint["y"], waypoint.get("yaw", 0.0))
            self.navigate_to(goal_pose, name)
            time.sleep(self.inter_waypoint_pause_sec)

            self.get_logger().info(f"arrived at {name}, waiting {self.spray_delay_sec:.1f}s before spray")
            self._spin_for(self.spray_delay_sec)
            self.start_pump_spray()
            self.wait_for_pump_spray()
            time.sleep(self.aim_stop_settle_sec)

            photo_path = self.capture_photo(name)
            self.append_record(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "waypoint_index": index,
                    "waypoint_name": name,
                    "goal": {
                        "x": float(waypoint["x"]),
                        "y": float(waypoint["y"]),
                        "yaw": float(waypoint.get("yaw", 0.0)),
                    },
                    "photo_path": str(photo_path),
                    "fire_suppression_sequence": {
                        "spray_delay_sec": float(self.spray_delay_sec),
                        "pump_spray_sec": 5.0,
                        "returned_to_nav_mode": True,
                    },
                }
            )
            self.get_logger().info(f"saved photo for {name}: {photo_path}")
            time.sleep(self.inter_waypoint_pause_sec)

        if self.return_to_start:
            self.navigate_to(start_pose, "start_pose")
        self.get_logger().info("patrol2 finished")

    def cleanup(self):
        proc = self._pump_proc
        try:
            self._pump_proc = None
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=3.0)
        except Exception:
            try:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        self._dog.close()


def main(args=None):
    rclpy.init(args=args)
    node = Patrol2Node()
    exit_code = 0
    try:
        node.run_patrol()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("patrol2 interrupted")
    except Exception as exc:
        exit_code = 1
        node.get_logger().error(f"patrol2 failed: {exc}")
        raise
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code:
        raise SystemExit(exit_code)
