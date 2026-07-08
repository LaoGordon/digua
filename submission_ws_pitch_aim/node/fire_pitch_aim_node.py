#!/usr/bin/env python3
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

from .lite3_udp import DOG_IP, DOG_PORT, HeartbeatThread, Lite3UDP


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class FirePitchAimNode(Node):
    def __init__(self):
        super().__init__("fire_pitch_aim_node")
        self._declare_parameters()

        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.fire_class_name = str(self.get_parameter("fire_class_name").value).strip().lower()
        self.min_score = float(self.get_parameter("min_score").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.target_y_ratio = float(self.get_parameter("target_y_ratio").value)
        self.deadband_px = float(self.get_parameter("deadband_px").value)
        self.release_deadband_px = float(self.get_parameter("release_deadband_px").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.up_kp_axis_per_px = float(self.get_parameter("up_kp_axis_per_px").value)
        self.down_kp_axis_per_px = float(self.get_parameter("down_kp_axis_per_px").value)
        self.up_ki_axis_per_px_sec = float(self.get_parameter("up_ki_axis_per_px_sec").value)
        self.down_ki_axis_per_px_sec = float(self.get_parameter("down_ki_axis_per_px_sec").value)
        self.integral_limit_px_sec = float(self.get_parameter("integral_limit_px_sec").value)
        self.up_bias_axis = int(self.get_parameter("up_bias_axis").value)
        self.down_bias_axis = int(self.get_parameter("down_bias_axis").value)
        self.up_axis_limit = int(self.get_parameter("up_axis_limit").value)
        self.down_axis_limit = int(self.get_parameter("down_axis_limit").value)
        self.min_axis = int(self.get_parameter("min_axis").value)
        self.max_axis = int(self.get_parameter("max_axis").value)
        self.target_y_filter_alpha = float(self.get_parameter("target_y_filter_alpha").value)
        self.invert_axis = bool(self.get_parameter("invert_axis").value)
        self.send_zero_on_target_loss = bool(self.get_parameter("send_zero_on_target_loss").value)
        self.log_throttle_sec = float(self.get_parameter("log_throttle_sec").value)
        self.dog_ip = str(self.get_parameter("dog_ip").value or DOG_IP)
        self.dog_port = int(self.get_parameter("dog_port").value or DOG_PORT)
        self.auto_stand = bool(self.get_parameter("auto_stand").value)
        self.auto_navi_mode = bool(self.get_parameter("auto_navi_mode").value)
        self.auto_pose_mode = bool(self.get_parameter("auto_pose_mode").value)
        self.stand_wait_sec = float(self.get_parameter("stand_wait_sec").value)
        self.navi_wait_sec = float(self.get_parameter("navi_wait_sec").value)
        self.pose_wait_sec = float(self.get_parameter("pose_wait_sec").value)
        self.heartbeat_hz = float(self.get_parameter("heartbeat_hz").value)

        self.target_center_y = self.image_height * self.target_y_ratio
        self.latest_target_y = None
        self.filtered_target_y = None
        self.latest_target_score = None
        self.latest_target_stamp = 0.0
        self.last_sent_axis = None
        self.last_log_time = 0.0
        self.last_control_time = time.monotonic()
        self.hold_center = False
        self.integral_error = 0.0

        self.dog = Lite3UDP(ip=self.dog_ip, port=self.dog_port)
        self.heartbeat = HeartbeatThread(self.dog, hz=self.heartbeat_hz)
        self.heartbeat.start()
        time.sleep(0.2)

        if self.auto_navi_mode:
            self.get_logger().info("sending NAVI_MODE")
            self.dog.switch_navi_mode()
            time.sleep(self.navi_wait_sec)
        if self.auto_stand:
            self.get_logger().info("sending STAND_SIT (toggle) and waiting for stand")
            self.dog.stand_sit()
            time.sleep(self.stand_wait_sec)
        if self.auto_pose_mode:
            self.get_logger().info("sending POSE_MODE")
            self.dog.switch_pose_mode()
            time.sleep(self.pose_wait_sec)

        self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self._detections_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / self.control_hz, self._control_loop)
        self.get_logger().info(
            f"started fire_pitch_aim_node detections_topic={self.detections_topic} "
            f"target_center_y={self.target_center_y:.1f} up_kp_axis_per_px={self.up_kp_axis_per_px:.1f} "
            f"down_kp_axis_per_px={self.down_kp_axis_per_px:.1f}"
        )

    def _declare_parameters(self):
        self.declare_parameter("detections_topic", "/fire/detections")
        self.declare_parameter("fire_class_name", "fire")
        self.declare_parameter("min_score", 0.25)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("target_y_ratio", 0.5)
        self.declare_parameter("deadband_px", 18.0)
        self.declare_parameter("release_deadband_px", 36.0)
        self.declare_parameter("target_timeout_sec", 0.4)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("up_kp_axis_per_px", 80.0)
        self.declare_parameter("down_kp_axis_per_px", 100.0)
        self.declare_parameter("up_ki_axis_per_px_sec", 0.0)
        self.declare_parameter("down_ki_axis_per_px_sec", 0.0)
        self.declare_parameter("integral_limit_px_sec", 200.0)
        self.declare_parameter("up_bias_axis", 0)
        self.declare_parameter("down_bias_axis", 0)
        self.declare_parameter("up_axis_limit", 12000)
        self.declare_parameter("down_axis_limit", 15000)
        self.declare_parameter("min_axis", 5000)
        self.declare_parameter("max_axis", 60000)
        self.declare_parameter("target_y_filter_alpha", 0.45)
        self.declare_parameter("invert_axis", False)
        self.declare_parameter("send_zero_on_target_loss", True)
        self.declare_parameter("log_throttle_sec", 1.0)
        self.declare_parameter("dog_ip", DOG_IP)
        self.declare_parameter("dog_port", DOG_PORT)
        self.declare_parameter("auto_stand", True)
        self.declare_parameter("auto_navi_mode", True)
        self.declare_parameter("auto_pose_mode", True)
        self.declare_parameter("stand_wait_sec", 8.0)
        self.declare_parameter("navi_wait_sec", 0.5)
        self.declare_parameter("pose_wait_sec", 0.5)
        self.declare_parameter("heartbeat_hz", 10.0)

    def _detections_callback(self, msg: Detection2DArray):
        best_y = None
        best_score = -1.0
        for detection in msg.detections:
            det_score = -1.0
            det_is_fire = False
            for result in detection.results:
                class_id = str(result.hypothesis.class_id).strip().lower()
                score = float(result.hypothesis.score)
                if class_id == self.fire_class_name and score >= self.min_score and score > det_score:
                    det_score = score
                    det_is_fire = True
            if det_is_fire and det_score > best_score:
                best_score = det_score
                best_y = float(detection.bbox.center.position.y)

        if best_y is not None:
            if self.filtered_target_y is None:
                self.filtered_target_y = best_y
            else:
                alpha = clamp(self.target_y_filter_alpha, 0.0, 1.0)
                self.filtered_target_y = alpha * best_y + (1.0 - alpha) * self.filtered_target_y
            self.latest_target_y = self.filtered_target_y
            self.latest_target_score = best_score
            self.latest_target_stamp = time.monotonic()

    def _compute_axis(self, target_y: float):
        error_px = target_y - self.target_center_y
        error_abs = abs(error_px)
        error_sign = 1 if error_px > 0 else -1
        now = time.monotonic()
        if self.hold_center:
            if error_abs <= self.release_deadband_px:
                self.last_control_time = now
                self.integral_error *= 0.5
                return 0, error_px
            self.hold_center = False

        if error_abs < self.deadband_px:
            self.hold_center = True
            self.last_control_time = now
            self.integral_error *= 0.5
            return 0, error_px

        dt = max(1e-3, now - self.last_control_time)
        self.last_control_time = now
        self.integral_error += error_px * dt
        self.integral_error = clamp(
            self.integral_error,
            -self.integral_limit_px_sec,
            self.integral_limit_px_sec,
        )

        if error_sign > 0:
            axis_f = (
                self.down_bias_axis
                + self.down_kp_axis_per_px * error_px
                + self.down_ki_axis_per_px_sec * self.integral_error
            )
            axis_limit = min(self.down_axis_limit, self.max_axis)
        else:
            axis_f = (
                -self.up_bias_axis
                + self.up_kp_axis_per_px * error_px
                + self.up_ki_axis_per_px_sec * self.integral_error
            )
            axis_limit = min(self.up_axis_limit, self.max_axis)

        axis = int(round(axis_f))
        axis = int(clamp(axis, -axis_limit, axis_limit))
        if axis != 0 and abs(axis) < self.min_axis:
            axis = self.min_axis if axis > 0 else -self.min_axis
        if self.invert_axis:
            axis = -axis
        return axis, error_px

    def _log_throttled(self, text: str):
        now = time.monotonic()
        if now - self.last_log_time >= self.log_throttle_sec:
            self.get_logger().info(text)
            self.last_log_time = now

    def _send_axis(self, axis: int):
        limited_axis = int(clamp(axis, -self.max_axis, self.max_axis))
        self.dog.send_pitch_axis(limited_axis)
        self.last_sent_axis = limited_axis

    def _control_loop(self):
        now = time.monotonic()
        target_valid = (
            self.latest_target_y is not None
            and (now - self.latest_target_stamp) <= self.target_timeout_sec
        )

        if not target_valid:
            self.last_control_time = now
            self.hold_center = False
            self.integral_error = 0.0
            if self.send_zero_on_target_loss:
                self._send_axis(0)
            self._log_throttled("target lost or stale, sending pitch axis 0")
            return

        axis, error_px = self._compute_axis(self.latest_target_y)
        self._send_axis(axis)
        self._log_throttled(
            f"target_y={self.latest_target_y:.1f} target_center_y={self.target_center_y:.1f} "
            f"error_px={error_px:.1f} score={self.latest_target_score:.2f} "
            f"int_err={self.integral_error:.1f} axis_req={axis} axis_sent={self.last_sent_axis}"
        )

    def destroy_node(self):
        try:
            self.dog.send_pitch_axis(0)
            time.sleep(0.05)
            self.dog.send_pitch_axis(0)
        except Exception:
            pass
        self.heartbeat.stop()
        self.dog.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FirePitchAimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
