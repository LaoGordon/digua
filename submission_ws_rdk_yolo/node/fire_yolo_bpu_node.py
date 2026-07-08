#!/usr/bin/env python3
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped, PoseStamped
from hbm_runtime import HB_HBMRuntime
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    distance_m: float | None
    point_xyz: tuple[float, float, float] | None


def letterbox(image: np.ndarray, target_size: int = 640):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, left, top


class FireYoloBPU:
    def __init__(self, model_path: str, input_size: int = 640):
        self.runtime = HB_HBMRuntime(model_path)
        if not self.runtime.model_names:
            raise RuntimeError(f"failed to load model: {model_path}")
        self.model_name = self.runtime.model_names[0]
        input_names = self.runtime.input_names.get(self.model_name, [])
        output_names = self.runtime.output_names.get(self.model_name, [])
        if not input_names:
            raise RuntimeError(f"model has no input tensors: {model_path}")
        if not output_names:
            raise RuntimeError(f"model has no output tensors: {model_path}")
        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_size = input_size
        self.output_shape = tuple(self.runtime.output_shapes[self.model_name][self.output_name])

    def preprocess(self, frame_bgr: np.ndarray):
        padded, scale, pad_left, pad_top = letterbox(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        return chw, scale, pad_left, pad_top

    def infer(self, frame_bgr: np.ndarray, conf_thres: float, iou_thres: float):
        input_tensor, scale, pad_left, pad_top = self.preprocess(frame_bgr)
        outputs = self.runtime.run({self.input_name: input_tensor})
        model_outputs = outputs.get(self.model_name)
        if not model_outputs or self.output_name not in model_outputs:
            raise RuntimeError(f"missing output tensor '{self.output_name}' for model '{self.model_name}'")
        preds = np.asarray(model_outputs[self.output_name], dtype=np.float32)
        preds = preds.reshape(self.output_shape)
        preds = np.squeeze(preds, axis=0).T

        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        keep = scores >= conf_thres
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        if len(scores) == 0:
            return []

        nms_boxes = []
        decoded = []
        h, w = frame_bgr.shape[:2]
        for box, score in zip(boxes_xywh, scores):
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2 - pad_left) / scale
            y1 = (cy - bh / 2 - pad_top) / scale
            x2 = (cx + bw / 2 - pad_left) / scale
            y2 = (cy + bh / 2 - pad_top) / scale
            x1 = int(max(0, min(w - 1, round(x1))))
            y1 = int(max(0, min(h - 1, round(y1))))
            x2 = int(max(0, min(w - 1, round(x2))))
            y2 = int(max(0, min(h - 1, round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            decoded.append((x1, y1, x2, y2, float(score)))
            nms_boxes.append([x1, y1, x2 - x1, y2 - y1])

        if not decoded:
            return []

        indices = cv2.dnn.NMSBoxes(nms_boxes, [item[4] for item in decoded], conf_thres, iou_thres)
        if len(indices) == 0:
            return []

        result = []
        for idx in np.array(indices).reshape(-1):
            x1, y1, x2, y2, score = decoded[int(idx)]
            result.append(Detection(x1, y1, x2, y2, score, None, None))
        return result


class FireYoloBPUNode(Node):
    def __init__(self):
        super().__init__("fire_yolo_bpu_node")
        self._declare_parameters()

        self.model_path = str(self.get_parameter("model_path").value)
        self.color_width = int(self.get_parameter("color_width").value)
        self.color_height = int(self.get_parameter("color_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.depth_min_valid_m = float(self.get_parameter("depth_min_valid_m").value)
        self.depth_max_valid_m = float(self.get_parameter("depth_max_valid_m").value)
        self.display = bool(self.get_parameter("display").value)
        self.display_window_name = str(self.get_parameter("display_window_name").value)
        self.camera_mode = str(self.get_parameter("camera_mode").value).strip().lower()
        self.camera_detect_timeout_sec = float(self.get_parameter("camera_detect_timeout_sec").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.fire_class_name = str(self.get_parameter("fire_class_name").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.point_topic = str(self.get_parameter("point_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.detector = FireYoloBPU(self.model_path)
        self.detections_pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.point_pub = self.create_publisher(PointStamped, self.point_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_color = None
        self._latest_depth = None
        self._latest_ros_frame_id = None
        self._camera_info_ready = False
        self._intr_fx = None
        self._intr_fy = None
        self._intr_cx = None
        self._intr_cy = None
        self._ros_mode_active = False
        self._direct_mode_active = False
        self._direct_thread = None
        self._camera_retry_timer = None
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.color_intrinsics = None

        self._setup_camera_mode()
        self._process_timer = self.create_timer(1.0 / max(self.camera_fps, 1), self._process_once)
        self.get_logger().info(
            f"started fire_yolo_bpu_node model={self.model_path} detections_topic={self.detections_topic} "
            f"camera_mode={self.camera_mode} ros_active={self._ros_mode_active} direct_active={self._direct_mode_active}"
        )

    def _declare_parameters(self):
        self.declare_parameter("model_path", "/home/sunrise/ws_rdk_yolo/fire_yolov8n_s100p.hbm")
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("depth_min_valid_m", 0.10)
        self.declare_parameter("depth_max_valid_m", 5.00)
        self.declare_parameter("display", True)
        self.declare_parameter("display_window_name", "fire_yolo_bpu")
        self.declare_parameter("camera_mode", "auto")
        self.declare_parameter("camera_detect_timeout_sec", 2.0)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("fire_class_name", "fire")
        self.declare_parameter("detections_topic", "/fire/detections")
        self.declare_parameter("point_topic", "/fire/point")
        self.declare_parameter("pose_topic", "/fire/pose")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

    def _setup_camera_mode(self):
        if self.camera_mode not in ("auto", "ros", "direct"):
            raise ValueError(f"unsupported camera_mode: {self.camera_mode}")
        if self.camera_mode == "direct":
            self._start_direct_camera()
            return
        if self.camera_mode == "ros":
            if self._ros_topics_available():
                self._start_ros_camera()
            else:
                self.get_logger().info("ROS camera topics not ready yet, waiting for camera startup")
                self._camera_retry_timer = self.create_timer(1.0, self._retry_ros_camera_mode)
            return
        if self._ros_topics_available():
            self._start_ros_camera()
            return
        self.get_logger().warn("ROS camera topics not available yet, falling back to direct camera mode")
        self._start_direct_camera()

    def _ros_topics_available(self):
        deadline = time.monotonic() + self.camera_detect_timeout_sec
        required = {
            self.color_topic: "sensor_msgs/msg/Image",
            self.depth_topic: "sensor_msgs/msg/Image",
            self.camera_info_topic: "sensor_msgs/msg/CameraInfo",
        }
        while time.monotonic() < deadline:
            topics = dict(self.get_topic_names_and_types())
            ok = True
            for topic_name, type_name in required.items():
                types = topics.get(topic_name, [])
                if type_name not in types:
                    ok = False
                    break
            if ok:
                return True
            time.sleep(0.1)
        return False

    def _start_ros_camera(self):
        self._ros_mode_active = True
        self.create_subscription(Image, self.color_topic, self._color_callback, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 10)
        self.get_logger().info(
            f"using ROS camera topics color={self.color_topic} depth={self.depth_topic} info={self.camera_info_topic}"
        )

    def _start_direct_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.camera_fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.color_width, self.color_height, rs.format.z16, self.camera_fps
        )
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_profile.get_intrinsics()
        self.frame_id = self.frame_id or color_profile.stream_name()
        self._direct_mode_active = True
        self._direct_thread = threading.Thread(target=self._run_direct_loop, daemon=True)
        self._direct_thread.start()
        self.get_logger().info("using direct RealSense access")

    def _retry_ros_camera_mode(self):
        if self._ros_mode_active or self._direct_mode_active:
            if self._camera_retry_timer is not None:
                self._camera_retry_timer.cancel()
                self._camera_retry_timer = None
            return
        if self._ros_topics_available():
            self.get_logger().info("ROS camera topics appeared, switching to ROS camera mode")
            self._start_ros_camera()
            if self._camera_retry_timer is not None:
                self._camera_retry_timer.cancel()
                self._camera_retry_timer = None

    def _camera_info_callback(self, msg: CameraInfo):
        if len(msg.k) >= 9:
            self._intr_fx = float(msg.k[0])
            self._intr_fy = float(msg.k[4])
            self._intr_cx = float(msg.k[2])
            self._intr_cy = float(msg.k[5])
            self._camera_info_ready = True
            if not self.frame_id:
                self.frame_id = msg.header.frame_id

    def _color_callback(self, msg: Image):
        image = self._image_to_bgr(msg)
        if image is None:
            return
        with self._frame_lock:
            self._latest_color = image
            self._latest_ros_frame_id = msg.header.frame_id

    def _depth_callback(self, msg: Image):
        depth = self._image_to_depth_m(msg)
        if depth is None:
            return
        with self._frame_lock:
            self._latest_depth = depth

    def _image_to_bgr(self, msg: Image):
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return None
        channels = 3
        image = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        expected = msg.height * msg.step
        if image.size < expected:
            return None
        image = image[:expected].reshape(msg.height, msg.step)
        image = image[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if msg.encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding == "bgr8":
            return image.copy()
        return None

    def _image_to_depth_m(self, msg: Image):
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return None
        if msg.encoding == "16UC1":
            image = np.frombuffer(bytes(msg.data), dtype=np.uint16)
            expected = msg.height * (msg.step // 2)
            if image.size < expected:
                return None
            image = image[:expected].reshape(msg.height, msg.step // 2)
            image = image[:, : msg.width].astype(np.float32) * 0.001
            return image.copy()
        if msg.encoding == "32FC1":
            image = np.frombuffer(bytes(msg.data), dtype=np.float32)
            expected = msg.height * (msg.step // 4)
            if image.size < expected:
                return None
            image = image[:expected].reshape(msg.height, msg.step // 4)
            image = image[:, : msg.width]
            return image.copy()
        return None

    def _estimate_depth_and_point(self, depth_frame, x1, y1, x2, y2):
        if depth_frame is None:
            return None, None
        if isinstance(depth_frame, np.ndarray):
            return self._estimate_depth_and_point_from_array(depth_frame, x1, y1, x2, y2)
        cx = max(0, min((x1 + x2) // 2, self.color_width - 1))
        cy = max(0, min((y1 + y2) // 2, self.color_height - 1))
        distances = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px = max(0, min(cx + dx, self.color_width - 1))
                py = max(0, min(cy + dy, self.color_height - 1))
                dist = float(depth_frame.get_distance(px, py))
                if self.depth_min_valid_m <= dist <= self.depth_max_valid_m:
                    distances.append(dist)
        if not distances:
            return None, None
        distance_m = float(np.median(np.asarray(distances, dtype=np.float32)))
        point = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [float(cx), float(cy)], distance_m)
        return distance_m, (float(point[0]), float(point[1]), float(point[2]))

    def _estimate_depth_and_point_from_array(self, depth_image, x1, y1, x2, y2):
        if not self._camera_info_ready:
            return None, None
        height, width = depth_image.shape[:2]
        cx = max(0, min((x1 + x2) // 2, width - 1))
        cy = max(0, min((y1 + y2) // 2, height - 1))
        distances = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px = max(0, min(cx + dx, width - 1))
                py = max(0, min(cy + dy, height - 1))
                dist = float(depth_image[py, px])
                if self.depth_min_valid_m <= dist <= self.depth_max_valid_m:
                    distances.append(dist)
        if not distances:
            return None, None
        distance_m = float(np.median(np.asarray(distances, dtype=np.float32)))
        x = (float(cx) - self._intr_cx) / self._intr_fx * distance_m
        y = (float(cy) - self._intr_cy) / self._intr_fy * distance_m
        z = distance_m
        return distance_m, (x, y, z)

    def _publish_outputs(self, header: Header, detections):
        det_array = Detection2DArray()
        det_array.header = header
        for det in detections:
            msg = Detection2D()
            msg.bbox.center.position.x = float((det.x1 + det.x2) / 2.0)
            msg.bbox.center.position.y = float((det.y1 + det.y2) / 2.0)
            msg.bbox.size_x = float(det.x2 - det.x1)
            msg.bbox.size_y = float(det.y2 - det.y1)
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = self.fire_class_name
            result.hypothesis.score = float(det.score)
            if det.point_xyz is not None:
                result.pose.pose.position.x = float(det.point_xyz[0])
                result.pose.pose.position.y = float(det.point_xyz[1])
                result.pose.pose.position.z = float(det.point_xyz[2])
                result.pose.pose.orientation.w = 1.0
            msg.results.append(result)
            det_array.detections.append(msg)
        self.detections_pub.publish(det_array)

        best_candidates = [det for det in detections if det.point_xyz is not None]
        if best_candidates:
            best = max(best_candidates, key=lambda d: d.score)
            point_msg = PointStamped()
            point_msg.header = header
            point_msg.point.x = float(best.point_xyz[0])
            point_msg.point.y = float(best.point_xyz[1])
            point_msg.point.z = float(best.point_xyz[2])
            self.point_pub.publish(point_msg)

            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.pose.position.x = float(best.point_xyz[0])
            pose_msg.pose.position.y = float(best.point_xyz[1])
            pose_msg.pose.position.z = float(best.point_xyz[2])
            pose_msg.pose.orientation.w = 1.0
            self.pose_pub.publish(pose_msg)

    def _draw(self, frame, detections, fps):
        vis = frame.copy()
        for det in detections:
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 2)
            label = f"{self.fire_class_name} {det.score:.2f}"
            if det.distance_m is not None:
                label += f" {det.distance_m:.2f}m"
            if det.point_xyz is not None:
                label += f" x={det.point_xyz[0]:.2f} y={det.point_xyz[1]:.2f}"
            cv2.putText(
                vis,
                label,
                (det.x1, max(24, det.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.putText(vis, f"FPS {fps:.2f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vis, "Q/ESC quit", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def _run_direct_loop(self):
        if self.display:
            cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as exc:
                self.get_logger().error(f"wait_for_frames failed: {exc}")
                time.sleep(0.1)
                continue
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            self._process_frame(color, depth_frame, self.frame_id)

    def _process_once(self):
        if not self._ros_mode_active:
            return
        with self._frame_lock:
            color = None if self._latest_color is None else self._latest_color.copy()
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            frame_id = self._latest_ros_frame_id or self.frame_id
        if color is None:
            return
        if depth is None and not self._camera_info_ready:
            self._process_frame(color, None, frame_id)
            return
        self._process_frame(color, depth, frame_id)

    def _process_frame(self, color, depth_frame, frame_id):
        detections = self.detector.infer(color, self.conf_thres, self.iou_thres)
        for idx, det in enumerate(detections):
            distance_m, point_xyz = self._estimate_depth_and_point(depth_frame, det.x1, det.y1, det.x2, det.y2)
            detections[idx].distance_m = distance_m
            detections[idx].point_xyz = point_xyz

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id or self.frame_id
        self._publish_outputs(header, detections)

        now = time.time()
        prev_t = getattr(self, "_prev_t", now)
        fps = 1.0 / max(now - prev_t, 1e-6)
        self._prev_t = now

        if self.display:
            if not getattr(self, "_window_created", False):
                cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
                self._window_created = True
            vis = self._draw(color, detections, fps)
            cv2.imshow(self.display_window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                self.get_logger().info("display requested stop")
                rclpy.shutdown()

    def destroy_node(self):
        self._stop_event.set()
        if self._direct_thread and self._direct_thread.is_alive():
            self._direct_thread.join(timeout=1.0)
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        if self.display:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FireYoloBPUNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
