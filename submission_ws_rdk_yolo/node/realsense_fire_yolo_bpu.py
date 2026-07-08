#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
from hobot_dnn import pyeasy_dnn


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    distance_m: float | None


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
        models = pyeasy_dnn.load(model_path)
        if not models:
            raise RuntimeError(f"failed to load model: {model_path}")
        self.model = models[0]
        self.input_size = input_size
        self.output_props = self.model.outputs[0].properties
        self.output_shape = tuple(self.output_props.shape)
        self.output_elems = int(np.prod(self.output_shape))
        self.output_channel_stride = self.output_props.stride[1] // np.dtype(np.float32).itemsize

    def preprocess(self, frame_bgr: np.ndarray):
        padded, scale, pad_left, pad_top = letterbox(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        return chw, scale, pad_left, pad_top

    def infer(self, frame_bgr: np.ndarray, conf_thres: float, iou_thres: float, depth_frame=None):
        input_tensor, scale, pad_left, pad_top = self.preprocess(frame_bgr)
        outputs = self.model.forward(input_tensor)
        raw = np.asarray(outputs[0].buffer, dtype=np.float32).reshape(-1)
        channels = self.output_shape[1]
        points = self.output_shape[2]
        padded = raw.reshape(channels, self.output_channel_stride)[:channels, :points]
        preds = padded.reshape(self.output_shape)
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
            distance_m = None
            if depth_frame is not None:
                cx = max(0, min((x1 + x2) // 2, frame_bgr.shape[1] - 1))
                cy = max(0, min((y1 + y2) // 2, frame_bgr.shape[0] - 1))
                dist = float(depth_frame.get_distance(cx, cy))
                if dist > 0:
                    distance_m = dist
            result.append(Detection(x1, y1, x2, y2, score, distance_m))
        return result


def draw_detections(frame: np.ndarray, detections: list[Detection], fps: float):
    vis = frame.copy()
    for det in detections:
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 2)
        label = f"fire {det.score:.2f}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.2f}m"
        cv2.putText(vis, label, (det.x1, max(24, det.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, f"FPS {fps:.2f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(vis, "Q/ESC quit", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/sunrise/ws_rdk_yolo/fire_yolov8n_s100p.hbm")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = FireYoloBPU(args.model)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    pipeline.start(config)
    if not args.no_display:
        cv2.namedWindow("fire_yolo_bpu", cv2.WINDOW_NORMAL)

    frame_count = 0
    prev_t = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            detections = detector.infer(color, args.conf_thres, args.iou_thres, depth_frame)

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            frame_count += 1

            if args.no_display:
                print(f"frame={frame_count} detections={len(detections)} fps={fps:.2f}")
            else:
                vis = draw_detections(color, detections, fps)
                cv2.imshow("fire_yolo_bpu", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
