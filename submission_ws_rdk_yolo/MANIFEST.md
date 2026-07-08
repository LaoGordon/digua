# ws_rdk_yolo 核心代码清单

RDK S100P BPU 火焰检测节点 —— 感知层。
用 YOLOv8n 在 BPU 上实时检测火焰，结合 RealSense 深度相机输出火焰的 3D 位置。

## 目录结构

```
submission_ws_rdk_yolo/
├── MANIFEST.md
├── node/
│   ├── fire_yolo_bpu_node.py      ★ BPU 火焰检测 ROS2 节点（537行）
│   ├── realsense_fire_yolo_bpu.py  独立运行版（无ROS，直接RealSense+BPU）
│   └── __init__.py
└── config/
    ├── fire_yolo_bpu_params.yaml   运行参数（模型路径/置信度/相机）
    ├── fire_yolo_bpu.launch.py     ROS2 launch
    ├── setup.py                    ament_python 构建
    ├── package.xml
    ├── run_fire_yolo_bpu_ros2.sh    ROS 模式启动脚本
    ├── run_fire_yolo_realsense_bpu.sh  独立模式启动脚本
    └── start_realsense_ros_gui.sh   RealSense 启动脚本
```

## 核心文件说明

### `fire_yolo_bpu_node.py`（537行）—— BPU 火焰检测节点

**职责**：从 RealSense 相机取 RGB+Depth → BPU 跑 YOLOv8n 检测火焰 → 深度反投影出 3D 坐标 → 发布检测结果。

**架构亮点**：

1. **双相机模式自适应**（`camera_mode: auto/ros/direct`）
   - `ros` 模式：订阅 ROS topic（`/camera/camera/color/image_raw`），适合多节点共享相机
   - `direct` 模式：直接调 `pyrealsense2`，不依赖 ROS 相机驱动
   - `auto` 模式：先探测 ROS topic，超时则回退 direct（鲁棒性设计）

2. **BPU 推理封装**（`FireYoloBPU` 类）
   - `HB_HBMRuntime(model_path)` —— 加载 `.hbm` 模型
   - `preprocess` —— letterbox 到 640×640 + BGR→RGB + 归一化
   - `infer` —— BPU 前向 + NMS 后处理
   - 解码 YOLOv8 输出（xywh→xyxy + 置信度过滤 + NMS）

3. **3D 位置估计**（`_estimate_depth_and_point`）
   - 检测框中心取 5×5 邻域深度中位数（抗噪）
   - 相机内参反投影：像素 + 深度 → 相机坐标系 3D 点
   - 深度有效范围过滤（0.1m ~ 5m）

4. **输出话题**：
   - `/fire/detections`（Detection2DArray）—— 所有检测框 + 3D 位置
   - `/fire/point`（PointStamped）—— 置信度最高的火焰 3D 点
   - `/fire/pose`（PoseStamped）—— 同上，pose 形式

### `realsense_fire_yolo_bpu.py`（198行）—— 独立运行版

不依赖 ROS 的精简版，直接 RealSense + BPU + OpenCV 显示。
适合快速测试 BPU 推理效果，不需要整个 ROS 栈。

## 模型

- `fire_yolov8n_s100p.hbm` —— YOLOv8n 编译为 RDK S100P BPU 格式（march=nash-m）
- 单类别检测（fire），输入 640×640
- 在 BPU 上推理，不占 CPU（与 NeuralPathPlanner 共享 BPU 算力）

## 运行参数要点（fire_yolo_bpu_params.yaml）

- `model_path: fire_yolov8n_s100p.hbm`
- `conf_thres: 0.25` / `iou_thres: 0.45` —— YOLO 检测阈值
- `camera_mode: ros` —— 用 ROS 相机话题
- `depth_max_valid_m: 5.0` —— 火焰检测最大有效距离 5m

## 在整个系统中的位置

```
[RealSense 相机] → /camera/.../image_raw
       ↓
[fire_yolo_bpu_node]（本包，BPU 检测）
       ↓ /fire/detections（2D框 + 3D位置）
[fire_pitch_aim_node]（瞄准层） / [patrol2_node]（巡逻层）
```

这是感知层的核心，提供"火在哪、多远"的信息给下游。
