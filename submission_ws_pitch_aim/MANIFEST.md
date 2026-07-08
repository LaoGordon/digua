# ws_pitch_aim 核心代码清单

Lite3 俯仰角瞄准控制节点 —— 控制层。
根据火焰检测的像素位置，用 PI 控制器调整机器狗俯仰角（pitch axis），让喷淋口对准火焰。

## 目录结构

```
submission_ws_pitch_aim/
├── MANIFEST.md
├── node/
│   ├── fire_pitch_aim_node.py   ★ PI 瞄准控制器（265行）
│   ├── lite3_udp.py              Lite3 UDP 指令封装（含 pitch_axis 指令）
│   ├── test_hold_pitch.py        俯仰角保持测试（调试用）
│   └── __init__.py
└── config/
    ├── fire_pitch_aim_params.yaml  运行参数（PI增益/死区/限幅）
    ├── fire_pitch_aim.launch.py    ROS2 launch
    ├── setup.py                    ament_python 构建
    ├── package.xml
    ├── run_fire_pitch_aim.sh       瞄准模式启动脚本
    └── run_hold_pitch_10s.sh       保持俯仰角测试脚本
```

## 核心文件说明

### `fire_pitch_aim_node.py`（265行）—— PI 瞄准控制器

**职责**：订阅火焰检测结果 → 计算俯仰角误差 → PI 控制 → 发送 pitch_axis 指令到机器狗。

**控制算法核心**（`_compute_axis`）：

```
error_px = 火焰中心y - 图像中心y   （正=火焰在下方→需要低头, 负=火焰在上方→需要抬头）
```

1. **死区设计**（防抖震）：
   - `deadband_px`（4px）—— 误差小于此值，停止控制（进入 hold_center）
   - `release_deadband_px`（7px）—— 滞回，必须误差大于此值才退出 hold_center
   - 这避免了在目标附近反复震荡

2. **非对称 PI 控制**（抬头/低头分开调参）：
   - 抬头（error<0）：`up_kp_axis_per_px` + `up_ki_axis_per_px_sec`
   - 低头（error>0）：`down_kp_axis_per_px` + `down_ki_axis_per_px_sec`
   - 机器狗抬头和低头需要的力矩不同（重力影响），所以增益分开

3. **输出限幅**：
   - `min_axis` / `max_axis` —— pitch_axis 最小/最大值
   - `up_axis_limit` / `down_axis_limit` —— 方向分别限幅

4. **低通滤波**（`target_y_filter_alpha`）：
   - 检测结果先做 EMA 滤波再进控制器，抑制检测抖动

**初始化序列**（启动时自动）：
1. 发心跳（保活）
2. `switch_navi_mode()` —— 切导航模式
3. `stand_sit()` —— 让机器狗站立
4. `switch_pose_mode()` —— 切姿态模式（允许俯仰调整）

**目标丢失处理**：
- 超过 `target_timeout_sec`（0.4s）没收到检测 → 发送 pitch_axis=0（回正）

### `lite3_udp.py`（85行）—— Lite3 指令封装（增强版）

比 patrol2 的版本多了 pitch 控制相关指令：
- `ADJUST_PITCH = 0x21010130` —— 俯仰调整指令码
- `send_pitch_axis(value)` —— 发送俯仰角指令（int32）
- `DOC_PITCH_MIN/MAX` —— 文档定义的俯仰范围（±32767）
- 完整指令码表：心跳/站坐/navi/pose/move模式/俯仰调整

### `test_hold_pitch.py`（88行）—— 调试用

发送固定俯仰角并保持 10 秒，用于测试机器狗 pitch 响应特性、标定 kp 参数。

## 控制参数要点（fire_pitch_aim_params.yaml）

- `up_kp_axis_per_px: 220` / `down_kp_axis_per_px: 220` —— 每像素误差的 P 增益
- `up_ki_axis_per_px_sec: 18` / `down_ki_axis_per_px_sec: 18` —— 积分增益
- `up_bias_axis: 4500` / `down_bias_axis: 4500` —— 前馈偏置（补偿重力）
- `deadband_px: 4.0` / `release_deadband_px: 7.0` —— 滞回死区
- `control_hz: 20.0` —— 控制频率 20Hz
- `target_y_ratio: 0.5` —— 目标在图像垂直中心

## 在整个系统中的位置

```
[fire_yolo_bpu_node]（感知层）
       ↓ /fire/detections（火焰框中心y坐标）
[fire_pitch_aim_node]（本包，控制层）
       ↓ UDP pitch_axis 指令
[Lite3 运动控制器]（机器狗俯仰关节）
```

这是"检测→瞄准"闭环的控制核心，把火焰的像素位置转换为机器狗的俯仰运动。
