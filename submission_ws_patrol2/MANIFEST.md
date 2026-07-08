# ws_patrol2 核心代码清单

Lite3 四足机器狗自主巡逻灭火应用 —— 巡逻调度层。
沿预设航点导航 → 到达后触发喷淋灭火 → 拍照记录 → 返回起点。

## 目录结构

```
submission_ws_patrol2/
├── MANIFEST.md
├── node/
│   ├── patrol2_node.py      ★ 巡逻主逻辑（航点导航 + 喷淋协调 + 拍照记录）
│   ├── lite3_udp.py          Lite3 UDP 指令封装（心跳/站坐/navi模式）
│   └── __init__.py
└── config/
    ├── patrol2_params.yaml    运行参数（航点文件/超时/狗IP）
    ├── patrol2.launch.py      ROS2 launch
    ├── setup.py               ament_python 构建
    ├── package.xml
    └── run_patrol2.sh         启动脚本
```

## 核心文件说明

### `patrol2_node.py`（374行）—— 巡逻主逻辑

**职责**：串联整个巡逻流程的上层调度节点。

核心流程（`run_patrol()`）：
1. 加载航点（`patrol_waypoints.yaml`）
2. 等待 Nav2 action server（`/navigate_to_pose`）
3. 记录起点位姿（用于最终返回）
4. 逐个航点循环：
   - `navigate_to(waypoint)` —— 调 Nav2 导航到目标点
   - 喷淋延迟 `spray_delay_sec`
   - `start_pump_spray()` —— 子进程启动 `pump_ros2`（喷淋节点）
   - `wait_for_pump_spray()` —— 等喷淋完成
   - `capture_photo()` —— 保存现场照片
   - `append_record()` —— 写 JSONL 记录
5. 返回起点（可选）

**关键设计**：
- 通过 Nav2 的 `NavigateToPose` action 做导航（不是直接发 cmd_vel）
- 喷淋泵是独立 ROS2 包（`ws_pump`），用 `subprocess.Popen` 在新进程组启动
- TF 查询 `map→base` 获取机器狗实时位姿
- 整个流程不直接控制机器狗运动（符合"严禁发送移动指令"红线，运动交给 Nav2）

### `lite3_udp.py`（37行）—— Lite3 指令封装

UDP 协议封装，向机器狗运动控制器（`192.168.137.120:43893`）发指令：
- `send_heartbeat()` —— 心跳保活
- `switch_navi_mode()` —— 切换到导航模式（让机器狗接受 Nav2 的 cmd_vel）
- `stand_sit()` —— 站立/趴下切换
- 数据包格式：`struct.pack("<IiI", code, value, 0)`

## 与其他模块的关系

```
[patrol2_node]（本包，调度）
     │ navigate_to_pose action
     ↓
[Nav2 + NeuralPathPlanner]（导航栈，BPU路径规划）
     ↓ cmd_vel
[机器狗运动层]
     ↑ pump_spray（子进程）
[patrol2_node] 启动喷淋
```

patrol2 是上层"任务调度"，它不自己灭火、不自己导航，而是协调 Nav2（导航）和 pump_ros2（喷淋）两个子系统。

## 运行参数要点（patrol2_params.yaml）

- `nav_goal_timeout_sec: 180.0` —— 单航点导航超时 3 分钟
- `spray_delay_sec: 2.0` —— 到达后等 2 秒再喷淋（让机器狗站稳）
- `return_to_start: true` —— 巡逻完返回起点
- `dog_ip: 192.168.137.120` —— Lite3 运动控制器 IP
