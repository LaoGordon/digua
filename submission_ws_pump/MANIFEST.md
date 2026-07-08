# ws_pump 核心代码清单

Lite3 消防机器狗喷淋泵控制节点 —— 执行层。
巡逻到达火点后，通过 RDK 板载 GPIO 驱动水泵喷淋灭火。

## 目录结构

```
submission_ws_pump/
├── MANIFEST.md
├── node/                       ROS2 喷淋节点
│   ├── pump_spray_node.py      ★ 一次性喷淋节点（GPIO控制水泵，117行）
│   ├── pump_node.py            常驻泵控制节点（142行）
│   └── __init__.py
├── config/
│   ├── pump_params.yaml         喷淋参数（引脚/时长）
│   ├── pump_spray.launch.py     ROS2 launch
│   ├── setup.py                 ament_python 构建
│   └── package.xml
└── standalone/                  独立脚本（无ROS依赖）
    ├── pump_control.py          直接GPIO控制（调试用，117行）
    └── pump_hold_off.py         泵保持关闭（安全守护）
```

## 核心文件说明

### `pump_spray_node.py`（117行）—— 一次性喷淋节点

**职责**：启动后驱动 GPIO 引脚 HIGH 持续 `spray_duration_sec` 秒，然后强制 LOW，节点退出。

核心逻辑：
1. `GPIO.setmode(BOARD)` + `GPIO.setup(pin, OUT, initial=LOW)` —— 初始化 RDK 板载 GPIO（`Hobot.GPIO` 库）
2. 喷淋前**停止 `pump_hold_off.service`**（释放安全守护，允许出水）
3. `GPIO.output(pin, HIGH)` 持续设定时长
4. 喷淋后**强制 LOW** + **重启 holdoff 服务**（恢复安全守护）
5. 信号处理：SIGINT/SIGTERM 时也强制关泵（安全兜底）

**安全设计要点**：
- 引脚默认 LOW（上电即关泵）
- 退出时强制 LOW（无论正常/异常退出）
- 与 `pump_hold_off.service` 联动（holdoff 服务确保非喷淋期间泵绝对关闭）
- 使用 BOARD pin 37（RDK S100P 物理 GPIO）

### `pump_node.py`（142行）—— 常驻泵控制节点

订阅 ROS2 指令，可多次触发喷淋（相比 pump_spray_node 的一次性模式）。

### `pump_hold_off.py` —— 安全守护服务

作为 systemd user service 运行，周期性强制泵 GPIO 为 LOW。
确保即使喷淋节点崩溃，泵也不会意外持续出水。

## 硬件接口

- **RDK S100P GPIO**：BOARD pin 37（`Hobot.GPIO` 库）
- **水泵**：继电器/驱动板接 GPIO，HIGH=喷水 LOW=停止
- 喷淋时长默认 5.0 秒（`spray_duration_sec`）

## 在整个系统中的位置

```
[patrol2_node]（调度层）
   ↓ subprocess: ros2 launch pump_ros2 pump_spray.launch.py
[pump_spray_node]（本包，执行层）
   ↓ Hobot.GPIO BOARD pin 37
[水泵继电器] → 物理喷淋
```

patrol2 导航到火点后，通过子进程启动本节点执行一次性喷淋，喷完节点自动退出。
这是"检测→导航→瞄准→喷淋"全链路的最后一环（物理执行）。
