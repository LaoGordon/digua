# fastlivo2_nav 核心代码清单

FAST-LIVO2 到 Nav2 的真机导航适配层 —— 把 SLAM 输出（odom/点云）适配为 Nav2 需要的标准接口（/odom、TF、/obstacle_points、2D 栅格地图）。

这是用户自主编写的适配层，**不属于** FAST-LIVO2 上游开源项目。

## 目录结构

```
submission_fastlivo2_nav/
├── MANIFEST.md
├── bridge/                    ★ FAST-LIVO2 → Nav2 桥接节点
│   ├── fastlivo_nav_bridge_node.cpp   (485行) odom重发布 + TF广播 + 地面过滤 + ICP局部定位
│   ├── bridge.yaml                    桥接参数
│   ├── bridge.launch.py               ROS2 launch
│   ├── CMakeLists.txt
│   └── package.xml
├── floor_mapper/              ★ 3D点云 → 2D栅格地图
│   ├── floor_mapper_node.cpp          (210行) 在线点云投影为 OccupancyGrid
│   ├── floor_mapper.yaml
│   ├── floor_mapper.launch.py
│   ├── CMakeLists.txt
│   └── package.xml
├── tools/                     ★ 离线工具脚本
│   ├── pcd_to_static_map.py           (400行) PCD→Nav2静态地图转换
│   ├── calibrate_time_offset.py       (39行)  IMU-LiDAR时间偏移标定
│   └── sysmon.py                      (213行) 系统监控(CPU/内存/BPU)
├── bringup/                   Nav2 集成配置
│   ├── neural_nav2_params.yaml        BPU神经规划器 Nav2 参数
│   ├── nav2_params.yaml               原版 NavfnPlanner Nav2 参数
│   ├── navigation_main.launch.py      完整导航启动入口
│   ├── navigation_bringup.launch.py   bridge+floor_mapper 启动
│   └── navigate_w_replanning_no_global_clear.xml  行为树
├── run_mapping_real.sh        建图启动脚本
├── run_navigation_real.sh     原版导航启动脚本
└── run_neural_navigation.sh   BPU神经导航启动脚本
```

## 核心文件说明

### bridge/fastlivo_nav_bridge_node.cpp（485行）—— SLAM→Nav2 桥接

**职责**：把 FAST-LIVO2 的输出适配为 Nav2 标准接口。

核心功能：
1. **odom 重发布**：订阅 FAST-LIVO2 的 `/aft_mapped_to_init`（自定义里程计），重发布为标准 `/odom`
2. **TF 广播**：发布 `map→odom`（静态或 ICP 校正）+ `odom→base`（来自 SLAM）
3. **障碍点云生成**：订阅 `/cloud_registered_lidar`，做**地面过滤**（z 范围裁剪），输出 `/obstacle_points` 供 costmap 用
4. **ICP 局部定位**（可选 `use_icp_localization`）：加载参考 PCD，用 ICP 做 map→odom 的实时校正，消除累积漂移

设计要点：
- 地面过滤参数 `ground_filter_min_z/max_z`：只保留离地 0.2~2.0m 的点作为障碍（过滤地面和天花板）
- map→odom 可配置初始偏移（`map_to_odom_x/y/z/yaw`），适配建图原点和导航原点不一致

### floor_mapper/floor_mapper_node.cpp（210行）—— 3D→2D 地图投影

**职责**：把 3D 点云实时投影为 2D OccupancyGrid，作为 Nav2 的动态地图层。

用途：不需要预先建 2D 地图时，直接从 SLAM 点云生成可导航的 2D 栅格。

### tools/pcd_to_static_map.py（400行）—— PCD→静态地图转换

**职责**：离线把建好的 3D PCD 点云转换为 Nav2 的 2D 静态地图（.pgm + .yaml）。

这是**建图→导航的关键衔接工具**：FAST-LIVO2 建图输出 PCD → 本工具转 2D → Nav2 用作 static_map。

### bringup/ —— Nav2 集成配置

- `neural_nav2_params.yaml`：BPU 神经规划器的完整 Nav2 配置（planner_server 用 NeuralPathPlanner 插件）
- `nav2_params.yaml`：原版 NavfnPlanner 配置（对比基线）
- 行为树 `navigate_w_replanning_no_global_clear.xml`：定制版导航行为树（持续重规划，不清全局代价图）

## 关于 FAST-LIVO2 本身（不在本提交内）

FAST-LIVO2 是上游开源项目（[github.com/LaoGordon/FAST-LIVO2](https://github.com/LaoGordon/FAST-LIVO2)），本工作区只包含对其的**适配层**，不包含 FAST-LIVO2 源码本身。

用户在 FAST-LIVO2 上做的工程适配（记录在 CHANGELOG）：
- MID360 雷达 IP / 外参配置修正
- IMU-LiDAR 时间戳同步修复（指数平滑偏移跟踪，commit b3385e2）
- ICP 特征阈值调优（适配倾斜安装的 MID360）

这些属于部署适配，如需作为"工程贡献"单独说明，见 FAST-LIVO2 仓库的 CHANGELOG_*.md。

## 在整个系统中的位置

```
[FAST-LIVO2 SLAM]（上游开源）
   ↓ /aft_mapped_to_init (odom)
   ↓ /cloud_registered_lidar (点云)
[fastlivo_nav_bridge]（本包，桥接）  ← 这是 SLAM 和 Nav2 之间的胶水层
   ↓ /odom + TF(map→odom→base)
   ↓ /obstacle_points (地面过滤后)
[floor_mapper / Nav2 costmap]（地图层）
   ↓
[NeuralPathPlanner / NavfnPlanner]（规划层）
```
