# NeuralPathPlanner 核心代码清单

BPU 加速神经全局路径规划器（Nav2 GlobalPlanner 插件）——
在 RDK S100P 上用 CNN 预测 cost-to-go 势场，替代 NavfnPlanner 的 Dijkstra，把规划算力从 CPU 卸载到 BPU。

## 目录结构

```
submission_neural_path_planner/
├── MANIFEST.md            ← 本文件
├── cpp/                   ← C++ 部署层（Nav2 插件 + BPU 推理 + 搜索算法）
│   ├── include/neural_path_planner/
│   │   ├── neural_planner.hpp      Nav2 GlobalPlanner 插件类定义
│   │   ├── planner_core.hpp        纯算法库头文件（gradientWalk/A*/planRigid/prunePath）
│   │   └── bpu_inference.hpp       BPU C++ 推理 wrapper（hb_dnn API 封装）
│   └── src/
│       ├── neural_planner.cpp      插件主逻辑：createPlan() 全流程
│       └── planner_core.cpp        路径搜索算法实现
├── python/                ← Python 训练/算法层
│   ├── ppcnet_model.py             PPCNet 模型定义（复刻 Caffagni 架构）
│   ├── field_search.py             势场路径提取（gradient_walk/astar_field）
│   ├── train_hybrid.py             势场训练（MaskedMSE + 假盆地惩罚）
│   ├── generate_field_dataset.py   代价感知势场训练数据生成
│   ├── planner_utils.py            ★ cost_aware_dijkstra_field（势场监督信号核心）
│   ├── nav2_inflation.py           精确复刻 Nav2 InflationLayer
│   └── export_ppcnet_onnx.py       导出 ONNX（供 hb_compile 编译为 BPU .bin/.bc）
├── integration/           ← Nav2 集成层
│   ├── CMakeLists.txt              构建配置（USE_BPU 编译开关）
│   ├── neural_path_planner_plugin.xml   pluginlib 插件注册
│   └── neural_planner_params.yaml       运行时参数示例
└── weights/               ← BPU 编译产物（板上可直接部署，运行必需）
    ├── ppcnet_full.bc                势场模型 BPU 编译版（hb_compile march=nash-m，板上实际部署，13MB）
    └── fire_yolov8n_s100p.hbm        火焰检测 YOLOv8n BPU 编译版（来自 ws_rdk_yolo，4.8MB）
```

> **注**：为满足提交包 <25MB 限制，未纳入 PyTorch 训练权重 `.pt`（单个 42MB）。
> 权重可用本包 `train_hybrid.py` 重新训练生成，再用 `export_ppcnet_onnx.py` + `hb_compile` 编译出 `.bc`。
> 这里只保留**板上可直接加载运行**的 BPU 编译产物。

## 模型文件说明

| 文件 | 类型 | 来源 | 用途 |
|------|------|------|------|
| `ppcnet_full.bc` | BPU 编译模型 | `hb_compile -m ppcnet_full.onnx --march nash-m` | **板上实际部署的文件**，BPU 直接加载 |
| `fire_yolov8n_s100p.hbm` | BPU 编译模型 | YOLOv8n 导出+编译 | 火焰检测，`ws_rdk_yolo` 节点加载 |

### 从代码复现权重的路径（.pt 已省略）

```
train_hybrid.py --data ./dataset_field --epochs 40   → ep40.pt (PyTorch 权重)
export_ppcnet_onnx.py --ckpt ep40.pt                  → ppcnet_full.onnx
hb_compile -m ppcnet_full.onnx --march nash-m         → ppcnet_full.bc (BPU 模型)
```

## 分层说明

### C++ 部署层（cpp/）—— 板子上运行的核心

| 文件 | 行数 | 作用 |
|------|------|------|
| `neural_planner.cpp` | 485 | **Nav2 插件主入口**。createPlan() 串联：costmap→障碍图→降采样到256×256→构造PE→BPU推理→走廊裁剪→搜索→剪枝→世界坐标 |
| `planner_core.cpp` | 365 | **搜索算法库**。gradientWalk（势场贪心，核心）、astarSearch（cost-weighted兜底）、planRigid（统一流程）、prunePath（捷径剪枝+密集插值） |
| `bpu_inference.hpp` | 115 | **BPU 推理封装**。load() + forward()，封装 hbDNNInferV2 + hbUCPSubmitTask + hbUCPWaitTaskDone |
| `neural_planner.hpp` | 124 | 插件类定义，InferenceBackend 枚举（STUB/ONNX/BPU 三后端） |
| `planner_core.hpp` | 189 | 算法库接口 + Cell/Path 类型 + 8连通邻居/octile距离/对角穿墙判定 |

### Python 训练层（python/）—— 论文创新点的来源

| 文件 | 行数 | 作用 |
|------|------|------|
| `planner_utils.py` | 172 | **★ 势场监督信号的数学核心**。cost_aware_dijkstra_field() 复刻 NavfnPlanner 的 navfn.cpp 代价传播公式（COST_NEUTRAL=50 + COST_FACTOR=0.8） |
| `ppcnet_model.py` | 208 | PPCNet 架构（base=64, 3层）、GaussianRelativePE 位置编码、PathPlannerLoss/DiceLoss |
| `field_search.py` | 237 | 势场路径提取算法（gradient_walk/steepest_descent/astar_field）+ 评估工具 |
| `generate_field_dataset.py` | 190 | 代价感知势场数据生成（合成迷宫/障碍/走廊 × Nav2 膨胀 × 代价加权 Dijkstra） |
| `train_hybrid.py` | 231 | 训练流程（MaskedMSE + FieldLossWithBasin 假盆地惩罚 + gradient_walk 端到端评估） |
| `nav2_inflation.py` | 97 | 精确复刻 Nav2 InflationLayer 的指数衰减代价曲线（保证训练/部署一致） |
| `export_ppcnet_onnx.py` | 192 | 导出 ONNX（完整版 + encoder-only 兜底），强制 IR=9 兼容 hb_compile |

### 集成层（integration/）—— 接入 Nav2

| 文件 | 作用 |
|------|------|
| `CMakeLists.txt` | USE_BPU=ON/OFF 编译开关，板子上链接 libdnn |
| `neural_path_planner_plugin.xml` | pluginlib 注册 NeuralPathPlanner 为 nav2_core::GlobalPlanner |
| `neural_planner_params.yaml` | 运行时参数（backend/model_path/corridor_width/w_cost 等） |

## 关键算法创新点（对应文件）

1. **学习型势场规划**：CNN 学 Dijkstra cost-to-go 势场，推理时一次前向 → gradientWalk 提取路径
   - 模型：`ppcnet_model.py` | 监督信号：`planner_utils.py:cost_aware_dijkstra_field`
   - 搜索：`planner_core.cpp:gradientWalk`

2. **cost 软代价搜索（撞墙修复）**：gradientWalk 的 f 值加 `w_cost * cost[i]` 软项，让搜索主动绕开 inflation 衰减带但不硬拒（不堵窄通道）
   - 实现：`planner_core.cpp:gradientWalk` 的 `costTerm` lambda
   - 配置：`neural_planner_params.yaml` 的 `w_cost: 0.3`

3. **走廊裁剪加速**：256×256 粗搜 → 膨胀走廊 → 全分辨率只在走廊内搜索（45ms → 10ms）
   - 实现：`neural_planner.cpp` 的 corridor 段

4. **密集插值修复打转**：prunePath 剪枝后在直连段间补密集点，保证 RegulatedPurePursuit 有足够 lookahead
   - 实现：`planner_core.cpp:prunePath` 的 dense 插值段

5. **BPU 静态输入适配**：把坐标→PE 的转换移到 CPU，BPU 只跑纯卷积（Conv/BN/ReLU/ConvTranspose/Add）
   - 导出：`export_ppcnet_onnx.py:PPCNetStaticInput`
   - 推理：`bpu_inference.hpp`

## 复现路径

```
# 1. 生成数据
python generate_field_dataset.py --n-samples 8000 --out-dir ./dataset_field

# 2. 训练
python train_hybrid.py --data ./dataset_field --out ./ckpt_field --epochs 40

# 3. 导出 ONNX
python export_ppcnet_onnx.py --ckpt ./ckpt_field/ep40.pt --out-dir ./models

# 4. BPU 编译（Docker 工具链，march=nash-m）
hb_compile -m ./models/ppcnet_full.onnx --march nash-m -i input 1x3x256x256 --skip compile

# 5. 板上部署（CMakeLists.txt: USE_BPU=ON）
colcon build --packages-select neural_path_planner --cmake-args -DUSE_BPU=ON
```

## 验证数据（684 次真机规划，2026-07-08）

- 连通率：100%（684/684）
- 不穿墙：100%
- 方法分布：100% gw（gradientWalk），0% thin/astar
- 延迟：均值 52.0ms / 中位数 48.4ms / p90 76.6ms / p99 103.5ms
- w_cost=0.3（cost 软项生效）

## 依赖

- C++ 部署：ROS2 Humble、Nav2、tf2、（板上）libdnn / hobot BPU 驱动
- Python 训练：PyTorch、NumPy、SciPy、ONNX、（验证用）ONNX Runtime
