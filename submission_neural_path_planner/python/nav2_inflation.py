"""
nav2_inflation.py — 精确复刻 Nav2 Costmap2D 的 InflationLayer。

为什么必须复刻这个公式(而不是简单"墙往外画粗"):
  Nav2 的膨胀不是硬边界,而是一条**衰减代价曲线**。
  你的 costmap 配置:
    robot_radius = 0.20
    inflation_radius = 0.25
    cost_scaling_factor = 3.0
  合成的有效禁区半径 = robot_radius + inflation_radius = 0.45m

  Nav2 cost 值刻度(决定模型能不能走):
    254 = LETHAL         墙本身,绝对禁区
    253 = INSCRIBED      机器人在此位必撞(内切圆压到墙),硬禁区
    1~252 = 衰减层        cost 随距离指数下降,规划器会尽量避开但理论上可走
    0   = 自由

  关键: 模型的"障碍"定义必须和 NavfnPlanner 一致。
  NavfnPlanner 用 lethal+inscribed(>=253)作硬障碍,衰减层只影响偏好。
  所以我们的合成训练数据,障碍 = cost >= 253 的格子。

公式来源: nav2_costmap_2d/src/inflation_layer.cpp 的 inflatedCost()
  设一点到最近障碍的距离为 d(单位: cell),
    inscribed = robot_radius / resolution            (内切半径,格)
    inflation_range = inscribed + inflation_radius / resolution
  cost 分段:
    d <= inscribed              → 253 (INSCRIBED)
    inscribed < d <= range      → round(252 * exp(-factor * (d - inscribed)))
    d > range                   → 0

这样训练数据的"障碍分布"和部署时 Nav2 实际给的 costmap 完全一致,
模型学到的"A* 行为"就是在同一个膨胀规则下的规划。
"""

import numpy as np
from scipy import ndimage


def inflate_nav2(grid, robot_radius_m, inflation_radius_m, cost_scaling_factor,
                 resolution):
    """复刻 Nav2 InflationLayer。

    参数:
      grid: (H,W) 0=自由 1=墙
      robot_radius_m, inflation_radius_m, cost_scaling_factor: Nav2 参数
      resolution: m/cell

    返回:
      cost: (H,W) uint8, Nav2 风格 cost 值
      hard_obstacle: (H,W) bool, True=cost>=253(模型不可走的禁区)
    """
    inscribed = robot_radius_m / resolution                       # 内切半径(格)
    inflation_range = inscribed + inflation_radius_m / resolution # 总膨胀范围(格)

    # 每个自由格到最近墙的距离(欧氏,格)
    # distance_transform_edt: 输入非零的位置,计算到零位置的距离。
    # 我们要"自由格到墙的距离",墙是 1,所以用 (grid==0) 作输入。
    free = (grid == 0)
    dist = ndimage.distance_transform_edt(free)  # 自由格到最近非自由(墙/边界)的距离

    # 计算 cost
    cost = np.zeros(grid.shape, dtype=np.float64)
    # 段1: 内切区(机器人必撞) -> 253
    # 段2: 衰减区 -> 252 * exp(-factor*(d-inscribed))
    in_inscribed = (dist <= inscribed) & free
    in_decay = (dist > inscribed) & (dist <= inflation_range) & free
    cost[in_inscribed] = 253.0
    cost[in_decay] = 252.0 * np.exp(-cost_scaling_factor * (dist[in_decay] - inscribed))
    # 墙本身
    cost[grid == 1] = 254.0
    cost = np.clip(cost, 0, 254).astype(np.uint8)

    # 硬禁区: NavfnPlanner 的障碍定义是 lethal(254) + inscribed(253)
    # 模型在这个集合外跑 A*,保证不穿墙也不贴墙
    hard_obstacle = cost >= 253

    return cost, hard_obstacle


if __name__ == "__main__":
    # 自检: 复刻你的真实参数
    # robot_radius=0.20, inflation_radius=0.25, cost_scaling=3.0, res=0.05
    # 一面墙在地图中央,看膨胀结果
    g = np.zeros((100, 100), dtype=np.uint8)
    g[48:52, :] = 1   # 中间一条横墙(4格粗)
    cost, hard = inflate_nav2(g, 0.20, 0.25, 3.0, 0.05)
    print("参数: robot_radius=0.20, inflation=0.25, scaling=3.0, res=0.05")
    print(f"内切半径 = {0.20/0.05} 格")
    print(f"总膨胀范围 = {0.20/0.05 + 0.25/0.05} 格")
    # 看距墙中心不同距离的 cost
    wall_center = 50
    print("\n距墙中心的 cost 衰减(向上看自由区):")
    for dy in range(0, 15):
        r = wall_center - dy
        if 0 <= r < 100:
            print(f"  距墙{dy:2d}格: cost={cost[r, 50]:3d}  hard_obstacle={hard[r, 50]}")
