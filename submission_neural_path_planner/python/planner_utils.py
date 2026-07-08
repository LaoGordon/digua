"""
planner_utils.py — A* 路径搜索 + 工具函数(训练监督信号生成用)。

A* 用作监督: 我们让 U-Net 学会模仿经典 A* 的输出,
但推理时不再做图搜索,只跑一次前向卷积(Neural A* encoder-only 思想)。
"""

import heapq
import numpy as np
from scipy.ndimage import binary_dilation


def astar(grid, start, goal):
    """经典 A* 路径搜索。

    参数:
      grid: (H,W) 0=自由 1=障碍
      start, goal: (row, col)
    返回: 路径格子列表(含起终点),失败返回 None。
    8 连通,对角线穿墙禁止(两侧同时障碍才禁止对角)。
    """
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal
    if grid[sr, sc] == 1 or grid[gr, gc] == 1:
        return None

    NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (1.414 - 2) * min(dr, dc)  # octile distance

    open_heap = [(h(sr, sc), 0.0, sr, sc)]
    g_cost = {(sr, sc): 0.0}
    came_from = {}
    closed = set()

    while open_heap:
        _, g, r, c = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path
        for dr, dc, w in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if grid[nr, nc] == 1:
                continue
            if dr != 0 and dc != 0:
                if grid[r + dr, c] == 1 and grid[r, c + dc] == 1:
                    continue
            ng = g + w
            if (nr, nc) in closed:
                continue
            if ng < g_cost.get((nr, nc), float("inf")):
                g_cost[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))
    return None


def path_to_mask(path, shape, dilate=1):
    """路径转二值掩码。dilate: 膨胀半径(让目标不是单像素细线,训练更稳)。"""
    m = np.zeros(shape, dtype=np.float32)
    for r, c in path:
        m[r, c] = 1.0
    if dilate > 0:
        m = binary_dilation(m, iterations=dilate).astype(np.float32)
    return m


def cost_aware_dijkstra_field(obs, cost_grid, goal, alpha=1.0, mode="navfn"):
    """代价感知 Dijkstra —— Nav2 衰减层代价作为边权。

    mode:
      "navfn"  → dist += COST_NEUTRAL + COST_FACTOR*cost   [复刻 NavfnPlanner 真实公式, 推荐]
                 COST_NEUTRAL=50, COST_FACTOR=0.8 (来自 navfn.cpp costNum)
                 自由格代价≈50, 贴墙格代价≈250 → 贴墙代价5x, 信号不被路径长度淹没
      "linear" → dist += edge_weight * (1 + alpha*cost/254)   [旧, 信号弱]
      "exp"    → 废弃(溢出), 保留仅为兼容
      "add"    → dist += edge_weight + alpha * cost/254        [旧, 信号弱]

    返回: (H,W) float32 cost-to-go [0,1], 障碍处=NaN。
    """
    H, W = obs.shape
    gr, gc = goal
    if obs[gr, gc]:
        free = np.argwhere(obs == 0)
        if len(free) == 0:
            return np.zeros((H, W), dtype=np.float32)
        idx = np.argmin(np.abs(free[:, 0] - gr) + np.abs(free[:, 1] - gc))
        gr, gc = int(free[idx, 0]), int(free[idx, 1])

    INF = float("inf")
    dist = np.full((H, W), INF, dtype=np.float64)
    dist[gr, gc] = 0.0
    pq = [(0.0, gr, gc)]

    cost_f = cost_grid.astype(np.float64)
    # navfn 真实公式 (来自 navfn.cpp costNum): COST_NEUTRAL + COST_FACTOR*cost
    # COST_NEUTRAL=50: 每格基础代价(自由格也有), 路径长度×50 不会被避墙信号淹没
    # COST_FACTOR=0.8: 衰减层放大系数
    if mode == "navfn":
        COST_NEUTRAL = 50.0
        COST_FACTOR = 0.8
        cost_enter = np.where(obs, INF, COST_NEUTRAL + COST_FACTOR * cost_f)
        # 障碍(lethal cost=254) 单独设极高值
        cost_enter = np.where(cost_f >= 253, INF, cost_enter)
    elif mode == "linear":
        cost_enter = np.where(obs, INF, 1.0 + alpha * cost_f / 254.0)
    elif mode == "exp":
        cost_enter = np.where(obs, INF, np.exp(alpha * cost_f / 254.0))
    elif mode == "add":
        cost_enter = np.where(obs, INF, alpha * cost_f / 254.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc, w in NEI:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if dr != 0 and dc != 0 and obs[r + dr, c] and obs[r, c + dc]:
                continue
            ce = cost_enter[nr, nc]
            if not np.isfinite(ce):
                continue
            if mode in ("linear", "exp"):
                nd = d + w * ce
            else:  # navfn, add: 边权 + 格代价
                nd = d + w + ce
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(pq, (nd, nr, nc))

    field = np.where(np.isinf(dist), np.nan, dist).astype(np.float32)
    mx = np.nanmax(field)
    if mx > 1e-8:
        field = field / mx
    return field


def gaussian_peak(shape, point, sigma=3.0):
    """在 point 处放一个高斯峰(截断窗口加速)。"""
    H, W = shape
    pr, pc = point
    m = np.zeros((H, W), dtype=np.float32)
    r = int(sigma * 4)
    r0, r1 = max(0, pr - r), min(H, pr + r + 1)
    c0, c1 = max(0, pc - r), min(W, pc + r + 1)
    if r0 >= r1 or c0 >= c1:
        m[pr, pc] = 1.0
        return m
    yy, xx = np.mgrid[r0:r1, c0:c1]
    g = np.exp(-((yy - pr) ** 2 + (xx - pc) ** 2) / (2 * sigma * sigma))
    m[r0:r1, c0:c1] = g.astype(np.float32)
    return m
