// Copyright 2026 longkang
//
// planner_core.hpp — 纯C++神经路径规划算法核心。
//
// 移植自 Python neural_planner_core.py + field_search.py:
//   - gradientWalk: 沿势场下降的贪心搜索(核心)
//   - astar: 经典A*(兜底)
//   - extractWaypoints: 连通域提取途经点(从细线score)
//   - plan: 统一规划流程(rigid策略)
//
// 不依赖ROS, 不依赖推理后端 —— 纯算法库。
// 推理结果(势场/score)由调用方传入, 这里只做搜索。

#ifndef NEURAL_PATH_PLANNER__PLANNER_CORE_HPP_
#define NEURAL_PATH_PLANNER__PLANNER_CORE_HPP_

#include <vector>
#include <utility>
#include <string>
#include <queue>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdint>

namespace neural_path_planner
{

using Cell = std::pair<int, int>;  // (row, col)
using Path = std::vector<Cell>;

// 8连通邻居 (dr, dc, weight)
struct Neighbor
{
  int dr, dc;
  double weight;
};

const std::vector<Neighbor> NEIGHBORS_8 = {
  {-1, 0, 1.0}, {1, 0, 1.0}, {0, -1, 1.0}, {0, 1, 1.0},
  {-1, -1, 1.414}, {-1, 1, 1.414}, {1, -1, 1.414}, {1, 1, 1.414}
};

/// octile distance (8连通的最优启发式)
inline double octile(int r1, int c1, int r2, int c2)
{
  int dr = std::abs(r1 - r2), dc = std::abs(c1 - c2);
  return (dr + dc) + (1.414 - 2.0) * std::min(dr, dc);
}

/// 判断对角线穿墙(两侧同时障碍才禁)
inline bool isDiagonalBlocked(const std::vector<uint8_t> & obs, int H, int W,
                               int r, int c, int dr, int dc)
{
  if (dr != 0 && dc != 0) {
    int idx1 = (r + dr) * W + c;
    int idx2 = r * W + (c + dc);
    if (obs[idx1] > 0 && obs[idx2] > 0) return true;
  }
  return false;
}

/// 到达判定: 路径首尾都在容差内
inline bool reached(const Path & path, Cell start, Cell goal, int tol = 2)
{
  if (path.size() < 2) return false;
  int ds = std::abs(path.front().first - start.first) +
           std::abs(path.front().second - start.second);
  int dg = std::abs(path.back().first - goal.first) +
           std::abs(path.back().second - goal.second);
  return ds <= tol && dg <= tol;
}

/// 统计路径穿墙格数
inline int countWall(const Path & path, const std::vector<uint8_t> & obs, int W)
{
  int cnt = 0;
  for (const auto & [r, c] : path) {
    if (obs[r * W + c] > 0) cnt++;
  }
  return cnt;
}

/**
 * @brief gradient_walk: 沿势场下降的贪心搜索。
 *
 * 从start出发, 优先队列按 field + 0.1*octile + w_cost*cost 排序, 走向goal。
 * 势场goal处应最低(≈0), 离goal越远值越大。
 * cost软项: 让搜索主动绕开inflation梯度带(离墙近cost高 → 惩罚大),
 *           但仍可通行(不硬拒), 因此不会堵窄通道。w_cost=0时退化为纯势场搜索。
 *
 * @param field (H*W) 势场, float, 已归一化(障碍处设大值)
 * @param obs (H*W) 障碍, 0=自由 1=障碍
 * @param H, W 地图尺寸
 * @param start, goal (row, col)
 * @param cost (H*W) 归一化代价[0,1], 来自costmap(空向量=禁用cost感知)
 * @param w_cost cost软项权重(默认0=禁用)
 * @param max_steps 步数上限(默认20000)
 * @return Path 路径(含start), 失败返回空
 */
Path gradientWalk(
  const std::vector<float> & field,
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  const std::vector<float> & cost = {},
  float w_cost = 0.0f,
  int max_steps = 20000);

/**
 * @brief A* 搜索(几何最优, 兜底用)。
 *
 * cost-weighted: 边权 = step_cost + w_cost*cost[neighbor], 复刻NavfnPlanner做法。
 * 离墙近 → 累积代价大 → 路径绕远但安全。w_cost=0时退化为纯几何最短。
 */
Path astarSearch(
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  const std::vector<float> & cost = {},
  float w_cost = 0.0f);

/**
 * @brief 从细线score提取途经点(连通域质心)。
 *
 * 找 score>thresh 的连通域, 每个域取质心, 按距start排序。
 * 移植自 Python extract_waypoints。
 *
 * @param score (H*W) 细线score [0,1]
 * @return std::vector<Cell> 途经点列表(不含start/goal)
 */
std::vector<Cell> extractWaypoints(
  const std::vector<float> & score,
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  float thresh = 0.3f,
  int max_wps = 4,
  int min_sep = 20);

/**
 * @brief 规划流程(rigid策略)。
 *
 * 1. gradientWalk(start→goal), 成功返回
 * 2. 失败→细线提途经点→分段gradientWalk
 * 3. 段失败→A*兜底
 *
 * cost/w_cost透传给gradientWalk和astarSearch, 实现cost-aware搜索。
 *
 * @param field_predict 势场推理回调: (s,g)→field(H*W float)
 * @param thin_predict 细线推理回调: (s,g)→score(H*W float)
 * @param cost (H*W) 归一化代价[0,1], 空向量=禁用cost感知
 * @param w_cost cost软项权重(0=禁用)
 * @return Path 完整路径, method输出"gw"/"thin"/"astar"
 */
using FieldPredictFn = std::function<std::vector<float>(Cell, Cell)>;
using ThinPredictFn = std::function<std::vector<float>(Cell, Cell)>;

Path planRigid(
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  const FieldPredictFn & field_predict,
  const ThinPredictFn & thin_predict,
  std::string & method,
  const std::vector<float> & cost = {},
  float w_cost = 0.0f);

/**
 * @brief 路径后处理: 捷径剪枝。
 *
 * 贪心扫描路径点, 跳过冗余绕路(如被假盆地吸入凹槽又绕出):
 * 从当前点试连最远可达点, 若连线无障碍则跳过中间点。
 * 不修改路径安全性——每条连线都检查了不经过障碍。
 *
 * @param path 输入路径
 * @param obs 障碍图(0=自由, >0=障碍)
 * @param W 地图宽度
 * @return Path 剪枝后路径
 */
Path prunePath(const Path & path, const std::vector<uint8_t> & obs, int W,
               const std::vector<uint8_t> & costmap_data = {}, int costmap_W = 0,
               int max_cost = 0);

}  // namespace neural_path_planner

#endif  // NEURAL_PATH_PLANNER__PLANNER_CORE_HPP_
