// planner_core.cpp — 纯C++神经路径规划算法实现。
// 移植自 Python field_search.py + neural_planner_core.py。

#include "neural_path_planner/planner_core.hpp"

namespace neural_path_planner
{

Path gradientWalk(
  const std::vector<float> & field,
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  const std::vector<float> & cost,
  float w_cost,
  int max_steps)
{
  auto [sr, sc] = start;
  auto [gr, gc] = goal;
  if (obs[sr * W + sc] > 0 || obs[gr * W + gc] > 0) return {};

  using PQItem = std::tuple<float, int, int>;
  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<>> pq;

  std::vector<bool> visited(H * W, false);
  std::vector<int> came_from(H * W, -1);  // 用vector代替unordered_map(快10倍+)

  auto idx = [W](int r, int c) { return r * W + c; };

  // NaN安全: 障碍处的势场可能是NaN, 替换成大值
  auto fieldSafe = [&field](int i) {
    float v = field[i];
    return std::isnan(v) ? 1e9f : v;
  };

  // cost软项: cost非空且w_cost>0时启用, 否则返回0
  bool use_cost = !cost.empty() && w_cost > 0.0f;
  auto costTerm = [&](int i) -> float {
    return use_cost ? w_cost * cost[i] : 0.0f;
  };

  float f0 = fieldSafe(idx(sr, sc))
           + 0.1f * static_cast<float>(octile(sr, sc, gr, gc))
           + costTerm(idx(sr, sc));
  pq.push({f0, sr, sc});
  visited[idx(sr, sc)] = true;

  int steps = 0;
  while (!pq.empty() && steps < max_steps) {
    auto [f, r, c] = pq.top();
    pq.pop();

    if (std::abs(r - gr) + std::abs(c - gc) <= 2) {
      Path path;
      int cur = idx(r, c);
      while (cur != idx(sr, sc)) {
        path.emplace_back(cur / W, cur % W);
        cur = came_from[cur];
        if (cur < 0) break;
      }
      path.emplace_back(sr, sc);
      std::reverse(path.begin(), path.end());
      return path;
    }

    for (const auto & [dr, dc, w] : NEIGHBORS_8) {
      int nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
      int ni = idx(nr, nc);
      if (obs[ni] > 0) continue;
      if (isDiagonalBlocked(obs, H, W, r, c, dr, dc)) continue;
      if (visited[ni]) continue;
      visited[ni] = true;
      came_from[ni] = idx(r, c);
      float nf = fieldSafe(ni)
               + 0.1f * static_cast<float>(octile(nr, nc, gr, gc))
               + costTerm(ni);
      pq.push({nf, nr, nc});
    }
    steps++;
  }
  return {};
}

Path astarSearch(
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  const std::vector<float> & cost,
  float w_cost)
{
  auto [sr, sc] = start;
  auto [gr, gc] = goal;
  if (obs[sr * W + sc] > 0 || obs[gr * W + gc] > 0) return {};

  using PQItem = std::tuple<float, float, int, int>;  // (f, g, r, c)
  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<>> pq;
  // 用vector代替unordered_map/set(大地图性能关键)
  std::vector<float> g_cost(H * W, std::numeric_limits<float>::infinity());
  std::vector<int> came_from(H * W, -1);
  std::vector<bool> closed(H * W, false);

  auto idx = [W](int r, int c) { return r * W + c; };

  // cost-weighted边权: step_cost + w_cost*cost[neighbor] (复刻NavfnPlanner)
  bool use_cost = !cost.empty() && w_cost > 0.0f;
  auto stepW = [&](int ni, double w) -> float {
    return use_cost ? static_cast<float>(w) + w_cost * cost[ni]
                    : static_cast<float>(w);
  };

  pq.push({static_cast<float>(octile(sr, sc, gr, gc)), 0.0f, sr, sc});
  g_cost[idx(sr, sc)] = 0.0f;

  while (!pq.empty()) {
    auto [f, g, r, c] = pq.top();
    pq.pop();
    int ci = idx(r, c);
    if (closed[ci]) continue;
    closed[ci] = true;

    if (r == gr && c == gc) {
      Path path;
      int cur = ci;
      while (cur != idx(sr, sc)) {
        path.emplace_back(cur / W, cur % W);
        cur = came_from[cur];
        if (cur < 0) break;
      }
      path.emplace_back(sr, sc);
      std::reverse(path.begin(), path.end());
      return path;
    }

    for (const auto & [dr, dc, w] : NEIGHBORS_8) {
      int nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
      int ni = idx(nr, nc);
      if (obs[ni] > 0) continue;
      if (isDiagonalBlocked(obs, H, W, r, c, dr, dc)) continue;
      if (closed[ni]) continue;
      float ng = g + stepW(ni, w);
      if (ng < g_cost[ni]) {
        g_cost[ni] = ng;
        came_from[ni] = ci;
        float nf = ng + static_cast<float>(octile(nr, nc, gr, gc));
        pq.push({nf, ng, nr, nc});
      }
    }
  }
  return {};
}

std::vector<Cell> extractWaypoints(
  const std::vector<float> & score,
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  float thresh,
  int max_wps,
  int min_sep)
{
  // 二值化 + 连通域标记(BFS)
  std::vector<bool> binary(H * W, false);
  for (int i = 0; i < H * W; i++) {
    binary[i] = (score[i] > thresh) && (obs[i] == 0);
  }

  std::vector<int> label(H * W, 0);
  int cur_label = 0;
  std::vector<std::pair<Cell, int>> components;  // (centroid, size)

  for (int r = 0; r < H; r++) {
    for (int c = 0; c < W; c++) {
      int idx = r * W + c;
      if (!binary[idx] || label[idx] > 0) continue;
      cur_label++;
      // BFS
      std::queue<std::pair<int, int>> bfs;
      bfs.push({r, c});
      label[idx] = cur_label;
      long sum_r = 0, sum_c = 0, size = 0;
      while (!bfs.empty()) {
        auto [br, bc] = bfs.front();
        bfs.pop();
        sum_r += br; sum_c += bc; size++;
        for (const auto & [dr, dc, w] : NEIGHBORS_8) {
          int nr = br + dr, nc = bc + dc;
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          int ni = nr * W + nc;
          if (binary[ni] && label[ni] == 0) {
            label[ni] = cur_label;
            bfs.push({nr, nc});
          }
        }
      }
      if (size >= 3) {
        components.push_back({{static_cast<int>(sum_r / size), static_cast<int>(sum_c / size)},
                              static_cast<int>(size)});
      }
    }
  }

  // 按距start远近排序
  auto [sr, sc] = start;
  std::sort(components.begin(), components.end(),
    [sr, sc](const auto & a, const auto & b) {
      int da = std::abs(a.first.first - sr) + std::abs(a.first.second - sc);
      int db = std::abs(b.first.first - sr) + std::abs(b.first.second - sc);
      return da < db;
    });

  // 筛选: 离start/goal足够远
  auto [gr, gc] = goal;
  std::vector<Cell> wps;
  for (const auto & [centroid, size] : components) {
    int d_s = std::abs(centroid.first - sr) + std::abs(centroid.second - sc);
    int d_g = std::abs(centroid.first - gr) + std::abs(centroid.second - gc);
    if (d_s > min_sep && d_g > min_sep) {
      wps.push_back(centroid);
      if (static_cast<int>(wps.size()) >= max_wps) break;
    }
  }
  return wps;
}

Path planRigid(
  const std::vector<uint8_t> & obs,
  int H, int W,
  Cell start, Cell goal,
  const FieldPredictFn & field_predict,
  const ThinPredictFn & thin_predict,
  std::string & method,
  const std::vector<float> & cost,
  float w_cost)
{
  // 1. 直接gw
  auto field = field_predict(start, goal);
  auto p = gradientWalk(field, obs, H, W, start, goal, cost, w_cost);
  if (!p.empty() && reached(p, start, goal)) {
    method = "gw";
    return p;
  }

  // 2. 细线提途经点 → 分段gw
  auto score = thin_predict(start, goal);
  auto wps = extractWaypoints(score, obs, H, W, start, goal);
  if (wps.empty()) {
    // 无途经点 → A*兜底
    method = "astar";
    return astarSearch(obs, H, W, start, goal, cost, w_cost);
  }

  // 分段: start → wp1 → wp2 → ... → goal
  std::vector<Cell> segs;
  segs.push_back(start);
  for (const auto & wp : wps) segs.push_back(wp);
  segs.push_back(goal);

  Path full;
  for (size_t i = 0; i < segs.size() - 1; i++) {
    auto seg_field = field_predict(segs[i], segs[i + 1]);
    auto seg_p = gradientWalk(seg_field, obs, H, W, segs[i], segs[i + 1], cost, w_cost);
    if (seg_p.empty() || !reached(seg_p, segs[i], segs[i + 1])) {
      // 段失败 → A*
      seg_p = astarSearch(obs, H, W, segs[i], segs[i + 1], cost, w_cost);
      if (seg_p.empty()) {
        method = "astar";
        return astarSearch(obs, H, W, start, goal, cost, w_cost);
      }
    }
    if (i == 0) {
      full.insert(full.end(), seg_p.begin(), seg_p.end());
    } else {
      full.insert(full.end(), seg_p.begin() + 1, seg_p.end());  // 去重复点
    }
  }
  method = "thin";
  return full;
}

Path prunePath(const Path & path, const std::vector<uint8_t> & obs, int W,
               const std::vector<uint8_t> & costmap_data, int costmap_W,
               int max_cost)
{
  if (path.size() < 3) return path;

  int H = static_cast<int>(obs.size()) / W;
  bool use_clearance = !costmap_data.empty() && max_cost > 0;

  // Bresenham 直线检查 + clearance:
  // 无障碍(硬约束) + 沿线cost不超过max_cost(软约束, 离墙足够远)
  auto lineClear = [&](Cell a, Cell b) -> bool {
    int r0 = a.first, c0 = a.second;
    int r1 = b.first, c1 = b.second;
    int dr = std::abs(r1 - r0), dc = std::abs(c1 - c0);
    int sr = r0 < r1 ? 1 : -1, sc = c0 < c1 ? 1 : -1;
    int err = dr - dc;
    while (r0 != r1 || c0 != c1) {
      // 硬约束: 不能在障碍上
      if (obs[r0 * W + c0] > 0) return false;
      // 软约束: clearance检查(用costmap原始cost)
      if (use_clearance) {
        int cm_idx = r0 * costmap_W + c0;
        if (cm_idx >= 0 && cm_idx < static_cast<int>(costmap_data.size())) {
          if (costmap_data[cm_idx] > max_cost) return false;
        }
      }
      int e2 = 2 * err;
      if (e2 > -dc) { err -= dc; r0 += sr; }
      if (e2 < dr) { err += dr; c0 += sc; }
    }
    // 检查终点
    if (obs[r0 * W + c0] > 0) return false;
    if (use_clearance) {
      int cm_idx = r0 * costmap_W + c0;
      if (cm_idx >= 0 && cm_idx < static_cast<int>(costmap_data.size())) {
        if (costmap_data[cm_idx] > max_cost) return false;
      }
    }
    return true;
  };

  Path result;
  result.push_back(path[0]);
  size_t anchor = 0;

  while (anchor < path.size() - 2) {
    size_t best = anchor + 1;
    // 从最远端往前找第一个能直连的点
    for (size_t j = path.size() - 1; j > anchor + 1; j--) {
      if (lineClear(path[anchor], path[j])) {
        best = j;
        break;
      }
    }
    result.push_back(path[best]);
    anchor = best;
  }

  // 保证 goal 在最后
  if (result.back() != path.back())
    result.push_back(path.back());

  // 插值: 在剪枝后的直连段之间补密集点, 保证RegulatedPurePursuit有足够的lookahead点
  Path dense;
  for (size_t i = 0; i < result.size() - 1; i++) {
    dense.push_back(result[i]);
    int dr = result[i + 1].first - result[i].first;
    int dc = result[i + 1].second - result[i].second;
    int dist = std::max(std::abs(dr), std::abs(dc));
    if (dist > 1) {
      int steps = dist;  // 每格一个点
      for (int s = 1; s < steps; s++) {
        dense.push_back({result[i].first + dr * s / steps,
                         result[i].second + dc * s / steps});
      }
    }
  }
  dense.push_back(result.back());

  return dense;
}

}  // namespace neural_path_planner
