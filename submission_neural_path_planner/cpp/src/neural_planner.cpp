// neural_planner.cpp — Nav2 GlobalPlanner 插件
//
// search_mode 参数控制消融:
//   "full"      → 原始costmap分辨率搜索(不穿墙, 当前方案)
//   "downsample"→ 256x256降采样搜索+放大回(旧方案, 可能穿墙)
//
// 切换: 改 neural_nav2_params.yaml 里 search_mode 即可

#include "neural_path_planner/neural_planner.hpp"
#include "neural_path_planner/planner_core.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/node_utils.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

namespace neural_path_planner
{

NeuralPathPlanner::NeuralPathPlanner() = default;
NeuralPathPlanner::~NeuralPathPlanner() = default;

void NeuralPathPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  auto node = parent.lock();
  if (!node) throw std::runtime_error("NeuralPathPlanner: parent node expired");

  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  name_ = std::move(name);
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  global_frame_ = costmap_ros_->getGlobalFrameID();

  std::string backend_str;
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".backend", rclcpp::ParameterValue(std::string("stub")));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".field_model_path", rclcpp::ParameterValue(std::string("")));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".thin_model_path", rclcpp::ParameterValue(std::string("")));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".collision_cost_threshold", rclcpp::ParameterValue(253.0));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".strategy", rclcpp::ParameterValue(std::string("rigid")));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".search_mode", rclcpp::ParameterValue(std::string("full")));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".corridor_width", rclcpp::ParameterValue(0));
  nav2_util::declare_parameter_if_not_declared(node, name_ + ".w_cost", rclcpp::ParameterValue(0.3));

  node->get_parameter(name_ + ".backend", backend_str);
  node->get_parameter(name_ + ".field_model_path", field_model_path_);
  node->get_parameter(name_ + ".thin_model_path", thin_model_path_);
  node->get_parameter(name_ + ".collision_cost_threshold", collision_cost_threshold_);
  node->get_parameter(name_ + ".strategy", strategy_);
  node->get_parameter(name_ + ".search_mode", search_mode_);
  node->get_parameter(name_ + ".corridor_width", corridor_width_);
  node->get_parameter(name_ + ".w_cost", w_cost_);

  if (backend_str == "bpu") backend_ = InferenceBackend::BPU;
  else if (backend_str == "onnx") backend_ = InferenceBackend::ONNX;
  else backend_ = InferenceBackend::STUB;

  RCLCPP_INFO(logger_, "[NeuralPathPlanner] backend=%s search_mode=%s corridor_width=%d w_cost=%.3f strategy=%s size=%ux%u",
    backend_str.c_str(), search_mode_.c_str(), corridor_width_, w_cost_, strategy_.c_str(),
    costmap_->getSizeInCellsX(), costmap_->getSizeInCellsY());

#ifdef USE_BPU
  if (backend_ == InferenceBackend::BPU && !field_model_path_.empty()) {
    field_bpu_ = std::make_unique<BpuInference>();
    if (!field_bpu_->load(field_model_path_)) {
      RCLCPP_ERROR(logger_, "[NeuralPathPlanner] 势场BPU模型加载失败");
      field_bpu_.reset(); backend_ = InferenceBackend::STUB;
    } else {
      RCLCPP_INFO(logger_, "[NeuralPathPlanner] 势场BPU模型加载成功");
    }
    // 细线模型(可选, gw失败时提途经点)
    if (!thin_model_path_.empty()) {
      thin_bpu_ = std::make_unique<BpuInference>();
      if (!thin_bpu_->load(thin_model_path_)) {
        RCLCPP_WARN(logger_, "[NeuralPathPlanner] 细线BPU模型加载失败, 细线禁用");
        thin_bpu_.reset();
      } else {
        RCLCPP_INFO(logger_, "[NeuralPathPlanner] 细线BPU模型加载成功");
      }
    }
  }
#else
  if (backend_ == InferenceBackend::BPU) {
    RCLCPP_WARN(logger_, "[NeuralPathPlanner] 未编译BPU, 退化为stub");
    backend_ = InferenceBackend::STUB;
  }
#endif
}

void NeuralPathPlanner::cleanup() { tf_.reset(); costmap_ros_.reset(); costmap_ = nullptr; }
void NeuralPathPlanner::activate() {}
void NeuralPathPlanner::deactivate() {}

std::vector<float> bilinearResize(const std::vector<float> & src, int sH, int sW, int dH, int dW)
{
  std::vector<float> dst(dH * dW);
  for (int r = 0; r < dH; r++) {
    float sr = static_cast<float>(r) * (sH - 1) / std::max(dH - 1, 1);
    int r0 = static_cast<int>(sr); int r1 = std::min(r0 + 1, sH - 1); float fr = sr - r0;
    for (int c = 0; c < dW; c++) {
      float sc = static_cast<float>(c) * (sW - 1) / std::max(dW - 1, 1);
      int c0 = static_cast<int>(sc); int c1 = std::min(c0 + 1, sW - 1); float fc = sc - c0;
      dst[r * dW + c] = src[r0 * sW + c0]*(1-fr)*(1-fc) + src[r0 * sW + c1]*(1-fr)*fc +
                        src[r1 * sW + c0]*fr*(1-fc) + src[r1 * sW + c1]*fr*fc;
    }
  }
  return dst;
}

nav_msgs::msg::Path NeuralPathPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path path;
  path.header.stamp = clock_ ? clock_->now() : rclcpp::Clock().now();
  path.header.frame_id = global_frame_;
  ++plan_count_;
  auto t0 = std::chrono::steady_clock::now();

  // ---- 坐标转换 ----
  unsigned int mx_s, my_s, mx_g, my_g;
  if (!worldToMap(start.pose.position.x, start.pose.position.y, mx_s, my_s) ||
      !worldToMap(goal.pose.position.x, goal.pose.position.y, mx_g, my_g))
  {
    return path;
  }

  const int costmap_H = static_cast<int>(costmap_->getSizeInCellsY());
  const int costmap_W = static_cast<int>(costmap_->getSizeInCellsX());
  const int MS = 256;

  // 决定搜索分辨率
  bool full_res = (search_mode_ == "full");
  int H = full_res ? costmap_H : MS;
  int W = full_res ? costmap_W : MS;
  float sr = static_cast<float>(MS) / costmap_H;
  float sc = static_cast<float>(MS) / costmap_W;

  // start/goal
  Cell start_cell, goal_cell;
  if (full_res) {
    start_cell = {static_cast<int>(my_s), static_cast<int>(mx_s)};
    goal_cell = {static_cast<int>(my_g), static_cast<int>(mx_g)};
  } else {
    start_cell = {static_cast<int>(my_s * sr), static_cast<int>(mx_s * sc)};
    goal_cell = {static_cast<int>(my_g * sr), static_cast<int>(mx_g * sc)};
  }

  // 障碍图 (full: 原始分辨率, downsample: 256x256)
  std::vector<uint8_t> obs(H * W, 0);
  if (full_res) {
    for (int r = 0; r < H; r++)
      for (int c = 0; c < W; c++)
        obs[r * W + c] = (costmap_->getCost(c, r) >= collision_cost_threshold_) ? 1 : 0;
  } else {
    for (int r = 0; r < MS; r++)
      for (int c = 0; c < MS; c++) {
        int r0 = static_cast<int>(r / sr);
        int r1 = std::min(static_cast<int>((r + 1) / sr), costmap_H);
        int c0 = static_cast<int>(c / sc);
        int c1 = std::min(static_cast<int>((c + 1) / sc), costmap_W);
        bool blocked = false;
        for (int cr = r0; cr < r1 && !blocked; cr++)
          for (int cc = c0; cc < c1; cc++)
            if (costmap_->getCost(cc, cr) >= collision_cost_threshold_) { blocked = true; break; }
        obs[r * MS + c] = blocked ? 1 : 0;
      }
  }

  // 归一化cost图 [0,1]: 让搜索主动绕开inflation梯度带(离墙近cost高)。
  // cost=0(自由) ~ cost=1(贴墙, 即cost≈253 INSCRIBED)。
  // 直接按getCost()/253归一化; ≥253的已被obs标为障碍, 不会进搜索, 无需特殊处理。
  // 仅在w_cost>0时构造, 省内存/时间。
  std::vector<float> cost_norm;
  if (w_cost_ > 0.0) {
    cost_norm.resize(H * W, 0.0f);
    const float inv_thr = 1.0f / static_cast<float>(collision_cost_threshold_);
    if (full_res) {
      for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
          cost_norm[r * W + c] = static_cast<float>(costmap_->getCost(c, r)) * inv_thr;
    } else {
      // 降采样: 取子区域cost均值(比max温和, 避免单格高cost过度惩罚)
      for (int r = 0; r < MS; r++)
        for (int c = 0; c < MS; c++) {
          int r0 = static_cast<int>(r / sr);
          int r1 = std::min(static_cast<int>((r + 1) / sr), costmap_H);
          int c0 = static_cast<int>(c / sc);
          int c1 = std::min(static_cast<int>((c + 1) / sc), costmap_W);
          long sum = 0, cnt = 0;
          for (int cr = r0; cr < r1; cr++)
            for (int cc = c0; cc < c1; cc++) { sum += costmap_->getCost(cc, cr); cnt++; }
          cost_norm[r * MS + c] = (cnt > 0 ? static_cast<float>(sum) / cnt : 0.0f) * inv_thr;
        }
    }
  }

  // start/goal 障碍检查
  if (obs[start_cell.first * W + start_cell.second] > 0 ||
      obs[goal_cell.first * W + goal_cell.second] > 0)
  {
    return makeStraightLine(start, goal);
  }

  // 降采样障碍图给BPU: zero-padding保持长宽比
  // pad成max(H,W)×max(H,W)正方形, padding区填障碍(1)
  // 再uniform缩放到256×256, 消除长宽比失真
  const int padded_size = std::max(costmap_H, costmap_W);
  const int pad_left = (padded_size - costmap_W) / 2;
  const int pad_top = (padded_size - costmap_H) / 2;
  const float uniform_scale = static_cast<float>(MS) / padded_size;

  std::vector<uint8_t> obs_bpu(MS * MS, 0);
  for (int r = 0; r < MS; r++)
    for (int c = 0; c < MS; c++) {
      // 256坐标 → padded坐标 → costmap坐标
      int pad_r0 = static_cast<int>(r / uniform_scale);
      int pad_r1 = std::min(static_cast<int>((r + 1) / uniform_scale), padded_size);
      int pad_c0 = static_cast<int>(c / uniform_scale);
      int pad_c1 = std::min(static_cast<int>((c + 1) / uniform_scale), padded_size);
      bool blocked = false;
      for (int pr = pad_r0; pr < pad_r1 && !blocked; pr++)
        for (int pc = pad_c0; pc < pad_c1 && !blocked; pc++) {
          // padded坐标 → costmap坐标 (减去padding偏移)
          int cr = pr - pad_top;
          int cc = pc - pad_left;
          if (cr < 0 || cr >= costmap_H || cc < 0 || cc >= costmap_W) {
            blocked = true;  // padding区 = 障碍
          } else {
            if (costmap_->getCost(cc, cr) >= collision_cost_threshold_) blocked = true;
          }
        }
      obs_bpu[r * MS + c] = blocked ? 1 : 0;
    }

  // ---- 推理回调 ----
  FieldPredictFn field_predict;
  ThinPredictFn thin_predict;

  if (backend_ == InferenceBackend::STUB) {
    field_predict = [H, W](Cell s, Cell g) -> std::vector<float> {
      std::vector<float> f(H * W);
      for (int r = 0; r < H; r++) for (int c = 0; c < W; c++)
        f[r * W + c] = static_cast<float>(std::abs(r - g.first) + std::abs(c - g.second));
      float mx = *std::max_element(f.begin(), f.end());
      if (mx > 1e-8f) for (auto & v : f) v /= mx;
      return f;
    };
    thin_predict = [H, W](Cell, Cell) -> std::vector<float> { return std::vector<float>(H * W, 0.0f); };
  }
#ifdef USE_BPU
  else if (backend_ == InferenceBackend::BPU && field_bpu_) {
    auto compute_pe = [MS](Cell center) -> std::vector<float> {
      float sigma = MS / 5.0f; float sigma_sq = sigma * sigma;
      float alpha = 1.0f / (2.0f * M_PI * sigma_sq);
      std::vector<float> pe(MS * MS); float mx = 0;
      for (int r = 0; r < MS; r++) for (int c = 0; c < MS; c++) {
        float dr = r - center.first, dc = c - center.second;
        float v = alpha * std::exp(-(dr * dr + dc * dc) / (2.0f * sigma_sq));
        pe[r * MS + c] = v; if (v > mx) mx = v;
      }
      if (mx > 1e-8f) for (auto & v : pe) v /= mx;
      return pe;
    };

    field_predict = [this, MS, H, W, &obs_bpu, compute_pe, full_res, pad_top, pad_left, uniform_scale](Cell s, Cell g) -> std::vector<float> {
      if (!field_bpu_) return {};
      // costmap坐标 → padded坐标 → 256坐标 (uniform scale, 无扭曲)
      Cell ms_s(static_cast<int>((s.first + pad_top) * uniform_scale),
                static_cast<int>((s.second + pad_left) * uniform_scale));
      Cell ms_g(static_cast<int>((g.first + pad_top) * uniform_scale),
                static_cast<int>((g.second + pad_left) * uniform_scale));
      auto pe_s = compute_pe(ms_s); auto pe_g = compute_pe(ms_g);
      std::vector<float> inp(MS * MS * 3);
      for (int i = 0; i < MS * MS; i++) {
        inp[i] = obs_bpu[i]; inp[MS * MS + i] = pe_s[i]; inp[2 * MS * MS + i] = pe_g[i];
      }
      auto field_small = field_bpu_->forward(inp, MS, MS);
      if (full_res) return bilinearResize(field_small, MS, MS, H, W);
      return field_small;  // downsample模式直接返回256x256
    };

    thin_predict = [this, MS, H, W, &obs_bpu, compute_pe, full_res, pad_top, pad_left, uniform_scale](Cell s, Cell g) -> std::vector<float> {
      if (!thin_bpu_) return std::vector<float>(H * W, 0.0f);
      Cell ms_s(static_cast<int>((s.first + pad_top) * uniform_scale),
                static_cast<int>((s.second + pad_left) * uniform_scale));
      Cell ms_g(static_cast<int>((g.first + pad_top) * uniform_scale),
                static_cast<int>((g.second + pad_left) * uniform_scale));
      auto pe_s = compute_pe(ms_s); auto pe_g = compute_pe(ms_g);
      std::vector<float> inp(MS * MS * 3);
      for (int i = 0; i < MS * MS; i++) {
        inp[i] = obs_bpu[i]; inp[MS * MS + i] = pe_s[i]; inp[2 * MS * MS + i] = pe_g[i];
      }
      auto score_small = thin_bpu_->forward(inp, MS, MS);
      if (full_res) return bilinearResize(score_small, MS, MS, H, W);
      return score_small;
    };
  }
#endif

  // ---- 走廊裁剪 (优化2): coarse 256x256 gw → corridor mask → restrict search ----
  std::vector<float> field_small_corridor;
  if (full_res && corridor_width_ > 0) {
    // 走廊的256x256坐标也用padded+uniform_scale (和obs_bpu一致)
    if (backend_ == InferenceBackend::STUB) {
      Cell g256 = {static_cast<int>((my_g + pad_top) * uniform_scale),
                   static_cast<int>((mx_g + pad_left) * uniform_scale)};
      field_small_corridor.resize(MS * MS);
      for (int r = 0; r < MS; r++)
        for (int c = 0; c < MS; c++)
          field_small_corridor[r * MS + c] = static_cast<float>(std::abs(r - g256.first) + std::abs(c - g256.second));
      float mx = *std::max_element(field_small_corridor.begin(), field_small_corridor.end());
      if (mx > 1e-8f) for (auto & v : field_small_corridor) v /= mx;
    }
#ifdef USE_BPU
    else if (backend_ == InferenceBackend::BPU && field_bpu_) {
      float sigma = MS / 5.0f, sigma_sq = sigma * sigma, alpha = 1.0f / (2.0f * M_PI * sigma_sq);
      Cell s256(static_cast<int>((my_s + pad_top) * uniform_scale),
                static_cast<int>((mx_s + pad_left) * uniform_scale));
      Cell g256(static_cast<int>((my_g + pad_top) * uniform_scale),
                static_cast<int>((mx_g + pad_left) * uniform_scale));
      std::vector<float> pe_s(MS * MS), pe_g(MS * MS), inp(MS * MS * 3);
      for (int r = 0; r < MS; r++) for (int c = 0; c < MS; c++) {
        float dr_s = r - s256.first, dc_s = c - s256.second;
        float v_s = alpha * std::exp(-(dr_s * dr_s + dc_s * dc_s) / (2.0f * sigma_sq));
        pe_s[r * MS + c] = v_s;
        float dr_g = r - g256.first, dc_g = c - g256.second;
        float v_g = alpha * std::exp(-(dr_g * dr_g + dc_g * dc_g) / (2.0f * sigma_sq));
        pe_g[r * MS + c] = v_g;
      }
      float mx_s = *std::max_element(pe_s.begin(), pe_s.end());
      float mx_g = *std::max_element(pe_g.begin(), pe_g.end());
      if (mx_s > 1e-8f) for (auto & v : pe_s) v /= mx_s;
      if (mx_g > 1e-8f) for (auto & v : pe_g) v /= mx_g;
      for (int i = 0; i < MS * MS; i++) {
        inp[i] = obs_bpu[i];
        inp[MS * MS + i] = pe_s[i];
        inp[2 * MS * MS + i] = pe_g[i];
      }
      field_small_corridor = field_bpu_->forward(inp, MS, MS);
    }
#endif
  }

  // 用走廊裁剪的 obs_search 替换原始 obs
  std::vector<uint8_t> obs_search = obs;
  if (!field_small_corridor.empty()) {
    Cell start_256 = {static_cast<int>((my_s + pad_top) * uniform_scale), static_cast<int>((mx_s + pad_left) * uniform_scale)};
    Cell goal_256 = {static_cast<int>((my_g + pad_top) * uniform_scale), static_cast<int>((mx_g + pad_left) * uniform_scale)};
    Path coarse = gradientWalk(field_small_corridor, obs_bpu, MS, MS, start_256, goal_256);
    if (!coarse.empty()) {
      int cw = corridor_width_;
      std::vector<uint8_t> corridor_mask(H * W, 0);
      for (const auto & [r256, c256] : coarse) {
        // 256坐标 → padded坐标 → costmap坐标(减偏移)
        int rf = static_cast<int>(r256 / uniform_scale) - pad_top;
        int cf = static_cast<int>(c256 / uniform_scale) - pad_left;
        for (int dr = -cw; dr <= cw; dr++)
          for (int dc = -cw; dc <= cw; dc++) {
            int nr = rf + dr, nc = cf + dc;
            if (nr >= 0 && nr < H && nc >= 0 && nc < W)
              corridor_mask[nr * W + nc] = 1;
           }
       }
       // 确保起点和终点在走廊内(粗搜终点在256x256空间靠近goal,
       // 但映射回全分辨率可能有偏移, 导致goal被挡)
       for (int dr = -3; dr <= 3; dr++)
         for (int dc = -3; dc <= 3; dc++) {
           int nsr = start_cell.first + dr, nsc = start_cell.second + dc;
           if (nsr >= 0 && nsr < H && nsc >= 0 && nsc < W)
             corridor_mask[nsr * W + nsc] = 1;
           int ngr = goal_cell.first + dr, ngc = goal_cell.second + dc;
           if (ngr >= 0 && ngr < H && ngc >= 0 && ngc < W)
             corridor_mask[ngr * W + ngc] = 1;
         }
       for (int i = 0; i < H * W; i++)
        if (!corridor_mask[i]) obs_search[i] = 1;
      RCLCPP_DEBUG(logger_, "[NeuralPathPlanner] corridor: coarse %zu pts, masked", coarse.size());
    }
  }

  // 走廊模式下复用第一次BPU输出, 省掉第二次BPU推理(5ms)
  FieldPredictFn field_predict_final = field_predict;
  if (!field_small_corridor.empty()) {
    field_predict_final = [&field_small_corridor, H, W, MS](Cell, Cell) -> std::vector<float> {
      return bilinearResize(field_small_corridor, MS, MS, H, W);
    };
  }

  // ---- 搜索 ----
  std::string method;
  Path grid_path = planRigid(obs_search, H, W, start_cell, goal_cell,
                             field_predict_final, thin_predict, method,
                             cost_norm, static_cast<float>(w_cost_));

  last_inference_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();

  if (grid_path.empty()) {
    return makeStraightLine(start, goal);
  }

  // 捷径剪枝: 削掉凹槽绕路 (无clearance, 恢复效果最好的版本)
  auto before_prune = grid_path.size();
  grid_path = prunePath(grid_path, obs, W);
  if (grid_path.size() < before_prune) {
    RCLCPP_DEBUG(logger_, "[NeuralPathPlanner] prune: %zu → %zu poses",
      before_prune, grid_path.size());
  }

  // ---- 路径转世界坐标 ----
  for (const auto & [r, c] : grid_path) {
    unsigned int cm_r, cm_c;
    if (full_res) {
      cm_r = r; cm_c = c;
    } else {
      cm_r = static_cast<unsigned int>(r / sr);
      cm_c = static_cast<unsigned int>(c / sc);
    }
    cm_r = std::min(cm_r, static_cast<unsigned int>(costmap_H - 1));
    cm_c = std::min(cm_c, static_cast<unsigned int>(costmap_W - 1));
    double wx, wy;
    mapToWorld(cm_c, cm_r, wx, wy);
    geometry_msgs::msg::PoseStamped p = start;
    p.header = path.header;
    p.pose.position.x = wx; p.pose.position.y = wy; p.pose.position.z = 0.0;
    path.poses.push_back(p);
  }

  RCLCPP_INFO(logger_, "[NeuralPathPlanner] #%lu: %zu poses, %.1fms (%s) [%s]",
    static_cast<unsigned long>(plan_count_), path.poses.size(), last_inference_ms_,
    method.c_str(), search_mode_.c_str());

  return path;
}

// 直线回退
nav_msgs::msg::Path NeuralPathPlanner::makeStraightLine(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal) const
{
  nav_msgs::msg::Path path;
  path.header.stamp = clock_ ? clock_->now() : rclcpp::Clock().now();
  path.header.frame_id = global_frame_;
  double dx = goal.pose.position.x - start.pose.position.x;
  double dy = goal.pose.position.y - start.pose.position.y;
  double dist = std::hypot(dx, dy);
  auto push = [&](double wx, double wy) {
    geometry_msgs::msg::PoseStamped p = start;
    p.header = path.header; p.pose.position.x = wx; p.pose.position.y = wy; p.pose.position.z = 0.0;
    path.poses.push_back(p);
  };
  push(start.pose.position.x, start.pose.position.y);
  if (dist > 1e-6) {
    unsigned int steps = std::max(1u, static_cast<unsigned int>(dist / 0.05));
    for (unsigned int i = 1; i <= steps; ++i) {
      double t = static_cast<double>(i) / steps;
      push(start.pose.position.x + dx * t, start.pose.position.y + dy * t);
    }
  }
  return path;
}

bool NeuralPathPlanner::worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const
{ return costmap_ ? costmap_->worldToMap(wx, wy, mx, my) : false; }

void NeuralPathPlanner::mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const
{ if (costmap_) costmap_->mapToWorld(mx, my, wx, wy); }

bool NeuralPathPlanner::inCollision(unsigned int mx, unsigned int my) const
{ return costmap_ ? costmap_->getCost(mx, my) >= collision_cost_threshold_ : false; }

}  // namespace neural_path_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(neural_path_planner::NeuralPathPlanner, nav2_core::GlobalPlanner)
