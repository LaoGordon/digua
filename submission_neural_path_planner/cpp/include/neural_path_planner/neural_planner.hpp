// Copyright 2026 longkang
//
// BPU-accelerated neural global path planner for Nav2.
//
// Design (see docs/ARCHITECTURE.md):
//   里程碑0: 退化模式  —— model 未加载时直接返回起终点之间的直线，证明插件能被 Nav2 加载。
//   里程碑1: ONNX 推理 —— 加载 U-Net guidance map 模型，在 CPU (ONNX Runtime) 上跑通"地图+起终点→路径掩码"。
//   里程碑2: BPU 推理  —— 同一份模型经 hb_mapper 编译为 .bin 后，走 Horizon BPU runtime (libdnn / hbmvp)。
//                        接口保持一致，只换底层推理后端。
//
// 三个后端共用同一个 createPlan() 流程：把全局 costmap 当作输入栅格，
// 模型输出二值路径掩码 → 细化 → 世界坐标路径 → CPU 碰撞兜底。
// 模型不可用时一律回退直线，保证 Nav2 不会因为插件崩溃而瘫掉。

#ifndef NEURAL_PATH_PLANNER__NEURAL_PLANNER_HPP_
#define NEURAL_PATH_PLANNER__NEURAL_PLANNER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "nav2_core/global_planner.hpp"
#include "rclcpp/rclcpp.hpp"

#ifdef USE_BPU
#include "neural_path_planner/bpu_inference.hpp"
#endif

namespace neural_path_planner
{

// 推理后端类型。configure() 阶段选定，运行时不变。
enum class InferenceBackend
{
  STUB = 0,    // PC验证用: 几何势场stub(不加载模型)
  ONNX = 1,    // ONNX Runtime CPU(阶段1)
  BPU = 2,     // Horizon BPU runtime(阶段2, 仅RDK S100P)
};

/**
 * @brief 神经网络全局路径规划器（Nav2 GlobalPlanner 插件）。
 */
class NeuralPathPlanner : public nav2_core::GlobalPlanner
{
public:
  NeuralPathPlanner();
  ~NeuralPathPlanner() override;

  // ---- nav2_core::GlobalPlanner 接口 ----

  /// Nav2 planner_server 在 activate 阶段调用一次，读参数、加载模型。
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  /// 每次规划请求时调用。返回世界坐标系下的路径。
  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  // ---- 工具方法 ----

  /// 把世界坐标点转成 costmap 栅格索引。越界返回 false。
  bool worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const;

  /// 把 costmap 栅格转成世界坐标。
  void mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const;

  /**
   * @brief 里程碑0：直线回退。
   * 在 start 和 goal 之间等距插值，每点做碰撞检查。被穿透的点丢弃，
   * 但只要起终点本身合法就返回一条（哪怕是退化的）路径，保证 Nav2 不抛异常。
   */
  nav_msgs::msg::Path makeStraightLine(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) const;

  /// 单点碰撞检查：在 costmap 上取该格 cost，>= 阈值视为碰撞。
  bool inCollision(unsigned int mx, unsigned int my) const;

  // ---- 状态 ----

  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  rclcpp::Logger logger_{rclcpp::get_logger("NeuralPathPlanner")};
  rclcpp::Clock::SharedPtr clock_;
  std::string name_;

  // 参数（见 config/neural_planner_params.yaml）
  InferenceBackend backend_{InferenceBackend::STUB};
  std::string field_model_path_;    // 势场模型路径(ONNX或.hbm)
  std::string thin_model_path_;     // 细线模型路径
  double collision_cost_threshold_{253.0};
  double straight_line_resolution_{0.05};
  double goal_tolerance_{0.5};
  bool visualize_{false};
  std::string global_frame_;
  std::string strategy_{std::string("rigid")};
  std::string search_mode_{std::string("full")};  // "full" or "downsample" (消融用)
  int corridor_width_{0};  // 搜索窗口裁剪宽度(格), 0=禁用
  double w_cost_{0.3};  // cost软项权重: 让搜索主动绕开inflation带, 0=禁用(纯势场)

  // BPU模型(板上编译时启用)
#ifdef USE_BPU
  std::unique_ptr<class BpuInference> field_bpu_;
  std::unique_ptr<class BpuInference> thin_bpu_;
#endif

  // 统计（写日志 + 后续论文实验用）
  uint64_t plan_count_{0};
  double last_inference_ms_{0.0};
};

}  // namespace neural_path_planner

#endif  // NEURAL_PATH_PLANNER__NEURAL_PLANNER_HPP_
