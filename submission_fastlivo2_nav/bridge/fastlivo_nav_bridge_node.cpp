#include <atomic>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

class FastlivoNavBridgeNode : public rclcpp::Node
{
public:
  FastlivoNavBridgeNode()
  : Node("fastlivo_nav_bridge")
  {
    fastlivo_odom_topic_ = declare_parameter<std::string>(
      "fastlivo_odom_topic", "/aft_mapped_to_init");
    fastlivo_cloud_topic_ = declare_parameter<std::string>(
      "fastlivo_cloud_topic", "/cloud_registered_lidar");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    obstacle_topic_ = declare_parameter<std::string>("obstacle_topic", "/obstacle_points");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base");
    publish_identity_map_to_odom_ = declare_parameter<bool>("publish_identity_map_to_odom", true);
    ground_filter_min_z_ = declare_parameter<double>("ground_filter_min_z", 0.2);
    ground_filter_max_z_ = declare_parameter<double>("ground_filter_max_z", 2.0);

    map_to_odom_x_ = declare_parameter<double>("map_to_odom_x", 0.0);
    map_to_odom_y_ = declare_parameter<double>("map_to_odom_y", 0.0);
    map_to_odom_z_ = declare_parameter<double>("map_to_odom_z", 0.0);
    map_to_odom_yaw_ = declare_parameter<double>("map_to_odom_yaw", 0.0);

    double half_yaw = map_to_odom_yaw_ / 2.0;
    map_to_odom_qx_ = 0.0;
    map_to_odom_qy_ = 0.0;
    map_to_odom_qz_ = std::sin(half_yaw);
    map_to_odom_qw_ = std::cos(half_yaw);

    use_icp_localization_ = declare_parameter<bool>("use_icp_localization", false);
    reference_pcd_path_ = declare_parameter<std::string>("reference_pcd_path", "");
    icp_voxel_size_ = declare_parameter<double>("icp_voxel_size", 0.3);
    ref_voxel_size_ = declare_parameter<double>("ref_voxel_size", 0.1);
    icp_max_iterations_ = declare_parameter<int>("icp_max_iterations", 30);
    icp_max_correspondence_distance_ = declare_parameter<double>("icp_max_correspondence_distance", 2.0);
    icp_transformation_epsilon_ = declare_parameter<double>("icp_transformation_epsilon", 1e-6);
    icp_euclidean_fitness_epsilon_ = declare_parameter<double>("icp_euclidean_fitness_epsilon", 0.05);
    icp_max_accumulated_clouds_ = declare_parameter<int>("icp_max_accumulated_clouds", 5);
    icp_hz_ = declare_parameter<double>("icp_hz", 1.0);

    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    obstacle_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(obstacle_topic_, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Timer to keep publishing last known TF even when FAST-LIVO2 stops
    tf_keepalive_timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      [this]() {
        if (!latest_odom_valid_) return;
        rclcpp::Time now = this->now();
        geometry_msgs::msg::TransformStamped odom_to_base;
        odom_to_base.header.stamp = now;
        odom_to_base.header.frame_id = odom_frame_;
        odom_to_base.child_frame_id = base_frame_;
        odom_to_base.transform.translation.x = latest_odom_tx_;
        odom_to_base.transform.translation.y = latest_odom_ty_;
        odom_to_base.transform.translation.z = latest_odom_tz_;
        odom_to_base.transform.rotation.x = latest_odom_qx_;
        odom_to_base.transform.rotation.y = latest_odom_qy_;
        odom_to_base.transform.rotation.z = latest_odom_qz_;
        odom_to_base.transform.rotation.w = latest_odom_qw_;
        tf_broadcaster_->sendTransform(odom_to_base);
        publish_map_to_odom(now);
      });

    fastlivo_odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      fastlivo_odom_topic_, 50,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg)
      {
        handle_fastlivo_odom(*msg);
      });

    fastlivo_cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      fastlivo_cloud_topic_, 10,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
      {
        handle_fastlivo_cloud(*msg);
        if (use_icp_localization_) {
          handle_cloud_for_icp(*msg);
        }
      });

    if (use_icp_localization_) {
      init_icp_localization();
    }

    RCLCPP_INFO(
      get_logger(),
      "fastlivo_nav_bridge started. "
      "input_odom=%s input_cloud=%s odom_topic=%s obstacle_topic=%s "
      "identity=%s icp_2d=%s",
      fastlivo_odom_topic_.c_str(),
      fastlivo_cloud_topic_.c_str(),
      odom_topic_.c_str(),
      obstacle_topic_.c_str(),
      publish_identity_map_to_odom_ ? "true" : "false",
      use_icp_localization_ ? "true" : "false");
  }

private:
  void init_icp_localization()
  {
    if (reference_pcd_path_.empty()) {
      RCLCPP_ERROR(get_logger(), "use_icp_localization=true but reference_pcd_path is empty");
      use_icp_localization_ = false;
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr full_ref(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(reference_pcd_path_, *full_ref) == -1) {
      RCLCPP_ERROR(get_logger(), "Failed to load reference PCD: %s", reference_pcd_path_.c_str());
      use_icp_localization_ = false;
      return;
    }
    RCLCPP_INFO(get_logger(), "Loaded reference PCD: %s (%zu points)",
      reference_pcd_path_.c_str(), full_ref->size());

    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_2d(new pcl::PointCloud<pcl::PointXYZ>);
    ref_2d->reserve(full_ref->size());
    for (const auto & p : full_ref->points) {
      if (std::isfinite(p.x) && std::isfinite(p.y)) {
        ref_2d->push_back(pcl::PointXYZ(p.x, p.y, 0.0f));
      }
    }
    RCLCPP_INFO(get_logger(), "Projected to 2D: %zu points", ref_2d->size());

    pcl::VoxelGrid<pcl::PointXYZ> ref_vg;
    ref_vg.setInputCloud(ref_2d);
    ref_vg.setLeafSize(ref_voxel_size_, ref_voxel_size_, ref_voxel_size_);
    ref_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    ref_vg.filter(*ref_cloud_);
    RCLCPP_INFO(get_logger(), "Reference downsampled: %zu -> %zu points (voxel=%.2fm)",
      ref_2d->size(), ref_cloud_->size(), ref_voxel_size_);

    if (ref_cloud_->size() < 100) {
      RCLCPP_ERROR(get_logger(), "Reference cloud too sparse (%zu points), disabling ICP",
        ref_cloud_->size());
      use_icp_localization_ = false;
      return;
    }

    icp_ = std::make_shared<pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
    icp_->setMaximumIterations(icp_max_iterations_);
    icp_->setMaxCorrespondenceDistance(icp_max_correspondence_distance_);
    icp_->setTransformationEpsilon(icp_transformation_epsilon_);
    icp_->setEuclideanFitnessEpsilon(icp_euclidean_fitness_epsilon_);
    icp_->setInputTarget(ref_cloud_);

    icp_initial_guess_ = Eigen::Matrix4f::Identity();

    RCLCPP_INFO(get_logger(),
      "2D ICP localization ready. ref=%zu pts, voxel=%.2f, max_iter=%d, max_corr=%.1f, hz=%.1f",
      ref_cloud_->size(), icp_voxel_size_, icp_max_iterations_,
      icp_max_correspondence_distance_, icp_hz_);
  }

  void handle_cloud_for_icp(const sensor_msgs::msg::PointCloud2 & msg)
  {
    if (!ref_cloud_ || ref_cloud_->empty()) return;

    rclcpp::Time now = this->now();
    if ((now - last_icp_time_).seconds() < 1.0 / icp_hz_) return;
    last_icp_time_ = now;

    pcl::PointCloud<pcl::PointXYZ>::Ptr live_2d(new pcl::PointCloud<pcl::PointXYZ>);
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (std::isfinite(*iter_x) && std::isfinite(*iter_y)) {
        float z = *iter_z;
        if (z >= ground_filter_min_z_ && z <= ground_filter_max_z_) {
          live_2d->push_back(pcl::PointXYZ(*iter_x, *iter_y, 0.0f));
        }
      }
    }

    if (live_2d->empty()) return;

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(live_2d);
    vg.setLeafSize(icp_voxel_size_, icp_voxel_size_, 1.0f);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*filtered);

    if (filtered->size() < 10) return;

    {
      std::lock_guard<std::mutex> lock(accum_mutex_);
      accumulated_clouds_.push_back(filtered);
      accumulated_stamps_.push_back(msg.header.stamp);
      while (static_cast<int>(accumulated_clouds_.size()) > icp_max_accumulated_clouds_) {
        accumulated_clouds_.pop_front();
        accumulated_stamps_.pop_front();
      }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
    builtin_interfaces::msg::Time cloud_stamp;
    {
      std::lock_guard<std::mutex> lock(accum_mutex_);
      for (const auto & c : accumulated_clouds_) {
        *merged += *c;
      }
      cloud_stamp = accumulated_stamps_.back();
    }

    if (merged->size() < 20) return;

    pcl::PointCloud<pcl::PointXYZ> aligned;
    icp_->setInputSource(merged);
    icp_->align(aligned, icp_initial_guess_);

    if (!icp_->hasConverged()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "2D ICP not converged, skipping");
      return;
    }

    double fitness = icp_->getFitnessScore();
    if (fitness > icp_euclidean_fitness_epsilon_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "2D ICP fitness %.4f > threshold %.4f, skipping",
        fitness, icp_euclidean_fitness_epsilon_);
      return;
    }

    Eigen::Matrix4f T_icp = icp_->getFinalTransformation();
    icp_initial_guess_ = T_icp;

    // ICP matches cloud_in_odom_frame -> ref_in_map_frame
    // So T_icp IS T_map_odom directly: transforms odom-frame point to map frame
    double mx = T_icp(0, 3);
    double my = T_icp(1, 3);
    double m_yaw = std::atan2(T_icp(1, 0), T_icp(0, 0));

    tf2::Quaternion q_map_odom;
    q_map_odom.setRPY(0.0, 0.0, m_yaw);

    icp_m2o_x_.store(mx);
    icp_m2o_y_.store(my);
    icp_m2o_z_.store(0.0);
    icp_m2o_qx_.store(q_map_odom.x());
    icp_m2o_qy_.store(q_map_odom.y());
    icp_m2o_qz_.store(q_map_odom.z());
    icp_m2o_qw_.store(q_map_odom.w());
    icp_valid_.store(true);

    publish_map_to_odom_transform(msg.header.stamp, mx, my, 0.0,
      q_map_odom.x(), q_map_odom.y(), q_map_odom.z(), q_map_odom.w());

    RCLCPP_INFO(get_logger(),
      "2D ICP: fitness=%.4f map->odom=(%.3f,%.3f,yaw=%.3f)",
      fitness, mx, my, m_yaw);
  }

  nav_msgs::msg::Odometry lookup_odom_at(const builtin_interfaces::msg::Time & stamp)
  {
    std::lock_guard<std::mutex> lock(odom_buffer_mutex_);
    rclcpp::Time target(stamp);
    nav_msgs::msg::Odometry best;
    double best_dt = 1e9;
    for (const auto & o : odom_buffer_) {
      double dt = std::abs((rclcpp::Time(o.header.stamp) - target).seconds());
      if (dt < best_dt) {
        best_dt = dt;
        best = o;
      }
    }
    return best;
  }

  void handle_fastlivo_odom(const nav_msgs::msg::Odometry & msg)
  {
    {
      std::lock_guard<std::mutex> lock(odom_buffer_mutex_);
      odom_buffer_.push_back(msg);
      if (odom_buffer_.size() > 100) odom_buffer_.pop_front();
    }

    publish_map_to_odom(msg.header.stamp);

    nav_msgs::msg::Odometry nav_odom = msg;
    nav_odom.header.frame_id = odom_frame_;
    nav_odom.child_frame_id = base_frame_;
    odom_publisher_->publish(nav_odom);

    geometry_msgs::msg::TransformStamped odom_to_base;
    odom_to_base.header.stamp = msg.header.stamp;
    odom_to_base.header.frame_id = odom_frame_;
    odom_to_base.child_frame_id = base_frame_;
    latest_odom_tx_ = msg.pose.pose.position.x;
    latest_odom_ty_ = msg.pose.pose.position.y;
    latest_odom_tz_ = msg.pose.pose.position.z;
    latest_odom_qx_ = msg.pose.pose.orientation.x;
    latest_odom_qy_ = msg.pose.pose.orientation.y;
    latest_odom_qz_ = msg.pose.pose.orientation.z;
    latest_odom_qw_ = msg.pose.pose.orientation.w;
    latest_odom_valid_ = true;

    odom_to_base.transform.translation.x = latest_odom_tx_;
    odom_to_base.transform.translation.y = latest_odom_ty_;
    odom_to_base.transform.translation.z = latest_odom_tz_;
    odom_to_base.transform.rotation = msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(odom_to_base);
  }

  void handle_fastlivo_cloud(const sensor_msgs::msg::PointCloud2 & msg)
  {
    sensor_msgs::msg::PointCloud2 filtered_cloud;
    filtered_cloud.header = msg.header;
    filtered_cloud.header.frame_id = odom_frame_;
    filtered_cloud.fields = msg.fields;
    filtered_cloud.is_bigendian = msg.is_bigendian;
    filtered_cloud.point_step = msg.point_step;
    filtered_cloud.is_dense = msg.is_dense;
    filtered_cloud.height = 1;
    filtered_cloud.width = 0;
    filtered_cloud.data.reserve(msg.data.size());

    sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
    size_t point_step = msg.point_step;
    size_t in_count = msg.width * msg.height;
    for (size_t i = 0; i < in_count; ++i) {
      float z = iter_z[i];
      if (z >= ground_filter_min_z_ && z <= ground_filter_max_z_) {
        size_t src_offset = i * point_step;
        const uint8_t* src = &msg.data[src_offset];
        filtered_cloud.data.insert(filtered_cloud.data.end(), src, src + point_step);
        filtered_cloud.width++;
      }
    }

    filtered_cloud.row_step = filtered_cloud.point_step * filtered_cloud.width;
    obstacle_publisher_->publish(filtered_cloud);
  }

  void publish_map_to_odom(const builtin_interfaces::msg::Time & stamp)
  {
    if (use_icp_localization_ && icp_valid_.load()) {
      publish_map_to_odom_transform(stamp,
        icp_m2o_x_.load(), icp_m2o_y_.load(), icp_m2o_z_.load(),
        icp_m2o_qx_.load(), icp_m2o_qy_.load(), icp_m2o_qz_.load(), icp_m2o_qw_.load());
      return;
    }

    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = stamp;
    map_to_odom.header.frame_id = map_frame_;
    map_to_odom.child_frame_id = odom_frame_;

    if (publish_identity_map_to_odom_) {
      map_to_odom.transform.translation.x = 0.0;
      map_to_odom.transform.translation.y = 0.0;
      map_to_odom.transform.translation.z = 0.0;
      map_to_odom.transform.rotation.w = 1.0;
    } else {
      map_to_odom.transform.translation.x = map_to_odom_x_;
      map_to_odom.transform.translation.y = map_to_odom_y_;
      map_to_odom.transform.translation.z = map_to_odom_z_;
      map_to_odom.transform.rotation.x = map_to_odom_qx_;
      map_to_odom.transform.rotation.y = map_to_odom_qy_;
      map_to_odom.transform.rotation.z = map_to_odom_qz_;
      map_to_odom.transform.rotation.w = map_to_odom_qw_;
    }

    tf_broadcaster_->sendTransform(map_to_odom);
  }

  void publish_map_to_odom_transform(
    const builtin_interfaces::msg::Time & stamp,
    double tx, double ty, double tz,
    double qx, double qy, double qz, double qw)
  {
    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = stamp;
    map_to_odom.header.frame_id = map_frame_;
    map_to_odom.child_frame_id = odom_frame_;
    map_to_odom.transform.translation.x = tx;
    map_to_odom.transform.translation.y = ty;
    map_to_odom.transform.translation.z = tz;
    map_to_odom.transform.rotation.x = qx;
    map_to_odom.transform.rotation.y = qy;
    map_to_odom.transform.rotation.z = qz;
    map_to_odom.transform.rotation.w = qw;
    tf_broadcaster_->sendTransform(map_to_odom);
  }

  std::string fastlivo_odom_topic_;
  std::string fastlivo_cloud_topic_;
  std::string odom_topic_;
  std::string obstacle_topic_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  bool publish_identity_map_to_odom_{true};
  double ground_filter_min_z_{0.2};
  double ground_filter_max_z_{2.0};
  double map_to_odom_x_{0.0};
  double map_to_odom_y_{0.0};
  double map_to_odom_z_{0.0};
  double map_to_odom_yaw_{0.0};
  double map_to_odom_qx_{0.0};
  double map_to_odom_qy_{0.0};
  double map_to_odom_qz_{0.0};
  double map_to_odom_qw_{1.0};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fastlivo_odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr fastlivo_cloud_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr tf_keepalive_timer_;

  std::deque<nav_msgs::msg::Odometry> odom_buffer_;
  std::mutex odom_buffer_mutex_;

  bool use_icp_localization_{false};
  std::string reference_pcd_path_;
  double icp_voxel_size_{0.3};
  double ref_voxel_size_{0.2};
  int icp_max_iterations_{30};
  double icp_max_correspondence_distance_{2.0};
  double icp_transformation_epsilon_{1e-6};
  double icp_euclidean_fitness_epsilon_{0.05};
  int icp_max_accumulated_clouds_{5};
  double icp_hz_{1.0};

  pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_;
  std::shared_ptr<pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> icp_;
  Eigen::Matrix4f icp_initial_guess_{Eigen::Matrix4f::Identity()};
  std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
  std::deque<builtin_interfaces::msg::Time> accumulated_stamps_;
  std::mutex accum_mutex_;
  rclcpp::Time last_icp_time_{0, 0, RCL_ROS_TIME};

  std::atomic<double> icp_m2o_x_{0.0};
  std::atomic<double> icp_m2o_y_{0.0};
  std::atomic<double> icp_m2o_z_{0.0};
  std::atomic<double> icp_m2o_qx_{0.0};
  std::atomic<double> icp_m2o_qy_{0.0};
  std::atomic<double> icp_m2o_qz_{0.0};
  std::atomic<double> icp_m2o_qw_{1.0};
  std::atomic<bool> icp_valid_{false};

  double latest_odom_tx_{0.0};
  double latest_odom_ty_{0.0};
  double latest_odom_tz_{0.0};
  double latest_odom_qx_{0.0};
  double latest_odom_qy_{0.0};
  double latest_odom_qz_{0.0};
  double latest_odom_qw_{1.0};
  bool latest_odom_valid_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FastlivoNavBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
