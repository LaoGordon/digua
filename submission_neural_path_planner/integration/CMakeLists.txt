cmake_minimum_required(VERSION 3.8)
project(neural_path_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

# Optional: OpenCV (used by cv_bridge-free image copy in the inference path).
# If unavailable, the planner still builds; model I/O is done with raw buffers.
find_package(OpenCV QUIET)

# ONNX Runtime (PC上用CPU推理, 板上切换BPU)
option(USE_BPU "编译BPU推理后端(仅板子上)" OFF)

add_library(${PROJECT_NAME} SHARED
  src/neural_planner.cpp
  src/planner_core.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# BPU后端(板子上编译时用 -DUSE_BPU=ON)
if(USE_BPU)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_BPU=1)
  target_include_directories(${PROJECT_NAME} PRIVATE /usr/hobot/include /usr/include/hobot)
  find_library(DNN_LIB dnn PATHS /usr/hobot/lib)
  target_link_libraries(${PROJECT_NAME} ${DNN_LIB})
endif()

target_link_libraries(${PROJECT_NAME}
  ${rclcpp_LIBRARIES}
  ${rclcpp_lifecycle_LIBRARIES}
  ${pluginlib_LIBRARIES}
  ${nav2_core_LIBRARIES}
  ${nav2_costmap_2d_LIBRARIES}
  ${nav2_util_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${nav_msgs_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
)

if(OpenCV_FOUND)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  target_compile_definitions(${PROJECT_NAME} PRIVATE NEURAL_PLANNER_HAVE_OPENCV=1)
endif()

# Mark as a pluginlib plugin class so it can be loaded via plugin_description.
pluginlib_export_plugin_description_file(nav2_core
  plugins/neural_path_planner_plugin.xml)

ament_export_dependencies(rclcpp rclcpp_lifecycle pluginlib nav2_core nav2_costmap_2d)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  DIRECTORY plugins config launch scripts docs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
