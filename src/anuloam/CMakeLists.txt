cmake_minimum_required(VERSION 3.8)
project(anuloam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(imu_preintegration src/imu_preintegration.cpp)
ament_target_dependencies(imu_preintegration rclcpp sensor_msgs)
target_link_libraries(imu_preintegration gtsam)

# Explicitly control RPATH so conda's lib directory is never embedded in the
# installed binary. Without this, CMake auto-appends any transitive dependency
# path (e.g. miniconda3/lib via spdlog/fmt) to RUNPATH, which makes the
# dynamic linker pick up conda's old libstdc++ instead of the system one,
# causing GLIBCXX version-not-found errors at runtime.
set_target_properties(imu_preintegration PROPERTIES
  INSTALL_RPATH "/opt/ros/humble/lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS imu_preintegration
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
