cmake_minimum_required(VERSION 3.8)
project(anuloam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C++23
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(imu_preintegration src/imu_preintegration.cpp)
ament_target_dependencies(imu_preintegration rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros)
target_link_libraries(imu_preintegration gtsam)
add_executable(feature_extraction src/featureExtraction.cpp)
ament_target_dependencies(feature_extraction rclcpp sensor_msgs nav_msgs pcl_ros pcl_conversions)
target_link_libraries(feature_extraction 
    ${PCL_LIBRARIES} 
    OpenMP::OpenMP_CXX
)

# Explicitly control RPATH so conda's lib directory is never embedded in the
# installed binary. Without this, CMake auto-appends any transitive dependency
# path (e.g. miniconda3/lib via spdlog/fmt) to RUNPATH, which makes the
# dynamic linker pick up conda's old libstdc++ instead of the system one,
# causing GLIBCXX version-not-found errors at runtime.
foreach(target imu_preintegration feature_extraction)
  set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "/opt/ros/jazzy/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endforeach()

install(
  TARGETS feature_extraction imu_preintegration
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
