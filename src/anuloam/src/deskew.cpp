#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Geometry>
#include "utils.hpp"


struct PointXYZIRT {
    PCL_ADD_POINT4D;  
    float intensity;
    std::uint16_t ring;
    float time;                  
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint16_t, ring, ring)
    (float, time, time)
)

class DeskewNode : public rclcpp::Node {
public:
    DeskewNode() : Node("deskew"), odom_q_(500) {
        imu_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        lidar_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // Subs and pubs
        rclcpp::SubscriptionOptions lidar_options;
        lidar_options.callback_group = lidar_cb_group_;
        subPoints_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points_raw",
            50,
            std::bind(&DeskewNode::deskewCallback, this, std::placeholders::_1),
            lidar_options
        );

        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = imu_cb_group_;
        subImuOdom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom",
            10,
            std::bind(&DeskewNode::imuOdomCallback, this, std::placeholders::_1),
            imu_options
        );

        pubPCL_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/deskewed_raw", 10);
    }

    void imuOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        odom_q_.push_back(*msg);
    }

    void deskewCallback(const sensor_msgs::msg::PointCloud2::SharedPtr pcl) {
        PROFILE_BLOCK("Deskew_Callback");
        // RCLCPP_WARN(this->get_logger(), "Entered Deskew!");
        double scan_start_time = stamp2sec(pcl->header.stamp);
        double scan_end_time = scan_start_time + 0.1;

        Eigen::Isometry3f transform_start = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f transform_end = Eigen::Isometry3f::Identity();
        bool ready_to_deskew = false;

        // Scoped to return mutex as soon as possible
        {
            PROFILE_BLOCK("IMU_LOCK_TRANSFORMS");
            std::lock_guard<std::mutex> lock(imu_mutex_);

            if (odom_q_.empty()) return;
            
            auto get_time = [](const nav_msgs::msg::Odometry& msg) {
                return stamp2sec(msg.header.stamp);
            };

            size_t start_idx = binarySearchClosest(odom_q_, scan_start_time, get_time);
            size_t end_idx = binarySearchClosest(odom_q_, scan_end_time, get_time);

            if (get_time(odom_q_[end_idx]) >= scan_end_time - 0.01) {
                transform_start = odomMsgToEigen(odom_q_[start_idx]);
                transform_end = odomMsgToEigen(odom_q_[end_idx]);
                ready_to_deskew = true;
            }
        }

        if (!ready_to_deskew) {
            RCLCPP_WARN(this->get_logger(), "Waiting for IMU data to catch up to LiDAR...");
            return; 
        }

        Eigen::Isometry3f delta_transform = transform_start.inverse() * transform_end;
        Eigen::Quaternionf delta_q(delta_transform.rotation());
        Eigen::Vector3f delta_t = delta_transform.translation();

        // Convert ROS -> PCL
        pcl::PointCloud<PointXYZIRT>::Ptr currentCloud(new pcl::PointCloud<PointXYZIRT>);
        pcl::fromROSMsg(*pcl, *currentCloud);

        for (auto& pt : currentCloud->points) {
            float ratio = pt.time / 0.1f; 
            
            Eigen::Quaternionf q_interp = Eigen::Quaternionf::Identity().slerp(ratio, delta_q);
            Eigen::Vector3f t_interp = delta_t * ratio;
            
            Eigen::Vector3f p_raw(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_deskewed = q_interp * p_raw + t_interp;
            
            pt.x = p_deskewed.x();
            pt.y = p_deskewed.y();
            pt.z = p_deskewed.z();
        }

        sensor_msgs::msg::PointCloud2 deskewed_msg;
        pcl::toROSMsg(*currentCloud, deskewed_msg);
        
        // Preserve the original header (timestamp and frame_id) so TF knows where this cloud belongs
        deskewed_msg.header = pcl->header; 
        
        pubPCL_->publish(deskewed_msg);
    }

private:
    rclcpp::CallbackGroup::SharedPtr imu_cb_group_;
    rclcpp::CallbackGroup::SharedPtr lidar_cb_group_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdom_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPoints_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPCL_;
    CircularBuffer<nav_msgs::msg::Odometry> odom_q_;
    std::mutex imu_mutex_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DeskewNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}