#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
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
    DeskewNode() : Node("deskew"), imu_q_(1000) {
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
        subImu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_correct", 
            100,
            std::bind(&DeskewNode::imuCallback, this, std::placeholders::_1),
            imu_options
        );

        pubPCL_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/deskewed_raw", 10);
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // Pre-align the IMU data to the LiDAR frame immediately upon arrival
        sensor_msgs::msg::Imu converted_msg = imuConverter(*msg);
        
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_q_.push_back(converted_msg);
    }

    void deskewCallback(const sensor_msgs::msg::PointCloud2::SharedPtr pcl) {
        PROFILE_BLOCK("Deskew_Callback");
        double scan_start_time = stamp2sec(pcl->header.stamp);
        double scan_end_time = scan_start_time + 0.1;

        Eigen::Quaternionf delta_q = Eigen::Quaternionf::Identity();
        bool ready_to_deskew = false;

        {
            PROFILE_BLOCK("IMU_LOCK_INTEGRATION");
            std::lock_guard<std::mutex> lock(imu_mutex_);

            if (imu_q_.empty()) return;

            auto get_time = [](const sensor_msgs::msg::Imu& msg) {
                return stamp2sec(msg.header.stamp);
            };

            size_t start_idx = binarySearchClosest(imu_q_, scan_start_time, get_time);
            size_t end_idx = binarySearchClosest(imu_q_, scan_end_time, get_time);

            if (get_time(imu_q_[end_idx]) >= scan_end_time - 0.01) {
                ready_to_deskew = true;
                
                double last_time = get_time(imu_q_[start_idx]);
                
                // Integrate angular velocity from start to end of scan
                for (size_t i = start_idx + 1; i <= end_idx; ++i) {
                    double current_time = get_time(imu_q_[i]);
                    double dt = current_time - last_time;

                    // Because of imuConverter, these axes are already in the Velodyne frame!
                    Eigen::Vector3f aligned_w(
                        imu_q_[i].angular_velocity.x,
                        imu_q_[i].angular_velocity.y,
                        imu_q_[i].angular_velocity.z
                    );

                    // Integrate the rotation step
                    float angle = aligned_w.norm() * dt;
                    if (angle > 1e-6) {
                        Eigen::Quaternionf step_q(Eigen::AngleAxisf(angle, aligned_w.normalized()));
                        delta_q = delta_q * step_q;
                    }
                    
                    last_time = current_time;
                }
                delta_q.normalize(); // Prevent compounding floating point drift
            }
        }

        if (!ready_to_deskew) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Waiting for IMU data to catch up to LiDAR...");
            return; 
        }

        // Force translation to zero per LIO-SAM logic
        Eigen::Vector3f delta_t = Eigen::Vector3f::Zero();

        pcl::PointCloud<PointXYZIRT>::Ptr currentCloud(new pcl::PointCloud<PointXYZIRT>);
        pcl::fromROSMsg(*pcl, *currentCloud);

        for (auto& pt : currentCloud->points) {
            float ratio = pt.time / 0.1f; 
            
            Eigen::Quaternionf q_interp = Eigen::Quaternionf::Identity().slerp(ratio, delta_q);
            Eigen::Vector3f t_interp = delta_t * ratio; // This will just be zero
            
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
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPoints_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPCL_;
    CircularBuffer<sensor_msgs::msg::Imu> imu_q_;
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