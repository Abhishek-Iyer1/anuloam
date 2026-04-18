#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>



class GlobalMapNode : public rclcpp::Node {
public:
    GlobalMapNode() : Node("global_map") {
        sub_points_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points_raw",
            10,
            std::bind(&GlobalMapNode::pointCloudCallback, this, std::placeholders::_1)
        );
        pub_lidar_odom_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/lidar_odom",
            10
        );
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_lidar_odom_;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 1) Convert ROS -> PCL
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.2f, 0.2f, 0.2f);
        vg.filter(*downsampled);

        // Add logging
        RCLCPP_INFO(this->get_logger(), "Received %zu points", cloud->points.size());

        this->publishLidarOdometry();
    }

    void publishLidarOdometry() {
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = rclcpp::Clock().now();
        this->pub_lidar_odom_->publish(odom_msg);
    }

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalMapNode>());
    rclcpp::shutdown();
    return 0;
}
