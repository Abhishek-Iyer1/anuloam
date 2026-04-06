#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <span>

struct PointXYZIR {
    PCL_ADD_POINT4D;
    float intensity;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint16_t, ring, ring)
)

class LidarFrame {
public:
    LidarFrame(const pcl::PointCloud<PointXYZIR>& cloud) : _pcl (cloud) {};
    LidarFrame(const sensor_msgs::msg::PointCloud2& msg) { pcl::fromROSMsg(msg, _pcl); }
    
    /**
     * @brief extract 2D scans for every ring in the 3D pointcloud scan
     * @param cloud 3D input scan of with ring information
     */
    void extractRings(const pcl::PointCloud<PointXYZIR>& cloud) {

        // extract indices for each point and form populate ```rings```
        std::vector<std::vector<int>> ringIndices(16, std::vector<int>());
        for (size_t j=0; j<cloud.points.size(); j++) {
            ringIndices[cloud.points[j].ring].push_back(j);
        }
        for (size_t ring=0; ring<16; ring++) {
            _rings.emplace_back(cloud, pcl::Indices(ringIndices[ring]));
        }
    };

    // TODO: How to protect against getRings() called when rings is empty?
    std::vector<pcl::PointCloud<PointXYZIR>> getRings() { return _rings; }
    
    pcl::PointCloud<PointXYZIR> getPCL() {return _pcl;}
    

private:
    pcl::PointCloud<PointXYZIR> _pcl;
    std::vector<pcl::PointCloud<PointXYZIR>> _rings;
};

class FeatureExtraction : public rclcpp::Node {
public:
  FeatureExtraction() : Node("feature_extraction") {
    // TODO: check what QoS profile to follow here
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/points_raw", 10,
        std::bind(&FeatureExtraction::pointCloudCallback, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/points_processed", 10);
    pubTest_ = create_publisher<sensor_msgs::msg::PointCloud2>("/points_test", 10);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<PointXYZIR>::Ptr cloud(new pcl::PointCloud<PointXYZIR>);
    pcl::fromROSMsg(*msg, *cloud);

    LidarFrame LF = LidarFrame(*msg);
    pcl::PointCloud<PointXYZIR> subCloud = LF.getPCL();
    LF.extractRings(subCloud);
    std::vector<pcl::PointCloud<PointXYZIR>> rings = LF.getRings();

    // Convert PCL -> ROS message and publish
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(rings[0], output);
    output.header = msg->header;
    pub_->publish(output);

    // Convert PCL -> ROS message and publish
    sensor_msgs::msg::PointCloud2 outputTest;
    pcl::toROSMsg(rings[1], outputTest);
    outputTest.header = msg->header;
    pubTest_->publish(outputTest);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubTest_;
};


int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeatureExtraction>());
  rclcpp::shutdown();
  return 0;
}
