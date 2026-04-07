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

    /**
     * @brief extract edge and patch features from 2D scans (rings)
     * @param ring 2D Pointcloud with all the points from the same ring arranged in a consecutive order
     * @param neighbours number of neighbours to consider in the set when calculating roughness (|S|, cardinality of set)
     * @param edgeThresh the roughness threshold above which a point is considered an edge
     * @param planarThresh the roughness threshold below which a point is considered a patch
     * @param totalFeatures the number of total features to extract from a 2D scan
     * @todo: Multithreading
     */
    void extract2DFeatures(const pcl::PointCloud<PointXYZIR>& ring) {
        
        // TODO: could be multi-threaded since each thread only has read access and needs to store the roughness
        size_t scanSize = ring.size();
        size_t currentFeatures = 0;

        for (size_t ind=0; ind<scanSize; ind++) {
            // calculate roughness of point
            int setStart = -(neighbours / 2);
            int setEnd =  (neighbours / 2);

            float sumX = 0, sumY = 0, sumZ = 0;

            for (int j = setStart; j <= setEnd; j++) {
                if (j == 0) continue;
                size_t wrapped = (ind + j + scanSize) % scanSize;
                sumX += ring[ind].x - ring[wrapped].x;
                sumY += ring[ind].y - ring[wrapped].y;
                sumZ += ring[ind].z - ring[wrapped].z;
                // std::printf("Differences are: %f, %f, %f\n", sumX, sumY, sumZ);
            }
            float norm = (ring[ind].x*ring[ind].x + ring[ind].y*ring[ind].y + ring[ind].z*ring[ind].z);
            float roughness = (sumX * sumX + sumY * sumY + sumZ * sumZ) / (neighbours * neighbours * norm);

            // std::printf("Norm is: %f\tRoughness is: %f\n", norm, roughness);

            // if above a threshold (i.e. edge), then add to edge container
            if ( roughness > edgeThresh) {
                _edges.push_back(ring[ind]);
                currentFeatures++;
            }

            // if below a certain threshold (i.e. patch), then add to the patch container
            else if ((roughness < planarThresh) && (currentFeatures < totalFeatures)) {
                _patches.push_back(ring[ind]);
                currentFeatures++;
            }
        }

        // TODO: Non Maximal Suppression equivalent -> get an even spread of features

        // TODO: Rejecting edge cases mentioned in the LOAM Paper (https://frc.ri.cmu.edu/~zhangji/publications/RSS_2014.pdf)
    }

    /**
     * @brief extract edge and patch features from the 3D scan by stacking 2D features (extract2DFeatures)
     * @todo: Multithreading
     */
    void extract3DFeatures() {
        this->extractRings(_pcl);
        for (auto& ring : this->_rings) {
            this->extract2DFeatures(ring);
        }
    }

    // TODO: How to protect against getRings() called when rings is empty?
    // TODO: Don't need to store R in the point if I already have them in rings
    std::vector<pcl::PointCloud<PointXYZIR>> getRings() { return _rings; }
    pcl::PointCloud<PointXYZIR> getPCL() {return _pcl;}
    pcl::PointCloud<PointXYZIR> getFeatures() {return _edges + _patches;}

private:
    pcl::PointCloud<PointXYZIR> _pcl;
    std::vector<pcl::PointCloud<PointXYZIR>> _rings;
    pcl::PointCloud<PointXYZIR> _edges;
    pcl::PointCloud<PointXYZIR> _patches;
    pcl::PointCloud<PointXYZIR> _features;
    int neighbours = 10; 
    float edgeThresh = 0.25; 
    float planarThresh = 1e-4; 
    int totalFeatures = 25;
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
    // pcl::Indices indices(200);
    // std::iota(indices.begin(), indices.end(), 0);

    
    LF.extract3DFeatures();
    pcl::PointCloud<PointXYZIR> features = LF.getFeatures();

    // for (size_t i = 0; i < subRing->points.size(); i++) {
    //     float azimuth = std::atan2(subRing->points[i].y, subRing->points[i].x) * 180.0 / M_PI;
    //     RCLCPP_INFO(this->get_logger(), "Point %zu: azimuth = %.2f°", i, azimuth);
    // }

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(features, output);
    output.header = msg->header;
    pub_->publish(output);

    // Convert PCL -> ROS message and publish
    sensor_msgs::msg::PointCloud2 outputTest;
    pcl::toROSMsg(rings[0], outputTest);
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
