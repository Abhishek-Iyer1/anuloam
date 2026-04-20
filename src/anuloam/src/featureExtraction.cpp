#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Geometry>
#include <vector>
#include "utils.hpp"

#define NUM_RINGS 16

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

// Add these to see the template implementations for custom types
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>

struct PointRoughness {
    size_t index;
    float roughness;
};


/**
 *  @todo: Need to support each point in the scan having a timestamp parameter
 */
class LidarFrame {
public:
    LidarFrame() = default;
    LidarFrame(const pcl::PointCloud<PointXYZIR>& cloud) : _pcl (cloud) {};
    LidarFrame(const sensor_msgs::msg::PointCloud2& msg) { pcl::fromROSMsg(msg, _pcl); }

    /**
     * @brief extract 2D scans for every ring in the 3D pointcloud scan
     * @param cloud 3D input scan of with ring information
     */
    void extractRings(const pcl::PointCloud<PointXYZIR>& cloud) {

        // extract indices for each point and form populate ```rings```
        std::vector<std::vector<int>> ringIndices(NUM_RINGS, std::vector<int>());
        for (size_t j=0; j<cloud.points.size(); j++) {
            ringIndices[cloud.points[j].ring].push_back(j);
        }
        for (size_t ring=0; ring<NUM_RINGS; ring++) {
            _rings.emplace_back(cloud, pcl::Indices(ringIndices[ring]));
        }
    };

    float computeRoughness(const pcl::PointCloud<PointXYZIR>& ring, size_t ind) {

        float norm = ring[ind].x * ring[ind].x + ring[ind].y * ring[ind].y + ring[ind].z * ring[ind].z;
        if (norm < minRange * minRange || norm > maxRange * maxRange) {
            return -1.0f; // r < 0 filtered out later
        }

        size_t scanSize = ring.size();
        int setStart = -(neighbours / 2);
        int setEnd = (neighbours / 2);

        float sumX = 0, sumY = 0, sumZ = 0;

        for (int j = setStart; j <= setEnd; j++) {
            if (j == 0) continue;
            size_t wrapped = (ind + j + scanSize) % scanSize;
            sumX += ring[ind].x - ring[wrapped].x;
            sumY += ring[ind].y - ring[wrapped].y;
            sumZ += ring[ind].z - ring[wrapped].z;
        }

        return (sumX * sumX + sumY * sumY + sumZ * sumZ) / (neighbours * neighbours * norm);
    }

    /**
     * @brief extract edge and patch features from 2D scans (rings)
     * @param ring 2D Pointcloud with all the points from the same ring arranged in a consecutive order
     * @param neighbours number of neighbours to consider in the set when calculating roughness (|S|, cardinality of set)
     * @param edgeThresh the roughness threshold above which a point is considered an edge
     * @param planarThresh the roughness threshold below which a point is considered a patch
     * @param totalFeatures the number of total features to extract from a 2D scan
     * @todo: Multithreading
     */
    void extract2DFeatures(
        const pcl::PointCloud<PointXYZIR>& ring,
        pcl::PointCloud<PointXYZIR>& edgeFeatures,
        pcl::PointCloud<PointXYZIR>& planarFeatures
    ) {

        // TODO: could be multi-threaded since each thread only has read access and needs to store the roughness
        size_t scanSize = ring.size();
        size_t numSections = 8;
        size_t edgeFeaturesPerSection = 2;
        size_t planarFeaturesPerSection = 4;

        // Add 2 edge and 4 planar features per section to LidarFrames
        for (size_t section=1; section <= numSections; section++) {

            size_t startInd = static_cast<size_t>(static_cast<float>(section - 1) / numSections * scanSize);
            size_t endInd = static_cast<size_t>(static_cast<float>(section) / numSections * scanSize);
            size_t currentFeatures = 0;
            size_t sectionFeatures = static_cast<size_t>(totalFeatures / numSections);

            // std::printf("Current Section: %lu, startInd: %lu, endInd: %lu, sectionFeatures: %lu", section/numSections, startInd, endInd, sectionFeatures);

            std::vector<PointRoughness> roughnesses;
            roughnesses.reserve(endInd - startInd);

            for (size_t ind = startInd; ind < endInd; ind++) {
                float r = computeRoughness(ring, ind);
                if (r < 0) continue;
                roughnesses.push_back({ind, r});
            }

            // append k roughest points (i.e. edges) to _edges
            size_t kEdge = std::min(static_cast<size_t>(edgeFeaturesPerSection), roughnesses.size());

            std::partial_sort(
                roughnesses.begin(),
                roughnesses.begin() + kEdge,
                roughnesses.end(),
                [](const auto& a, const auto& b) { return a.roughness > b.roughness; }
            );

            for (size_t i = 0; i < edgeFeaturesPerSection && i < roughnesses.size(); i++) {
                if (roughnesses[i].roughness > edgeThresh) {
                    edgeFeatures.push_back(ring[roughnesses[i].index]);
                }
            }

            // append k smoothest points (i.e. patches) to _patches
            size_t kPatch = std::min(static_cast<size_t>(planarFeaturesPerSection), roughnesses.size());

            std::partial_sort(
                roughnesses.begin(),
                roughnesses.begin() + kPatch,
                roughnesses.end(),
                [](const auto& a, const auto& b) { return a.roughness < b.roughness; }
            );

            for (size_t i = 0; i < planarFeaturesPerSection && i < roughnesses.size(); i++) {
                if (roughnesses[i].roughness < planarThresh) {
                    planarFeatures.push_back(ring[roughnesses[i].index]);
                }
            }

        }

        // TODO: Rejecting edge cases mentioned in the LOAM Paper (https://frc.ri.cmu.edu/~zhangji/publications/RSS_2014.pdf)
    }

    /**
     * @brief extract edge and patch features from the 3D scan by stacking 2D features (extract2DFeatures)
     * @todo: Multithreading
     */
    void extract3DFeatures() {
        this->extractRings(_pcl);

        // create buckets per ring to containerize the memory accessed by each thread
        std::vector<pcl::PointCloud<PointXYZIR>> edgeBuckets(NUM_RINGS);
        std::vector<pcl::PointCloud<PointXYZIR>> planarBuckets(NUM_RINGS);

        #pragma omp parallel for num_threads(8)
        for (size_t i=0; i<NUM_RINGS; i++) {
            this->extract2DFeatures(this->_rings[i], edgeBuckets[i], planarBuckets[i]);
        }

        // collate returned data back into the edge and patches
        for (size_t i=0; i<NUM_RINGS; i++) {
            this->_edges += edgeBuckets[i];
            this->_patches += planarBuckets[i];
        }

        this->_features = this->_edges + this->_patches;
    }

    pcl::PointCloud<PointXYZIR> transformEdges(const Eigen::Isometry3f& tf) const {
        pcl::PointCloud<PointXYZIR> transformed;
        pcl::transformPointCloud(this->getEdges(), transformed, tf.matrix());
        return transformed;
    }

    pcl::PointCloud<PointXYZIR> transformPatches(const Eigen::Isometry3f& tf) const {
        pcl::PointCloud<PointXYZIR> transformed;
        pcl::transformPointCloud(this->getPatches(), transformed, tf.matrix());
        return transformed;
    }

    // TODO: How to protect against getRings() called when rings is empty?
    // TODO: Don't need to store R in the point if I already have them in rings
    std::vector<pcl::PointCloud<PointXYZIR>> getRings() { return _rings; }
    pcl::PointCloud<PointXYZIR> getPCL() const {return _pcl;}
    pcl::PointCloud<PointXYZIR> getEdges() const {return _edges;}
    pcl::PointCloud<PointXYZIR> getPatches() const {return _patches;}
    const pcl::PointCloud<PointXYZIR>& getFeatures() const { return _features; }

private:
    pcl::PointCloud<PointXYZIR> _pcl;
    std::vector<pcl::PointCloud<PointXYZIR>> _rings;
    pcl::PointCloud<PointXYZIR> _edges;
    pcl::PointCloud<PointXYZIR> _patches;
    pcl::PointCloud<PointXYZIR> _features;
    int neighbours = 16;
    float edgeThresh = 0.25;
    float planarThresh = 1e-4; //1e-4;
    int totalFeatures = 32;
    float minRange = 1.0;
    float maxRange = 20.0;
};

/**
 * @todo: Could make this more efficient by adding to localMap on being received as opposed to on getPointCloud.
 * Challenge is the design of removal of earlier pointclouds
 */
class LocalMap {
public:

    LocalMap(size_t size) :
        _keyframes(size),
        _voxelEdges(new pcl::PointCloud<PointXYZIR>()),
        _voxelPatches(new pcl::PointCloud<PointXYZIR>()),
        _upsampledPCL(new pcl::PointCloud<PointXYZIR>())
    {}

    const CircularBuffer<std::pair<LidarFrame, Eigen::Isometry3f>>& getKeyframes() const { return _keyframes; }

    /**
     * @todo: Currently building the mack from scratch every time keyframe is updated, can we do it incrementally instead?
     */
    void pushKeyframe(const std::pair<LidarFrame, Eigen::Isometry3f>& keyframe) {
        _keyframes.push_back(keyframe);
        updateVoxelMaps();
    }

    void updateVoxelMaps() {
        pcl::PointCloud<PointXYZIR>::Ptr rawEdges(new pcl::PointCloud<PointXYZIR>());
        pcl::PointCloud<PointXYZIR>::Ptr rawPatches(new pcl::PointCloud<PointXYZIR>());

        // Accumulate all features from the buffer window
        for (size_t i = 0; i < _keyframes.size(); ++i) {
            const auto& [frame, tf] = _keyframes[i];
            *rawEdges += frame.transformEdges(tf);
            *rawPatches += frame.transformPatches(tf);
        }

        *_upsampledPCL = *rawEdges;
        *_upsampledPCL += *rawPatches;

        // Downsample Edges (0.2m voxel length)
        pcl::VoxelGrid<PointXYZIR> edgeFilter;
        edgeFilter.setLeafSize(0.2f, 0.2f, 0.2f);
        edgeFilter.setInputCloud(rawEdges);
        edgeFilter.filter(*_voxelEdges);

        // Downsample Patches (0.4m voxel length)
        pcl::VoxelGrid<PointXYZIR> planeFilter;
        planeFilter.setLeafSize(0.4f, 0.4f, 0.4f);
        planeFilter.setInputCloud(rawPatches);
        planeFilter.filter(*_voxelPatches);

        // Build KD-Trees (used later for searching for nearest neighbours)
        if (_voxelEdges->size() > 0) _kdtreeEdges.setInputCloud(_voxelEdges);
        if (_voxelPatches->size() > 0) _kdtreePatches.setInputCloud(_voxelPatches);
    }

    // Find 2 neighbours for an edge point
    bool findEdgeNeighbors(const PointXYZIR& pi, std::vector<int>& indices, std::vector<float>& dists) const {
        if (_voxelEdges->empty()) return false;
        return _kdtreeEdges.nearestKSearch(pi, 2, indices, dists) > 0;
    }

    // Find 3 neighbours for a planar point
    bool findPlaneNeighbors(const PointXYZIR& pi, std::vector<int>& indices, std::vector<float>& dists) const {
        if (_voxelPatches->empty()) return false;
        return _kdtreePatches.nearestKSearch(pi, 3, indices, dists) > 0;
    }

    const pcl::PointCloud<PointXYZIR>& getEdges() const { return *_voxelEdges; }
    const pcl::PointCloud<PointXYZIR>& getPatches() const { return *_voxelPatches; }
    pcl::PointCloud<PointXYZIR> getVoxels() const { return *_voxelEdges + *_voxelPatches; }
    /**
     * @note: ONLY for internal testing
     */
    pcl::PointCloud<PointXYZIR>& getUpsampled() const { return *_upsampledPCL; }

private:
    CircularBuffer<std::pair<LidarFrame, Eigen::Isometry3f>> _keyframes;
    // Voxel Maps
    pcl::PointCloud<PointXYZIR>::Ptr _voxelEdges;
    pcl::PointCloud<PointXYZIR>::Ptr _voxelPatches;
    pcl::PointCloud<PointXYZIR>::Ptr _upsampledPCL;

    // Search Trees
    pcl::KdTreeFLANN<PointXYZIR> _kdtreeEdges;
    pcl::KdTreeFLANN<PointXYZIR> _kdtreePatches;
};

/**
 * @note: Both map and target must be in a common frame
 */
void scanMatching(const LocalMap& map, LidarFrame& target, Eigen::Isometry3f& currentGuess) {

    if (map.getKeyframes().size() == 0) {return;}
    const int maxIterations = 10;
    const float epsilon = 1e-4;
    const float maxDistSq = 1.0;

    for (int iter = 0; iter < maxIterations; ++iter) {

        // Go over the incoming LidarFrame
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Vector<float, 6> g = Eigen::Vector<float, 6>::Zero();

        const pcl::PointCloud<PointXYZIR> transformedEdges = target.transformEdges(currentGuess);
        const pcl::PointCloud<PointXYZIR> transformedPatches = target.transformPatches(currentGuess);

        for (const auto& edge : transformedEdges) {
            Eigen::Vector3f pi = edge.getVector3fMap();
            std::vector<int> indices;
            std::vector<float> dists;
            if (map.findEdgeNeighbors(edge, indices, dists)) {
                if (dists[1] > maxDistSq) continue;

                Eigen::Vector3f pj = (map.getEdges())[indices[0]].getVector3fMap();
                Eigen::Vector3f pl = (map.getEdges())[indices[1]].getVector3fMap();

                // Verification: Line segment must have non-zero length
                if ((pj - pl).norm() < 0.1f) continue;

                // Math: d = |(pi-pj) x (pi-pl)| / |pj-pl|
                // vUnit = (pj - pl) / || pj - pl ||
                // Jacobian = [ [vUnit]_x @ [P_i]_x     -[vUnit]_x ]
                // b_i = [e_0]_x @ vUnit
                Eigen::Vector3f vUnit = (pl - pj) / (pl - pj).norm();
                Eigen::Vector3f bi = (pi - pj).cross(vUnit); // 3x1

                // Initialize a fixed-size 3x6 matrix
                Eigen::Matrix<float, 3, 6> Ai;

                Eigen::Matrix3f vSkew = skew(vUnit);
                Eigen::Matrix3f pSkew = skew(pi);

                // Left side: Rotation (delta_theta) -> [vUnit]_x * [pi]_x
                Ai.leftCols<3>() = vSkew * pSkew;

                // Right side: Translation (delta_t) -> -[vUnit]_x
                Ai.rightCols<3>() = -vSkew;

                H.noalias() += Ai.transpose() * Ai;
                g.noalias() += Ai.transpose() * bi;

            }
        }

        for (const auto& patch : transformedPatches) {
            Eigen::Vector3f pi = patch.getVector3fMap();
            std::vector<int> indices;
            std::vector<float> dists;

            if (map.findPlaneNeighbors(patch, indices, dists)) {
                if (dists[2] > maxDistSq) continue;

                Eigen::Vector3f pu = (map.getPatches())[indices[0]].getVector3fMap();
                Eigen::Vector3f pv = (map.getPatches())[indices[1]].getVector3fMap();
                Eigen::Vector3f pw = (map.getPatches())[indices[2]].getVector3fMap();

                // Verification: Line segment must have non-zero length
                Eigen::Vector3f normal = (pu - pv).cross(pu - pw);
                float normalMag = normal.norm();
                Eigen::Vector3f unitNormal = normal / normalMag;

                if (normalMag < 0.1f) continue;
                // Math: d = |(pu - pv) x (pw - pv) . (pi - pv)| / |(pu - pv) x  (pw - pv)|
                // unit_normal = (pu - pv) x  (pw - pv) / ||(pu - pv) x  (pw - pv)||
                // Jacobian = [(pi x ni).T      unit_normal.T]
                // b_i = unit_normal.T @ (pi - pv)

                float bi = unitNormal.transpose().dot(pi - pv);
                Eigen::Matrix<float, 1, 6> Ai;
                Ai.leftCols<3>() = pi.cross(unitNormal).transpose();
                Ai.rightCols<3>() = unitNormal.transpose();

                H.noalias() += Ai.transpose() * Ai;
                g.noalias() += Ai.transpose() * bi;
            }
        }

        Eigen::Vector<float, 6> deltaTransform = H.ldlt().solve(-g);

        Eigen::Isometry3f nudge = Eigen::Isometry3f::Identity();
        float angle = deltaTransform.head<3>().norm();
        if (angle > 1e-6) {
            Eigen::Vector3f axis = deltaTransform.head<3>() / angle;
            nudge.linear() = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
        } else {
            // Fallback to identity + skew for infinitesimally small angles to avoid div by zero
            nudge.linear() = Eigen::Matrix3f::Identity() + skew(deltaTransform.head<3>());
        }

        nudge.translation() = deltaTransform.tail<3>();
        currentGuess = nudge * currentGuess;

        Eigen::Quaternionf q(currentGuess.linear());
        currentGuess.linear() = q.normalized().toRotationMatrix();

        if (deltaTransform.norm() < epsilon) {
            break;
        }

    }

}

class FeatureExtraction : public rclcpp::Node {
public:
  FeatureExtraction() : Node("feature_extraction"), localMap(50) {
    // TODO: check what QoS profile to follow here
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/points_raw", 10,
        std::bind(&FeatureExtraction::pointCloudCallback, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/points_processed", 10);
    pubLocalMap_ = create_publisher<sensor_msgs::msg::PointCloud2>("/points_local_map", 10);
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
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    scanMatching(localMap, LF, tf);

    // @todo: Add a check for if tf greater than some threshold. If so add it as a keyframe
    localMap.pushKeyframe({LF, tf});

    // for (size_t i = 0; i < subRing->points.size(); i++) {
    //     float azimuth = std::atan2(subRing->points[i].y, subRing->points[i].x) * 180.0 / M_PI;
    //     RCLCPP_INFO(this->get_logger(), "Point %zu: azimuth = %.2f°", i, azimuth);
    // }

    sensor_msgs::msg::PointCloud2 localMapROS;
    pcl::toROSMsg(localMap.getVoxels(), localMapROS);
    localMapROS.header = msg->header;
    pubLocalMap_->publish(localMapROS);

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(features, output);
    output.header = msg->header;
    pub_->publish(output);

    // Convert PCL -> ROS message and publish
    sensor_msgs::msg::PointCloud2 outputTest;
    pcl::toROSMsg(localMap.getUpsampled(), outputTest);
    outputTest.header = msg->header;
    pubTest_->publish(outputTest);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLocalMap_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubTest_;
  LocalMap localMap;
};


int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeatureExtraction>());
  rclcpp::shutdown();
  return 0;
}
