#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/impl/pcl_base.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Geometry>

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


namespace {

// gtsam::Pose3 uses double; scan matching uses float transforms
gtsam::Pose3 eigenIsometryToPose3(const Eigen::Isometry3f& T) {
    return gtsam::Pose3(
        gtsam::Rot3(T.rotation().cast<double>()),
        gtsam::Point3(T.translation().cast<double>())
    );
}

Eigen::Isometry3f pose3ToEigenIsometry(const gtsam::Pose3& pose) {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.linear() = pose.rotation().matrix().cast<float>();
    T.translation() = pose.translation().cast<float>();
    return T;
}

}  // namespace

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
            size_t wrapped = static_cast<size_t>((static_cast<int>(ind) + j + static_cast<int>(scanSize)) % static_cast<int>(scanSize));
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
            // size_t currentFeatures = 0;
            // size_t sectionFeatures = static_cast<size_t>(totalFeatures / numSections);

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



class GlobalMapNode : public rclcpp::Node {
public:
    GlobalMapNode() : Node("global_map"), localMap(50){
        // Subs and pubs
        subPoints_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points_raw",
            10,
            std::bind(&GlobalMapNode::pointCloudCallback, this, std::placeholders::_1)
        );
        subImuOdom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom_imu",
            10,
            std::bind(&GlobalMapNode::imuOdomCallback, this, std::placeholders::_1)
        );
        pubIncrementalOdom_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/odom_incremental",
            10
        );
        pubGlobalOdom_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/odom_global",
            10
        );


        // Noise models (Covariances)
        gtsam::Vector6 prior_sigmas;
        prior_sigmas << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2; // @todo: Update these values
        prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);

        gtsam::Vector6 odom_sigmas;
        odom_sigmas << 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1; // @todo: Update these values
        odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);

        // Initialize the ISAM2 Optimizer
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam_ = std::make_unique<gtsam::ISAM2>(parameters);
    }

private:
    // Subs and pubs
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPoints_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubIncrementalOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdom_;

    // LiDAR processing
    LocalMap localMap;
    const float keyframeTranslationThresh_ = 0.3f;   // 30 cm
    const float keyframeRotationThresh_ = 0.2618f;   // ~15 degrees in radians

    // Factor graph variables
    int keyFrameID = 0;
    gtsam::Pose3 currentPoseEstimate = gtsam::Pose3::Identity();

    // Noise models (Covariances)
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;

    // GTSAM backend variables
    std::unique_ptr<gtsam::ISAM2> isam_;
    gtsam::NonlinearFactorGraph gtSAMgraph_;
    gtsam::Values initialEstimate_;

    // Keyframe variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudKeyPoses3D_{new pcl::PointCloud<pcl::PointXYZ>()};
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudKeyframes_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeHistoryKeyPoses_{new pcl::KdTreeFLANN<pcl::PointXYZ>()};

    // Loop Closure Parameters
    double historySearchRadius_ = 15.0; // Meters
    int historySearchTimeDiff_ = 30;    // Minimum frames between current and history
    double icpFitnessThreshold_ = 0.3;  // Lower is better

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 1) Convert ROS -> PCL
        pcl::PointCloud<pcl::PointXYZI>::Ptr currentCloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *currentCloud);

        LidarFrame LF = LidarFrame(*msg);
        LF.extract3DFeatures();

        // @todo: Perform Scan Matching
        Eigen::Isometry3f currentTf = Eigen::Isometry3f::Identity();
        scanMatching(localMap, LF, currentTf);

        // TODO: Get relative pose. Scan matching gives pose wrt first frame
        // Get the last stored LidarFrame and tf pair from the local map circular buffer
        Eigen::Isometry3f relative_pose = currentTf;
        bool isKeyFrame = false;
        const auto& keyframes = localMap.getKeyframes();
        if (!keyframes.empty()) {
            const auto& [lastLidarFrame, lastTf] = keyframes.back();
            relative_pose = lastTf.inverse() * currentTf;
            Eigen::AngleAxisf aa(relative_pose.rotation());
            isKeyFrame = (relative_pose.translation().norm() > keyframeTranslationThresh_ || aa.angle() > keyframeRotationThresh_);
            if (!isKeyFrame) {
                return;
            }
        }

        // TODO: Publish whatever scan matching gives as the continuous odometry
        publishIncrementalOdometry(currentTf, msg->header.stamp);

        // @todo: Update Factor Graph with the pose
        updateGraph(eigenIsometryToPose3(relative_pose));

        // @todo: Detect Loop Closure
        detectLoopClosure(currentCloud);

        // @todo: Optimize the factor graph
        isam_->update(gtSAMgraph_, initialEstimate_);
        isam_->update(); // Often called twice to ensure convergence

        // Clear graph and estimates for next iteration
        gtSAMgraph_.resize(0);
        initialEstimate_.clear();

        // Publish global odometry
        gtsam::Values currentEstimate = isam_->calculateEstimate();
        Eigen::Isometry3f CurrentGlobalTf = pose3ToEigenIsometry(currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('X', keyFrameID - 1)));
        publishGlobalOdometry(CurrentGlobalTf, msg->header.stamp);

        // Add logging
        RCLCPP_INFO(this->get_logger(), "Received %zu points", currentCloud->points.size());

        // Add the current keyframe to the local map
        localMap.pushKeyframe({LF, currentTf});
    }

    void imuOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // @todo: Implement IMU-based odometry update
    }

    gtsam::Pose3 performScanMatching(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        // @todo: Implement PCL GICP/NDT registration here.
        // Match 'cloud' against your maintained sliding window map.
        // Return the resulting transform as a GTSAM Pose3.
        return gtsam::Pose3();
    }

    void updateGraph(const gtsam::Pose3& odomStep) {
        // @todo: Add Prior factors (if first node) or BetweenFactors to gtSAMgraph_
        // @todo: Insert the initial guess into initialEstimate_
        gtsam::Symbol currentKey('X', keyFrameID);

        if (keyFrameID == 0) {
            // Add a PriorFactor to anchor the graph at the origin.
            gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(
                currentKey, gtsam::Pose3::Identity(), prior_noise_)
            );
            // Add the initial guess (Identity)
            initialEstimate_.insert(currentKey, gtsam::Pose3::Identity());

            keyFrameID++;
            return;
        }

        gtsam::Symbol previousKey('X', keyFrameID - 1);
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(previousKey, currentKey, odomStep, odom_noise_));
        currentPoseEstimate = currentPoseEstimate.compose(odomStep);
        initialEstimate_.insert(currentKey, currentPoseEstimate);
        keyFrameID++;
    }

    void detectLoopClosure(const pcl::PointCloud<pcl::PointXYZI>::Ptr& currentCloud) {
        // @todo: Implement KD-Tree search on historical poses.
        // If distance < threshold, perform ICP between current cloud and historical cloud.
        // If ICP fitness score is good, add a gtsam::BetweenFactor to gtSAMgraph_.

        // 1. Save the current pose and cloud into our history
        pcl::PointXYZ currentPose3D;
        currentPose3D.x = currentPoseEstimate.translation().x();
        currentPose3D.y = currentPoseEstimate.translation().y();
        currentPose3D.z = currentPoseEstimate.translation().z();
        cloudKeyPoses3D_->push_back(currentPose3D);
        cloudKeyframes_.push_back(currentCloud);

        int currentID = cloudKeyPoses3D_->size() - 1;

        // We need a minimum amount of history to even attempt a loop closure
        if (currentID < historySearchTimeDiff_) {
            return;
        }

        // 2. Build the KD-Tree using our trajectory
        kdtreeHistoryKeyPoses_->setInputCloud(cloudKeyPoses3D_);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // 3. Search for historical poses within a certain radius (e.g., 15 meters)
        kdtreeHistoryKeyPoses_->radiusSearch(currentPose3D, historySearchRadius_, pointSearchInd, pointSearchSqDis);

        int loopMatchID = -1;

        // 4. Filter matches: We only want poses from the past, not recent frames
        for (size_t i = 0; i < pointSearchInd.size(); ++i) {
            int candidateID = pointSearchInd[i];
            if (currentID - candidateID > historySearchTimeDiff_) {
                loopMatchID = candidateID;
                break; // Found the closest valid historical frame!
            }
        }

        if (loopMatchID == -1) {
            return; // No valid loop closures found
        }

        // 5. Verification: Run ICP between current cloud and historical cloud
        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        icp.setMaxCorrespondenceDistance(1.5); // Max distance to associate points
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);

        pcl::PointCloud<pcl::PointXYZI>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZI>());
        icp.align(*unused_result);

        // 6. Evaluate and Add to Graph
        if (icp.hasConverged() == false || icp.getFitnessScore() > icpFitnessThreshold_) {
            RCLCPP_DEBUG(this->get_logger(), "Loop closure rejected. Fitness: %f", icp.getFitnessScore());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "LOOP CLOSURE DETECTED! Frame %d -> Frame %d. Fitness: %f",
                    currentID, loopMatchID, icp.getFitnessScore());

        // Extract the transformation matrix calculated by ICP
        Eigen::Matrix4f icpTransform = icp.getFinalTransformation();

        // Convert to GTSAM Pose3
        gtsam::Rot3 rot(icpTransform.block<3,3>(0,0).cast<double>());
        gtsam::Point3 trans(icpTransform.block<3,1>(0,3).cast<double>());
        gtsam::Pose3 loopPose(rot, trans);

        // 7. Add Loop Closure Factor to the Graph
        // Note: Loop closures usually get a slightly less confident noise model than consecutive odometry
        gtsam::Vector6 loopSigmas;
        loopSigmas << 0.1, 0.1, 0.1, 0.2, 0.2, 0.2;
        auto loopNoise = gtsam::noiseModel::Diagonal::Sigmas(loopSigmas);

        gtsam::Symbol currentKey('X', currentID);
        gtsam::Symbol historicalKey('X', loopMatchID);

        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            historicalKey, currentKey, loopPose, loopNoise));
    }

    void publishIncrementalOdometry(const Eigen::Isometry3f& currentTf, const rclcpp::Time& timestamp) {
        nav_msgs::msg::Odometry odomMsg;
        odomMsg.header.stamp = timestamp;
        odomMsg.header.frame_id = "map";
        odomMsg.child_frame_id = "lidar";
        odomMsg.pose.pose.position.x = currentTf.translation().x();
        odomMsg.pose.pose.position.y = currentTf.translation().y();
        odomMsg.pose.pose.position.z = currentTf.translation().z();
        Eigen::Quaternionf q(currentTf.rotation());
        odomMsg.pose.pose.orientation.x = q.x();
        odomMsg.pose.pose.orientation.y = q.y();
        odomMsg.pose.pose.orientation.z = q.z();
        odomMsg.pose.pose.orientation.w = q.w();
        pubIncrementalOdom_->publish(odomMsg);
    }

    void publishGlobalOdometry(const Eigen::Isometry3f& currentTf, const rclcpp::Time& timestamp) {
        nav_msgs::msg::Odometry odomMsg;
        odomMsg.header.stamp = timestamp;
        odomMsg.header.frame_id = "map";
        odomMsg.child_frame_id = "lidar";
        odomMsg.pose.pose.position.x = currentTf.translation().x();
        odomMsg.pose.pose.position.y = currentTf.translation().y();
        odomMsg.pose.pose.position.z = currentTf.translation().z();
        Eigen::Quaternionf q(currentTf.rotation());
        odomMsg.pose.pose.orientation.x = q.x();
        odomMsg.pose.pose.orientation.y = q.y();
        odomMsg.pose.pose.orientation.z = q.z();
        odomMsg.pose.pose.orientation.w = q.w();
        pubGlobalOdom_->publish(odomMsg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalMapNode>());
    rclcpp::shutdown();
    return 0;
}
