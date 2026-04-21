#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Geometry>

#include <unordered_map>
#include <chrono> 
#include <mutex> 
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


// Custom hash function for Eigen::Vector3i so we can use it in a std::unordered_map
struct VoxelHash {
    size_t operator()(const Eigen::Vector3i& voxel) const {
        const size_t p1 = 73856093;
        const size_t p2 = 19349663;
        const size_t p3 = 83492791;
        return (voxel.x() * p1) ^ (voxel.y() * p2) ^ (voxel.z() * p3);
    }
};

struct VoxelData {
    Eigen::Vector3f point;
    float intensity;
    float weight;
};

/**
 * @brief Stateful Voxel Hash Map. 
 * Performs O(1) weighted running averages (merges) for the Global Map visualization.
 */
class GlobalPointMap {
public:
    // Added min_weight threshold. 2.0f means a point must be seen at least twice.
    GlobalPointMap(float leaf_size = 0.4f, float max_range = 30.0f, float min_weight = 5.0f) 
        : leaf_size_(leaf_size), max_range_sq_(max_range * max_range), min_weight_(min_weight) {
        voxel_map_.reserve(100000); 
        // We set the pre-filter leaf size to match the hash map. 
        // This ensures we only do ~1 hash lookup per voxel per incoming frame.
        perFrameFilter_.setLeafSize(leaf_size_, leaf_size_, leaf_size_); 
    }

    // FAST PATH: Downsample, Hash, and Merge points using a weighted average
    void update(const pcl::PointCloud<pcl::PointXYZI>::Ptr& frame, const gtsam::Pose3& pose) {
        
        // 1. Downsample the incoming cloud FIRST to save computation
        pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>());
        perFrameFilter_.setInputCloud(frame);
        perFrameFilter_.filter(*downsampled);

        Eigen::Isometry3f transform = pose3ToEigenIsometry(pose);
        
        // 2. Transform and Merge the downsampled points
        for (const auto& pt : downsampled->points) {
            
            // MAX RANGE FILTER
            // Since pt is in the sensor frame, its distance to origin is its distance to the LiDAR
            float dist_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            if (dist_sq > max_range_sq_) {
                continue; 
            }

            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_world = transform * p;
            
            // Calculate which 3D voxel this point belongs to
            Eigen::Vector3i voxel_idx(
                std::floor(p_world.x() / leaf_size_),
                std::floor(p_world.y() / leaf_size_),
                std::floor(p_world.z() / leaf_size_)
            );

            auto it = voxel_map_.find(voxel_idx);
            if (it == voxel_map_.end()) {
                // Voxel is empty! Add it with weight 1.0
                voxel_map_[voxel_idx] = {p_world, pt.intensity, 1.0f};
            } else {
                // Voxel exists! Perform the running weighted average
                float w_old = it->second.weight;
                float w_new = w_old + 1.0f;
                
                it->second.point = (it->second.point * w_old + p_world) / w_new;
                it->second.intensity = (it->second.intensity * w_old + pt.intensity) / w_new;
                it->second.weight = w_new;
            }
        }
    }

    // SLOW PATH: Wipe the hash map and rebuild from history
    void construct(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& frames, 
                   const std::vector<gtsam::Pose3>& poses) {
        voxel_map_.clear();
        for (size_t i = 0; i < frames.size(); ++i) {
            update(frames[i], poses[i]);
        }
    }

    // Convert the Hash Map back into a PCL PointCloud for ROS
    pcl::PointCloud<pcl::PointXYZI>::Ptr getMap() {
        pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        out_cloud->points.reserve(voxel_map_.size());

        for (const auto& kv : voxel_map_) {
            // === NEW: MINIMUM WEIGHT FILTER ===
            if (kv.second.weight < min_weight_) {
                continue; 
            }

            pcl::PointXYZI pt;
            pt.x = kv.second.point.x();
            pt.y = kv.second.point.y();
            pt.z = kv.second.point.z();
            
            // TRICK: We overwrite the "Intensity" field with the "Weight"!
            // Map the Intensity channel in Foxglove to visualize point confidence/opacity.
            pt.intensity = kv.second.weight; 
            
            out_cloud->push_back(pt);
        }
        
        out_cloud->width = out_cloud->points.size();
        out_cloud->height = 1;
        out_cloud->is_dense = true;
        
        return out_cloud;
    }

private:
    float leaf_size_;
    float max_range_sq_; 
    float min_weight_; // NEW: Store the minimum weight to publish
    std::unordered_map<Eigen::Vector3i, VoxelData, VoxelHash> voxel_map_;
    pcl::VoxelGrid<pcl::PointXYZI> perFrameFilter_;
};


/**
 * @todo: Need to support each point in the scan having a timestamp parameter
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
        size_t numSections = 6;
        size_t edgeFeaturesPerSection = 1;
        size_t planarFeaturesPerSection = 3;

        // Add 2 edge and 4 planar features per section to LidarFrames
        for (size_t section=1; section <= numSections; section++) {

            size_t startInd = static_cast<size_t>(static_cast<float>(section - 1) / numSections * scanSize);
            size_t endInd = static_cast<size_t>(static_cast<float>(section) / numSections * scanSize);

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
void scanMatching(const LocalMap& map, LidarFrame& target, Eigen::Isometry3f& currentGuess,
    std::function<void(const pcl::PointCloud<PointXYZIR>&, int)> iterCallback = nullptr) {
        
    // PROFILE_BLOCK("scanMatching");
    if (map.getKeyframes().size() == 0) {return;}

    const int maxIterations = 25;
    const float epsilon = 1e-5;
    const float maxDistSq = 20.0;

    for (int iter = 0; iter < maxIterations; ++iter) {

        // Go over the incoming LidarFrame
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Vector<float, 6> g = Eigen::Vector<float, 6>::Zero();

        float edgeLoss = 0.0f;
        float patchLoss = 0.0f;
        int edgeCount = 0;
        int patchCount = 0;

        const pcl::PointCloud<PointXYZIR> transformedEdges = target.transformEdges(currentGuess);
        const pcl::PointCloud<PointXYZIR> transformedPatches = target.transformPatches(currentGuess);

        for (const auto& edge : transformedEdges) {
            Eigen::Vector3f pi = edge.getVector3fMap();
            std::vector<int> indices;
            std::vector<float> dists;
            bool found;
            {
                // PROFILE_BLOCK("KDTree_Edge_Search");
                found = map.findEdgeNeighbors(edge, indices, dists);
            }
            if (found) {
                // PROFILE_BLOCK("Edge_Jacobian_Calculation");
                if (dists[1] > maxDistSq) continue;

                Eigen::Vector3f pj = (map.getEdges())[indices[0]].getVector3fMap();
                Eigen::Vector3f pl = (map.getEdges())[indices[1]].getVector3fMap();

                // Verification: Line segment must have non-zero length
                if ((pj - pl).norm() < 1e-3f) continue;

                // Math: d = |(pi-pj) x (pi-pl)| / |pj-pl|
                // vUnit = (pj - pl) / || pj - pl ||
                // Jacobian = [ [vUnit]_x @ [P_i]_x     -[vUnit]_x ]
                // b_i = [e_0]_x @ vUnit
                Eigen::Vector3f vUnit = (pl - pj) / (pl - pj).norm();
                Eigen::Vector3f bi = (pi - pj).cross(vUnit); // 3x1

                edgeLoss += bi.squaredNorm();
                edgeCount++;

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
            bool found;
            {
                // PROFILE_BLOCK("KDTree_Plane_Search");
                found = map.findPlaneNeighbors(patch, indices, dists);
            }
            if (found) {
                // PROFILE_BLOCK("Plane_Jacobian_Calculation");
                if (dists[2] > maxDistSq) continue;

                Eigen::Vector3f pu = (map.getPatches())[indices[0]].getVector3fMap();
                Eigen::Vector3f pv = (map.getPatches())[indices[1]].getVector3fMap();
                Eigen::Vector3f pw = (map.getPatches())[indices[2]].getVector3fMap();

                // Verification: Line segment must have non-zero length
                Eigen::Vector3f normal = (pu - pv).cross(pu - pw);
                float normalMag = normal.norm();

                if (normalMag < 1e-3f) continue;

                Eigen::Vector3f unitNormal = normal / normalMag;

                // Math: d = |(pu - pv) x (pw - pv) . (pi - pv)| / |(pu - pv) x  (pw - pv)|
                // unit_normal = (pu - pv) x  (pw - pv) / ||(pu - pv) x  (pw - pv)||
                // Jacobian = [(pi x ni).T      unit_normal.T]
                // b_i = unit_normal.T @ (pi - pv)

                float bi = unitNormal.transpose().dot(pi - pv);

                patchLoss += (bi * bi);
                patchCount++;

                Eigen::Matrix<float, 1, 6> Ai;
                Ai.leftCols<3>() = pi.cross(unitNormal).transpose();
                Ai.rightCols<3>() = unitNormal.transpose();

                H.noalias() += Ai.transpose() * Ai;
                g.noalias() += Ai.transpose() * bi;
            }
        }

        if (iterCallback) {
            pcl::PointCloud<PointXYZIR> currentAligned;
            pcl::transformPointCloud(target.getFeatures(), currentAligned, currentGuess.matrix());
            iterCallback(currentAligned, iter);
        }

        // Levenberg-Marquardt Damping
        H += Eigen::Matrix<float, 6, 6>::Identity() * 0.1f;

        Eigen::Vector<float, 6> deltaTransform = H.ldlt().solve(-g);

        // Step Relaxation
        deltaTransform *= 0.5f;

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
            // std::printf("Converged at iteration %d!\n", iter);
            break;
        }

    }

}


class GlobalMapNode : public rclcpp::Node {
public:
    GlobalMapNode() : Node("global_map"), localMap(50){
        // Threading and Safety setup
        imu_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        lidar_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // Subs and pubs
        rclcpp::SubscriptionOptions lidar_options;
        lidar_options.callback_group = lidar_cb_group_;
        subPoints_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points_raw",
            50,
            std::bind(&GlobalMapNode::pointCloudCallback, this, std::placeholders::_1),
            lidar_options
        );
        
        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = imu_cb_group_;
        subImuOdom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom_imu",
            10,
            std::bind(&GlobalMapNode::imuOdomCallback, this, std::placeholders::_1),
            imu_options
        );
        
        pubIncrementalOdom_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/odom_incremental",
            10
        );
        pubGlobalOdom_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/odom_global",
            10
        );
        pubLocalMapDebug_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/local_map_lidar",
            10
        );
        pubTest_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_test", 10);
        pubGlobalMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/global_map", 10);

        // TF2 Broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Create the 1Hz Timer for the Global Map
        map_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&GlobalMapNode::globalMapTimerCallback, this),
            timer_cb_group_
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
    // Threading and Safety
    rclcpp::CallbackGroup::SharedPtr imu_cb_group_;
    rclcpp::CallbackGroup::SharedPtr lidar_cb_group_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
    rclcpp::TimerBase::SharedPtr map_timer_;
    std::mutex imu_mutex_;
    std::mutex map_mutex_;

    // TF2 Broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Global Map State Tracking
    GlobalPointMap global_point_map_;
    bool loop_closure_detected_ = false;
    size_t last_processed_keyframe_ = 0;

    // Subs and pubs
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPoints_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubIncrementalOdom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubGlobalOdom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLocalMapDebug_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubTest_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubGlobalMap_;

    // LiDAR processing
    LocalMap localMap;
    const float keyframeTranslationThresh_ = 0.3f;   // 30 cm
    const float keyframeRotationThresh_ = 0.2618f;   // ~15 degrees in radians

    // LiDAR odometry variables
    std::deque<nav_msgs::msg::Odometry> imu_q_;
    Eigen::Isometry3f current_lo_pose_ = Eigen::Isometry3f::Identity();

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
    std::vector<gtsam::Pose3> cloudKeyPoses6D_;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudKeyframes_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeHistoryKeyPoses_{new pcl::KdTreeFLANN<pcl::PointXYZ>()};

    // Loop Closure Parameters
    double historySearchRadius_ = 15.0; // Meters
    int historySearchTimeDiff_ = 30;    // Minimum frames between current and history
    double icpFitnessThreshold_ = 0.3;  // Lower is better

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // PROFILE_BLOCK("PointCloud_Callback");
        // Convert ROS -> PCL
        pcl::PointCloud<pcl::PointXYZI>::Ptr currentCloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *currentCloud);

        LidarFrame LF = LidarFrame(*msg);
        LF.extract3DFeatures();

        // Construct initial guess and do scan matching
        const auto& keyframes = localMap.getKeyframes();
        Eigen::Isometry3f currentTf = Eigen::Isometry3f::Identity();
        if (!keyframes.empty()) {
            double cloud_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
            currentTf = keyframes.back().second;

            // Use IMU orientation if available for the guess
            {
                std::lock_guard<std::mutex> lock(imu_mutex_); 
                if (!imu_q_.empty()) {
                    nav_msgs::msg::Odometry imu_odo_meas;
                    // TODO manage q with lock
                    while (!imu_q_.empty()) {
                        imu_odo_meas = imu_q_.front();
                        double imu_time = imu_odo_meas.header.stamp.sec + imu_odo_meas.header.stamp.nanosec * 1e-9;
                        if (imu_time > cloud_time) {
                            break;
                        }
                        imu_q_.pop_front();
                    }

                    // Extract the orientation from the IMU message
                    const auto& o = imu_odo_meas.pose.pose.orientation;
                    Eigen::Quaternionf imu_quat(o.w, o.x, o.y, o.z);

                    // Overwrite the rotation of the guess with the IMU rotation, keeping translation intact
                    currentTf.linear() = imu_quat.normalized().toRotationMatrix();
                }
            }

            auto visualizeIter = [&](const pcl::PointCloud<PointXYZIR>& alignedCloud, int iter) {
                (void)iter; // Fixes unused parameter warning
                sensor_msgs::msg::PointCloud2 scanMsg;
                pcl::toROSMsg(alignedCloud, scanMsg);
                scanMsg.header = msg->header;
                scanMsg.header.frame_id = "map";
                pubTest_->publish(scanMsg);
            };

            scanMatching(localMap, LF, currentTf, visualizeIter);
        }

        // Publish whatever scan matching gives as the continuous odometry
        publishIncrementalOdometry(currentTf, msg->header.stamp);

        // Add a check for if tf greater than some threshold. If so add it as a keyframe
        bool should_add_keyframe = false;
        Eigen::Isometry3f delta = Eigen::Isometry3f::Identity();
        if (keyframes.size() == 0) {
            should_add_keyframe = true;
        } else {
            size_t last_idx = keyframes.size() - 1;
            Eigen::Isometry3f last_pose = keyframes[last_idx].second;

            delta = last_pose.inverse() * currentTf;

            float translation_diff = delta.translation().norm();
            float rotation_diff = std::abs(Eigen::AngleAxisf(delta.rotation()).angle());

            // Thresholds: 0.5 meters or ~5.7 degrees (0.1 radians)
            if (translation_diff > 0.5f || rotation_diff > 0.1f) {
                should_add_keyframe = true;
            }
        }

        if (!should_add_keyframe) {
            return;
        }

        // Add the current keyframe to the local map
        localMap.pushKeyframe({LF, currentTf});
        // RCLCPP_INFO(this->get_logger(), "New keyframe added. Local map size: %zu", localMap.getKeyframes().size());

        publishLocalMap(msg->header.stamp);

        // Update Factor Graph with the pose
        updateGraph(eigenIsometryToPose3(delta));

        // // Detect Loop Closure
        detectLoopClosure(currentCloud);

        // Optimize the factor graph
        isam_->update(gtSAMgraph_, initialEstimate_);
        isam_->update();

        // Clear graph and estimates for next iteration
        gtSAMgraph_.resize(0);
        initialEstimate_.clear();

        // Repopulate cloudKeyPoses6D_ with ISAM2-corrected poses
        gtsam::Values currentEstimate = isam_->calculateEstimate();
        cloudKeyPoses6D_.clear();
        for (int i = 0; i < keyFrameID; ++i) {
            cloudKeyPoses6D_.push_back(currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('X', i)));
        }
    }

    void imuOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_q_.push_back(*msg);
    }
    
    void globalMapTimerCallback() {
        publishGlobalMap(this->now());
    }

    void updateGraph(const gtsam::Pose3& odomStep) {
        // PROFILE_BLOCK("Update_Graph");
        gtsam::Symbol currentKey('X', keyFrameID);

        // Add Prior factors (if first node) or BetweenFactors to gtSAMgraph_
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

        // Insert the initial guess into initialEstimate_
        gtsam::Symbol previousKey('X', keyFrameID - 1);
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(previousKey, currentKey, odomStep, odom_noise_));
        currentPoseEstimate = currentPoseEstimate.compose(odomStep);
        initialEstimate_.insert(currentKey, currentPoseEstimate);
        keyFrameID++;
    }

    void detectLoopClosure(const pcl::PointCloud<pcl::PointXYZI>::Ptr& currentCloud) {
        // PROFILE_BLOCK("LoopClosures");
        // @todo: Implement KD-Tree search on historical poses.
        // If distance < threshold, perform ICP between current cloud and historical cloud.
        // If ICP fitness score is good, add a gtsam::BetweenFactor to gtSAMgraph_.

        // 1. Save the current pose and cloud into our history
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            pcl::PointXYZ currentPose3D;
            currentPose3D.x = currentPoseEstimate.translation().x();
            currentPose3D.y = currentPoseEstimate.translation().y();
            currentPose3D.z = currentPoseEstimate.translation().z();
            cloudKeyPoses3D_->push_back(currentPose3D);
            cloudKeyPoses6D_.push_back(currentPoseEstimate);
            cloudKeyframes_.push_back(currentCloud);
        }

        // Add return to visualize global map without loop closure
        // return;

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
        kdtreeHistoryKeyPoses_->radiusSearch(pcl::PointXYZ(currentPoseEstimate.translation().x(), currentPoseEstimate.translation().y(), currentPoseEstimate.translation().z()), historySearchRadius_, pointSearchInd, pointSearchSqDis);

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
        icp.setInputSource(currentCloud);
        icp.setInputTarget(cloudKeyframes_[loopMatchID]);
        gtsam::Values estimate = isam_->calculateEstimate();
        gtsam::Pose3 pCurrent = currentPoseEstimate;
        gtsam::Pose3 pHistory = estimate.at<gtsam::Pose3>(gtsam::Symbol('X', loopMatchID));
        Eigen::Matrix4f initialGuess = pose3ToEigenIsometry(pHistory.inverse().compose(pCurrent)).matrix();
        pcl::PointCloud<pcl::PointXYZI>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZI>());
        icp.align(*unused_result, initialGuess);

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
            
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            loop_closure_detected_ = true;
        }
    }

    void publishLocalMap(const rclcpp::Time& timestamp) {
        // PROFILE_BLOCK("publishLocalMap");
        sensor_msgs::msg::PointCloud2 cloudMsg;
        pcl::toROSMsg(localMap.getVoxels(), cloudMsg);
        cloudMsg.header.stamp = timestamp;
        cloudMsg.header.frame_id = "map";
        pubLocalMapDebug_->publish(cloudMsg);
    }

    void publishGlobalMap(const rclcpp::Time& timestamp) {
        // PROFILE_BLOCK("publishGlobalMap");
        
        std::vector<gtsam::Pose3> poses_copy;
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> frames_copy;
        bool reconstruct_map = false;

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            if (cloudKeyframes_.empty()) return;
            
            poses_copy = cloudKeyPoses6D_;
            frames_copy = cloudKeyframes_;
            
            if (loop_closure_detected_) {
                reconstruct_map = true;
                loop_closure_detected_ = false; 
            }
        }

        if (reconstruct_map) {
            // PROFILE_BLOCK("GlobalMap_Construct_Slow_Path");
            global_point_map_.construct(frames_copy, poses_copy);
            last_processed_keyframe_ = frames_copy.size();
        } else {
            // PROFILE_BLOCK("GlobalMap_Update_Fast_Path");
            for (size_t i = last_processed_keyframe_; i < frames_copy.size(); ++i) {
                global_point_map_.update(frames_copy[i], poses_copy[i]);
            }
            last_processed_keyframe_ = frames_copy.size();
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr map_to_publish = global_point_map_.getMap();

        sensor_msgs::msg::PointCloud2 cloudMsg;
        pcl::toROSMsg(*map_to_publish, cloudMsg); // This will take a few ms
        cloudMsg.header.stamp = timestamp;
        cloudMsg.header.frame_id = "map";
        pubGlobalMap_->publish(cloudMsg);
    }

    void publishIncrementalOdometry(const Eigen::Isometry3f& currentTf, const rclcpp::Time& timestamp) {
        // 1. Publish the Odom Message
        nav_msgs::msg::Odometry odomMsg;
        odomMsg.header.stamp = timestamp;
        odomMsg.header.frame_id = "map";
        odomMsg.child_frame_id = "velodyne";
        odomMsg.pose.pose.position.x = currentTf.translation().x();
        odomMsg.pose.pose.position.y = currentTf.translation().y();
        odomMsg.pose.pose.position.z = currentTf.translation().z();
        Eigen::Quaternionf q(currentTf.rotation());
        odomMsg.pose.pose.orientation.x = q.x();
        odomMsg.pose.pose.orientation.y = q.y();
        odomMsg.pose.pose.orientation.z = q.z();
        odomMsg.pose.pose.orientation.w = q.w();
        pubIncrementalOdom_->publish(odomMsg);

        // 2. Broadcast the TF Frame so you can see the robot model in RViz/Foxglove
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = timestamp;
        t.header.frame_id = "map";
        t.child_frame_id = "velodyne"; // This should match your robot's base or lidar frame
        t.transform.translation.x = currentTf.translation().x();
        t.transform.translation.y = currentTf.translation().y();
        t.transform.translation.z = currentTf.translation().z();
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(t);
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
    
    // Create the node
    auto node = std::make_shared<GlobalMapNode>(); 
    
    // Create a multi-threaded executor
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
    
    // Add the node to the executor and spin
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}