#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

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



class GlobalMapNode : public rclcpp::Node {
public:
    GlobalMapNode() : Node("global_map") {
        // Subs and pubs
        subPoints_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points_raw",
            10,
            std::bind(&GlobalMapNode::pointCloudCallback, this, std::placeholders::_1)
        );
        subImuOdom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom_incremental",
            10,
            std::bind(&GlobalMapNode::imuOdomCallback, this, std::placeholders::_1)
        );
        pubLidarOdom_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/lidar_odom",
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
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLidarOdom_;

    // Factor graph variables
    int keyFrameID = 0;
    gtsam::Pose3 currentPoseEstimate = gtsam::Pose3::identity();

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

        // @todo: Perform Scan Matching
        gtsam::Pose3 relative_pose = performScanMatching(currentCloud);

        // @todo: Update Factor Graph with the pose
        updateGraph(relative_pose);

        // @todo: Detect Loop Closure
        detectLoopClosure(currentCloud);

        // @todo: Optimize the factor graph
        isam_->update(gtSAMgraph_, initialEstimate_);
        isam_->update(); // Often called twice to ensure convergence

        // Clear graph and estimates for next iteration
        gtSAMgraph_.resize(0);
        initialEstimate_.clear();

        // 6. Publish Odometry and TF
        gtsam::Values currentEstimate = isam_->calculateEstimate();
        publishLidarOdometry(currentEstimate, msg->header.stamp);

        // Add logging
        RCLCPP_INFO(this->get_logger(), "Received %zu points", currentCloud->points.size());
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
                currentKey, gtsam::Pose3::identity(), priorNoise_)
            );
            // Add the initial guess (Identity)
            initialEstimate_.insert(currentKey, gtsam::Pose3::identity());

            keyFrameID++;
            return;
        }

        gtsam::Symbol previousKey('X', keyFrameID - 1);
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(previousKey, currentKey, odomStep, odomNoise_));
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

    void publishLidarOdometry(const gtsam::Values& currentEstimate, const rclcpp::Time& timestamp) {
        nav_msgs::Odometry lidarOdom;
        lidarOdom.header.stamp = timestamp;
        lidarOdom.header.frame_id = "odom";
        lidarOdom.child_frame_id = "lidar";
        lidarOdom.pose.pose.position.x = currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('X', 0)).translation().x();
        lidarOdom.pose.pose.position.y = currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('X', 0)).translation().y();
        lidarOdom.pose.pose.position.z = currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('X', 0)).translation().z();
        lidarOdom.pose.pose.orientation = currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('X', 0)).rotation().toQuaternion();
        pubLidarOdom_->publish(lidarOdom);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalMapNode>());
    rclcpp::shutdown();
    return 0;
}
