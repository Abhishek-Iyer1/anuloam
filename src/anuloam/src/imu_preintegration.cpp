#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class ImuPreintegration : public rclcpp::Node
{
  public:
    ImuPreintegration() : Node("imu_preintegration")
    {
      // ROS pub and subs
      odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
      path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu_correct", 10, std::bind(&ImuPreintegration::imuCallback, this, std::placeholders::_1)
      );
      odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom_incremental", 10, std::bind(&ImuPreintegration::odomCallback, this, std::placeholders::_1)
      );

      // params
      declare_parameter("accel_noise_sigma", 9e-4);
      get_parameter("accel_noise_sigma", accel_noise_sigma_);
      declare_parameter("gyro_noise_sigma", 1.6e-4);
      get_parameter("gyro_noise_sigma", gyro_noise_sigma_);
      declare_parameter("accel_bias_rw_sigma", 5e-4);
      get_parameter("accel_bias_rw_sigma", accel_bias_rw_sigma_);
      declare_parameter("gyro_bias_rw_sigma", 7e-5);
      get_parameter("gyro_bias_rw_sigma", gyro_bias_rw_sigma_);

      // initial values
      prev_pose_ = gtsam::Pose3::Identity();
      prev_vel_ = gtsam::Vector3::Zero();
      prev_bias_ = gtsam::imuBias::ConstantBias(); // zeros
      // from LIOSAM, make params?
      prior_pose_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());
      prior_vel_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e-2); // wtf this is really high?
      prior_bias_noise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

      // init factor graph
      gtsam::ISAM2Params isam_params;
      isam_params.relinearizeThreshold = 0.1;
      isam_params.relinearizeSkip = 1;
      isam2_ = gtsam::ISAM2(isam_params);

      // priors
      graph_.addPrior(X(0), prev_pose_, prior_pose_noise_);
      graph_.addPrior(V(0), prev_vel_, prior_vel_noise_);
      graph_.addPrior(B(0), prev_bias_, prior_bias_noise_);
      values_.insert(X(0), prev_pose_);
      values_.insert(V(0), prev_vel_);
      values_.insert(B(0), prev_bias_);

      // first update with priors only
      isam2_.update(graph_, values_);
      graph_.resize(0); // clear after each update, isam2 retains internally
      values_.clear();

      // set up preintegrator
      // TODO need to check if gravity already subtracted
      boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> p = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(); 
      p->accelerometerCovariance = gtsam::Matrix33::Identity(3,3) * pow(accel_noise_sigma_, 2);
      p->gyroscopeCovariance = gtsam::Matrix33::Identity(3,3) * pow(gyro_noise_sigma_, 2);
      p->integrationCovariance = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // TODO: should this be tunable?
      p->biasAccCovariance = gtsam::Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma_, 2);
      p->biasOmegaCovariance = gtsam::Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma_, 2);

      imu_integrator_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(p, prev_bias_);
    }

  private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
      RCLCPP_DEBUG(this->get_logger(),
        "IMU | accel: [%.3f, %.3f, %.3f]  gyro: [%.3f, %.3f, %.3f]",
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
        msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z);

      // convert imu measurement to ego frame using imu extrinsics
      sensor_msgs::msg::Imu imu_meas = *msg;

      // integrate this imu message
      double imuTimeNow = imu_meas.header.stamp.sec + imu_meas.header.stamp.nanosec * 1e-9;
      double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTimeNow - lastImuT_imu);
      lastImuT_imu = imuTimeNow;
      imu_integrator_->integrateMeasurement(gtsam::Vector3(imu_meas.linear_acceleration.x, imu_meas.linear_acceleration.y, imu_meas.linear_acceleration.z), 
                                            gtsam::Vector3(imu_meas.angular_velocity.x, imu_meas.angular_velocity.y, imu_meas.angular_velocity.z), dt);

      // predict odometry
      gtsam::NavState pred = imu_integrator_->predict(prev_state_, prev_bias_);

      // publish odometry msg
      nav_msgs::msg::Odometry odom_msg;
      odom_msg.header.stamp = imu_meas.header.stamp;
      odom_msg.header.frame_id = "map"; // temp?
      odom_msg.child_frame_id = "imu";

      odom_msg.pose.pose.position.x = pred.position().x();
      odom_msg.pose.pose.position.y = pred.position().y();
      odom_msg.pose.pose.position.z = pred.position().z();
      odom_msg.pose.pose.orientation.x = pred.pose().rotation().toQuaternion().x();
      odom_msg.pose.pose.orientation.y = pred.pose().rotation().toQuaternion().y();
      odom_msg.pose.pose.orientation.z = pred.pose().rotation().toQuaternion().z();
      odom_msg.pose.pose.orientation.w = pred.pose().rotation().toQuaternion().w();

      odom_msg.twist.twist.linear.x = pred.velocity().x();
      odom_msg.twist.twist.linear.y = pred.velocity().y();
      odom_msg.twist.twist.linear.z = pred.velocity().z();
      odom_msg.twist.twist.angular.x = imu_meas.angular_velocity.x + prev_bias_.gyroscope().x();
      odom_msg.twist.twist.angular.y = imu_meas.angular_velocity.y + prev_bias_.gyroscope().y();
      odom_msg.twist.twist.angular.z = imu_meas.angular_velocity.z + prev_bias_.gyroscope().z();

      odom_pub_->publish(odom_msg);

      // publish path at 10 Hz
      if (imuTimeNow - lastPathT_ >= 0.1) {
        lastPathT_ = imuTimeNow;
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = odom_msg.header;
        pose_stamped.pose = odom_msg.pose.pose;
        path_.header = odom_msg.header;
        path_.poses.push_back(pose_stamped);
        path_pub_->publish(path_);
      }

      // publish transform
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header = odom_msg.header;
      tf_msg.child_frame_id = odom_msg.child_frame_id;
      tf_msg.transform.translation.x = odom_msg.pose.pose.position.x;
      tf_msg.transform.translation.y = odom_msg.pose.pose.position.y;
      tf_msg.transform.translation.z = odom_msg.pose.pose.position.z;
      tf_msg.transform.rotation = odom_msg.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf_msg);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      // create gtsam object from ROS message
      float p_x = msg->pose.pose.position.x;
      float p_y = msg->pose.pose.position.y;
      float p_z = msg->pose.pose.position.z;
      float r_x = msg->pose.pose.orientation.x;
      float r_y = msg->pose.pose.orientation.y;
      float r_z = msg->pose.pose.orientation.z;
      float r_w = msg->pose.pose.orientation.w;
      gtsam::Pose3 lidar_pose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

      // create IMU factor
      const gtsam::PreintegratedCombinedMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*imu_integrator_);
      gtsam::CombinedImuFactor imu_factor(X(key-1), V(key-1), X(key), V(key), B(key-1), B(key), preint_imu);

      // add IMU factor to graph
      graph_.add(imu_factor);

      // create and add lidar factor
      gtsam::PriorFactor<gtsam::Pose3> lidar_factor(X(key), lidar_pose); // prior factor because we have value = pose
      graph_.add(lidar_factor);

      // add values
      // use imu preintegration prediction to initialize optimization
      gtsam::NavState prop_state = imu_integrator_->predict(prev_state_, prev_bias_);
      values_.insert(X(key), prop_state.pose());
      values_.insert(V(key), prop_state.v());
      values_.insert(B(key), prev_bias_);

      // optimize
      isam2_.update(graph_, values_);
      // isam2_.update(); // TODO: liosam updates twice, do we need to?

      // update prev state with optimization
      gtsam::Values result = isam2_.calculateEstimate();
      prev_pose_ = result.at<gtsam::Pose3>(X(key));
      prev_vel_ = result.at<gtsam::Vector3>(V(key));
      prev_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
      prev_state_ = gtsam::NavState(prev_pose_, prev_vel_);

      // clear preintegration and graphs
      imu_integrator_->resetIntegrationAndSetBias(prev_bias_);
      graph_.resize(0);
      values_.clear();

      key++;
    }

    // ROS pub and sub
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    nav_msgs::msg::Path path_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Factor graph variables
    gtsam::ISAM2 isam2_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;

    // PreintegratedCombinedMeasurements integrates the bias random walk 
    // into the preintegration itself using the random walk sigma (not an actual 
    // covariance, how fast the bias drifts over time). 
    // PreintegratedImuMeasurements must be used in conjunction with a manual 
    // BetweenFactor<imuBias::ConstantBias> to approximate this and even then
    // it is an approximation. 
    std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> imu_integrator_;

    gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr prior_vel_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr prior_bias_noise_;

    gtsam::Pose3 prev_pose_;
    gtsam::Vector3 prev_vel_;
    gtsam::imuBias::ConstantBias prev_bias_;
    gtsam::NavState prev_state_;

    int key = 1;
    double lastImuT_imu = -1;
    double lastPathT_ = -1;

    // noise params - take from IMU datasheet
    float accel_noise_sigma_;
    float gyro_noise_sigma_;
    float accel_bias_rw_sigma_;
    float gyro_bias_rw_sigma_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuPreintegration>());
  rclcpp::shutdown();
  return 0;
}
