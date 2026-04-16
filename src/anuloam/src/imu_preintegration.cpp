#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

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
      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu_correct", 10, std::bind(&ImuPreintegration::imuCallback, this, std::placeholders::_1)
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
      prior_vel_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // wtf this is really high?
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
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) const {
      RCLCPP_INFO(this->get_logger(),
        "IMU | accel: [%.3f, %.3f, %.3f]  gyro: [%.3f, %.3f, %.3f]",
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
        msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z);

        // convert imu measurement to ego frame using imu extrinsics
        sensor_msgs::msg::Imu imu_meas = *msg;

        // integrate this imu message
        double dt = 0.005; // TODO: calculate this smartly
        imu_integrator_->integrateMeasurement(gtsam::Vector3(imu_meas.linear_acceleration.x, imu_meas.linear_acceleration.y, imu_meas.linear_acceleration.z), 
                                              gtsam::Vector3(imu_meas.angular_velocity.x, imu_meas.angular_velocity.y, imu_meas.angular_velocity.z), dt);
    }

    // ROS subscribers 
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

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

    int key = 1;

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
