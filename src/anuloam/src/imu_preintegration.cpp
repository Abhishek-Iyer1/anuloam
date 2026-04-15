#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>

class ImuPreintegration : public rclcpp::Node
{
  public:
    ImuPreintegration() : Node("imu_preintegration")
    {
      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu_correct", 10, std::bind(&ImuPreintegration::imuCallback, this, std::placeholders::_1)
      );
    }

  private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) const {
      RCLCPP_INFO(this->get_logger(),
        "IMU | accel: [%.3f, %.3f, %.3f]  gyro: [%.3f, %.3f, %.3f]",
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
        msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z);
    }

    // ROS subscribers 
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    // Factor graph variables
    gtsam::ISAM2 isam2_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;

    gtsam::PreintegratedImuMeasurements imu_integrator_;

    gtsam::imuBias::ConstantBias prev_bias_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuPreintegration>());
  rclcpp::shutdown();
  return 0;
}
