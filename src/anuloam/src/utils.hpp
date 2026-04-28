#include <vector>
#include <cassert>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <chrono>
#include <string>
#include <cstdio>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>

template <typename T>
class CircularBuffer {
public:
    CircularBuffer(size_t size) : _head(0), _isFull(false), _maxSize(size) {
        _data.resize(size);
    }

    void push_back(const T& elem) {
        _data[_head] = elem;
        _head++;

        if (_head >= _maxSize) {
            _head = 0;
            _isFull = true;
        }
    }

    size_t size() const {
        if (_isFull) {return _maxSize;}
        else {return _head;}
    }

    bool empty() const { return !_isFull && _head == 0; }

    const T& back() const {
        assert(!empty() && "CircularBuffer: back() called on empty buffer!");
        size_t lastIdx = (_head == 0) ? _maxSize - 1 : _head - 1;
        return _data[lastIdx];
    }

    /**
     * @warn: You are indexing chronologically, such that 0 is the oldest data point
     */
    const T& operator[](size_t i) const {
        assert(i < this->size() && "CircularBuffer: Index out of logical bounds!");
        if (_isFull) {
            return _data[(_head + i) % _maxSize];
        }
        return _data[i];
    }

private:
    std::vector<T> _data;
    size_t _head;
    bool _isFull;
    size_t _maxSize;
};

/**
 * @brief Finds the index of the element with the closest timestamp in a chronological CircularBuffer.
 * * @tparam T The type of data stored in the buffer.
 * @tparam TimeExtractor A callable type (e.g., lambda) that takes a T and returns a double (timestamp).
 * @param buffer The chronological CircularBuffer to search.
 * @param target_time The target timestamp in seconds.
 * @param extract_time The lambda to extract time from T.
 * @return size_t The index of the closest element.
 */
template <typename T, typename TimeExtractor>
size_t binarySearchClosest(const CircularBuffer<T>& buffer, double target_time, TimeExtractor extract_time) {
    if (buffer.empty()) return 0;

    size_t low = 0;
    size_t high = buffer.size() - 1;

    while (low <= high) {
        size_t mid = low + (high - low) / 2;
        double mid_time = extract_time(buffer[mid]);

        if (mid_time == target_time) {
            return mid; // Exact match found
        } else if (mid_time < target_time) {
            low = mid + 1;
        } else {
            if (mid == 0) break; // Prevent underflow on unsigned size_t
            high = mid - 1;
        }
    }

    // Boundary checks: target is entirely before or after the buffer's time window
    if (low == 0) return 0;
    if (low >= buffer.size()) return buffer.size() - 1;

    // The target falls between 'low - 1' and 'low'. Find which is closer.
    double diff_lower = target_time - extract_time(buffer[low - 1]);
    double diff_upper = extract_time(buffer[low]) - target_time;

    return (diff_lower <= diff_upper) ? (low - 1) : low;
}

/**
 * @brief Converts a ROS 2 Odometry message to an Eigen::Isometry3f
 */
inline Eigen::Isometry3f odomMsgToEigen(const nav_msgs::msg::Odometry& msg) {
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();

    Eigen::Quaternionf q(
        msg.pose.pose.orientation.w,
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z
    );

    tf.linear() = q.normalized().toRotationMatrix();

    tf.translation() = Eigen::Vector3f(
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    );

    return tf;
}

inline Eigen::Matrix3f skew(const Eigen::Vector3f& v) {
    Eigen::Matrix3f m;
    m <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return m;
}

inline double stamp2sec(const builtin_interfaces::msg::Time& stamp) {
    return stamp.sec + stamp.nanosec * 1e-9;
}

struct ScopedTimer {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    ScopedTimer(std::string n) : name(n), start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::printf("[PROFILE] %s took %.3f ms\n", name.c_str(), duration.count());
    }
};

// Macro to make it easy to drop into any block
#define PROFILE_BLOCK(name) ScopedTimer timer_##__LINE__(name)

/**
 * @brief: LIO-SAM IMU converter which performs necessary frame conversions
 */
inline const Eigen::Matrix3d extRot = (Eigen::Matrix3d() << 
    -1.0,  0.0,  0.0,
     0.0,  1.0,  0.0,
     0.0,  0.0, -1.0).finished();

inline const Eigen::Matrix3d extRPY = (Eigen::Matrix3d() << 
     0.0,  1.0,  0.0,
    -1.0,  0.0,  0.0,
     0.0,  0.0,  1.0).finished();

inline const Eigen::Quaterniond extQRPY(extRPY);

inline sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu& imu_in)
{
    sensor_msgs::msg::Imu imu_out = imu_in;
    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
    {
        RCLCPP_ERROR(rclcpp::get_logger("imu_converter"), "Invalid quaternion, please use a 9-axis IMU!");
        rclcpp::shutdown();
    }

    return imu_out;
}