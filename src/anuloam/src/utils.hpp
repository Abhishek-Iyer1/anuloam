#include <vector>
#include <cassert>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <chrono>
#include <string>
#include <cstdio>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Geometry>


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