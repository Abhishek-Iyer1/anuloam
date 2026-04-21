#include <vector>
#include <cassert>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <chrono>
#include <string>
#include <cstdio>

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