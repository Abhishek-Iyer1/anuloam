#include <vector>
#include <cassert>
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