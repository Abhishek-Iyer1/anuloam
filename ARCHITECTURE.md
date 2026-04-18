# Factor Graph Architecture

## Where the Factor Graph Lives

The factor graph lives inside the **backend state estimator node** — whichever
node is responsible for:

1. Deciding when to create new variables (keyframe selection)
2. Aggregating constraints from multiple sensor sources
3. Triggering and running optimization

The factor graph does not belong to the IMU because of its frequency. The IMU
is simply the highest-frequency sensor, which makes it a convenient backbone,
but the real owner is the state estimator.

In IMU-based systems, the backend estimator and IMU preintegration naturally
merge into a single node because of the bidirectional bias dependency (see
below). In IMU-free systems (lidar-only, visual SLAM), the backend is a
dedicated node that subscribes to relative pose measurements from frontend nodes.

---

## ImuPreintegrationNode Internal State

The factor graph and all GTSAM state live as member variables of the node:

```cpp
gtsam::ISAM2                          isam2_;          // incremental optimizer
gtsam::NonlinearFactorGraph           graph_;          // accumulates new factors
gtsam::Values                         values_;         // initial estimates for new variables
gtsam::PreintegratedImuMeasurements   imu_integrator_; // integrates raw IMU
gtsam::imuBias::ConstantBias          prev_bias_;      // last optimized bias
```

The preintegrated measurements are never published as a ROS message. They are
an internal GTSAM object constructed from raw IMU callbacks and consumed
directly when building an `ImuFactor`. There is no separate preintegration node.

---

## ROS Interface

The node has two subscriptions:

| Topic | Message Type | Rate | Callback purpose |
|-------|-------------|------|-----------------|
| `/imu/data` | `sensor_msgs/msg/Imu` | ~200 Hz | Buffer raw IMU, integrate into `imu_integrator_` |
| `/lidar_odom` | `nav_msgs/msg/Odometry` | ~10 Hz | Close preintegration window, add factors, optimize |

**`imuCallback`** only buffers or integrates the raw measurement. It does not
touch the factor graph.

**`odomCallback`** is the optimization trigger. It:
1. Drains buffered IMU messages up to the lidar timestamp (alignment)
2. Adds an `ImuFactor` for the accumulated preintegrated measurements
3. Adds a lidar pose constraint (`BetweenFactor` or `PoseFactor`)
4. Calls `isam2_->update(graph_, values_)`
5. Clears `graph_` and `values_` — iSAM2 retains factors internally
6. Extracts the corrected bias and resets `imu_integrator_`

Optimization runs at the lidar rate (~10 Hz), not the IMU rate (~200 Hz).

---

## The IMU Bidirectional Dependency

The IMU integrator and the optimizer have a tight feedback loop that requires
them to live in the same node.

**Forward direction — integrator feeds the graph:**

Raw IMU measurements are integrated into `imu_integrator_` using the current
bias estimate. At each keyframe, this produces an `ImuFactor` encoding the
predicted relative motion between keyframes:

```
ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), preintegrated_measurements)
```

The preintegrated measurements are baked with whatever bias was used during
integration.

**Backward direction — optimizer feeds the integrator:**

After `isam2_->update()`, the optimizer has a corrected bias estimate. This is
extracted and fed back into the integrator before the next window begins:

```cpp
auto corrected_bias = isam2_->calculateEstimate<gtsam::imuBias::ConstantBias>(B(key_));
imu_integrator_->resetIntegrationAndSetBias(corrected_bias);
```

Without this reset, the next window of integration starts with the old stale
bias, and the error compounds over time.

**The feedback loop:**

```
stale bias → bad preintegration → ImuFactor has error
                                         │
                            optimizer corrects bias estimate
                                         │
                            resetIntegrationAndSetBias()
                                         │
                            next window integrates with better bias
                                         ▲
                                    (repeat)
```

**Why splitting into two nodes breaks this:**

If the integrator and optimizer were in separate nodes, the corrected bias would
need to travel over a pub/sub round trip before the integrator could reset. At
200 Hz, even a few milliseconds of latency means the integrator has already
consumed new measurements with the stale bias before the correction arrives.
Keeping them co-located makes the reset a direct in-memory function call with
no latency.
