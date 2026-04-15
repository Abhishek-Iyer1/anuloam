# Design for Global Map class

## KeyFrame struct

Encapsulates all state and sensor data associated with a single keyframe.

    struct KeyFrame:
        - id                (size_t)
        - timestamp         (rclcpp::Time)
        - pose              (gtsam::Pose3)               -- optimized 6DOF pose in world frame
        - velocity          (gtsam::Vector3)             -- optimized velocity
        - imu_bias          (gtsam::imuBias::ConstantBias)
        - cloud             (pcl::PointCloud::Ptr)       -- downsampled feature cloud (edges + patches)
        - gtsam_key         (gtsam::Key)                 -- e.g. X(id), links to factor graph

Note: factors (IMU, LiDAR) are NOT stored per-keyframe — they live in the factor graph.

---

## GlobalMap class

### Data Members

    Keyframes & State
        - keyframes             (std::vector<KeyFrame>)
        - current_estimate      (gtsam::Values)          -- updated after every optimize() call

    Factor Graph & Solver
        - isam2                 (gtsam::ISAM2)           -- incremental solver, persistent
        - new_factors           (gtsam::NonlinearFactorGraph) -- staged factors, flushed on optimize()
        - new_values            (gtsam::Values)          -- staged initial guesses, flushed on optimize()

    IMU Pre-integration
        - imu_integrator        (gtsam::PreintegratedCombinedMeasurements) -- resets on each keyframe

    Loop Closure Spatial Index
        - keyframe_positions    (pcl::PointCloud<pcl::PointXYZ>::Ptr)  -- parallel to keyframes vector
        - kd_tree               (pcl::KdTreeFLANN<pcl::PointXYZ>)     -- spatial search over positions

### Functions

    IMU
        - add_imu_measurement(accel, gyro, dt)   -- feeds imu_integrator; called at IMU rate
        - add_imu_factor(from_id, to_id)         -- drains imu_integrator into new_factors; resets integrator

    Keyframes & Odometry
        - add_keyframe(cloud, initial_pose)      -- creates KeyFrame, inserts prior factor on first frame
        - add_lidar_factor(from_id, to_id, relative_pose, noise_model)

    Loop Closure
        - detect_loop_closure()                  -- radius search via kd_tree; returns candidate (i, j) pairs
        - add_loop_closure_factor(i, j, relative_pose, noise_model)  -- use robust noise model (Cauchy/Huber)

    Optimization & Output
        - optimize()                             -- calls isam2.update(new_factors, new_values); refreshes current_estimate
        - get_point_cloud(mode: full | sliding_window)  -- rebuilds map from current_estimate poses + keyframe clouds
