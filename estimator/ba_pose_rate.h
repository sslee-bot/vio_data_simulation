#ifndef VIO_ESTIMATOR_BA_POSE_RATE_H
#define VIO_ESTIMATOR_BA_POSE_RATE_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "../util/data_loader.h"
#include "vision_only.h"

namespace vio {

// Stage 2 of the decoupled estimator, solved as a factor-graph batch optimization
// (Ceres) instead of the EKF in ekf_pose_rate.h. This is the optimization-based
// counterpart used to compare against the filter.
//
// Variables: per camera-frame body pose (p_k, q_k) and velocity v_k, plus global
// gyro/accel biases and the scalar time offset td.
// Factors:
//   * IMU preintegration factor between consecutive frames (with first-order bias
//     correction), mirroring the VINS-Fusion IMU factor.
//   * Pose-measurement factor tying each state to the stage-1 camera body pose,
//     shifted by td along the pose rate [v; w] (the pose-level analog of the
//     VINS feature-velocity td trick).
class BaPoseRate {
public:
    struct Options {
        // IMU process-noise densities (same as the EKF, for a fair comparison).
        // The IMU factor is weighted by the propagated preintegration covariance,
        // as in VINS-Fusion (sqrt_info = chol(cov^{-1})).
        double n_gyro = 0.015;   // [rad/s/sqrt(Hz)]
        double n_acc = 0.019;    // [m/s^2/sqrt(Hz)]
        double n_bg = 1e-5;      // gyro bias random walk
        double n_ba = 1e-4;      // accel bias random walk
        // Pose-measurement std-devs (stage-1 camera poses).
        double m_pos = 0.02;     // position [m]
        double m_rot = 0.01;     // orientation [rad]
        double gravity = 9.81;   // ENU: g_w = (0,0,-gravity)
        bool estimate_td = true;
        int max_iterations = 50;
    };

    BaPoseRate() = default;
    explicit BaPoseRate(const Options& opt) : opt_(opt) {}

    std::vector<VisionOnly::Pose> Run(const std::vector<MotionData>& imu,
                                     const std::vector<VisionOnly::Pose>& cam_body,
                                     double td_init = 0.0);

    double td() const { return td_; }
    const Eigen::Vector3d& gyro_bias() const { return bg_; }
    const Eigen::Vector3d& acc_bias() const { return ba_; }

private:
    Options opt_;
    double td_ = 0.0;
    Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
};

} // namespace vio

#endif // VIO_ESTIMATOR_BA_POSE_RATE_H
