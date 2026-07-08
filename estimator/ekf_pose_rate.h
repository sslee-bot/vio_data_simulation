#ifndef VIO_ESTIMATOR_EKF_POSE_RATE_H
#define VIO_ESTIMATOR_EKF_POSE_RATE_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "../util/data_loader.h"
#include "vision_only.h"

namespace vio {

// Stage 2 of the decoupled (loosely-coupled) estimator.
//
// Given the IMU stream and the camera-derived body poses from stage 1 (used AS
// MEASUREMENTS), a compact error-state EKF estimates the IMU state
//   p, v, q (world<-body), gyro bias bg, accel bias ba
// together with the camera-IMU time offset td. Gravity is a known ENU constant.
//
// td convention matches the simulator injection: a camera frame labeled with
// timestamp t was actually captured at IMU time t + td. The measurement is thus a
// function of the state at t + td, giving the td Jacobian H_td = [v; w] (linear and
// angular velocity), following Li & Mourikis (2014).
class EkfPoseRate {
public:
    struct Options {
        // Initial std-devs for the diagonal state covariance.
        double sigma_p = 0.1;      // position [m]
        double sigma_v = 0.1;      // velocity [m/s]
        double sigma_theta = 0.05; // orientation [rad]
        double sigma_bg = 1e-3;    // gyro bias [rad/s]
        double sigma_ba = 1e-2;    // accel bias [m/s^2]
        double sigma_td = 0.05;    // time offset [s]

        // IMU process noise (continuous-time densities).
        double n_gyro = 0.015;     // [rad/s/sqrt(Hz)]
        double n_acc = 0.019;      // [m/s^2/sqrt(Hz)]
        double n_bg = 1e-5;        // gyro bias random walk
        double n_ba = 1e-4;        // accel bias random walk

        // Pose-measurement noise (from the stage-1 camera poses).
        double m_pos = 0.02;       // position measurement std [m]
        double m_rot = 0.01;       // orientation measurement std [rad]

        double gravity = 9.81;     // ENU: g_w = (0,0,-gravity)
        bool estimate_td = true;   // if false, td is held at its initial value
    };

    EkfPoseRate() = default;
    explicit EkfPoseRate(const Options& opt) : opt_(opt) {}

    // Run the filter over the whole dataset:
    //  - imu: full IMU stream (timestamps in IMU clock)
    //  - cam_body: stage-1 body poses at camera timestamps (measurements)
    //  - td_init: initial guess for the time offset [s]
    // Returns the estimated body trajectory (at IMU-propagation times).
    std::vector<VisionOnly::Pose> Run(const std::vector<MotionData>& imu,
                                     const std::vector<VisionOnly::Pose>& cam_body,
                                     double td_init = 0.0);

    double td() const { return td_; }
    double td_sigma() const;                 // sqrt of the td covariance entry
    const Eigen::Vector3d& gyro_bias() const { return bg_; }
    const Eigen::Vector3d& acc_bias() const { return ba_; }

private:
    // Error-state layout (15 + 1): [dp(0..2) dtheta(3..5) dv(6..8) dbg(9..11) dba(12..14) dtd(15)]
    static constexpr int kN = 16;

    void Propagate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double dt);
    void UpdatePose(const VisionOnly::Pose& meas);

    Options opt_;

    // Nominal state.
    Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity(); // world<-body
    Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
    double td_ = 0.0;

    // Last body-frame angular rate (gyro - bias) used for the td Jacobian.
    Eigen::Vector3d last_w_body_ = Eigen::Vector3d::Zero();

    Eigen::Matrix<double, kN, kN> P_ = Eigen::Matrix<double, kN, kN>::Identity();
};

} // namespace vio

#endif // VIO_ESTIMATOR_EKF_POSE_RATE_H
