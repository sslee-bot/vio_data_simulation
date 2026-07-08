#include "ekf_pose_rate.h"

#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>

namespace vio {
namespace {

Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

// Small-angle body-frame increment quaternion for a rotation vector.
Eigen::Quaterniond DeltaQ(const Eigen::Vector3d& dtheta) {
    Eigen::Quaterniond dq;
    dq.w() = 1.0;
    dq.vec() = 0.5 * dtheta;
    dq.normalize();
    return dq;
}

} // namespace

void EkfPoseRate::Propagate(const Eigen::Vector3d& gyro,
                            const Eigen::Vector3d& acc, double dt) {
    const Eigen::Matrix3d R = q_.toRotationMatrix();
    const Eigen::Vector3d w_hat = gyro - bg_;          // body angular rate
    const Eigen::Vector3d a_hat = acc - ba_;           // body specific force
    const Eigen::Vector3d g_w(0.0, 0.0, -opt_.gravity);
    const Eigen::Vector3d a_world = R * a_hat + g_w;

    // Error-state transition F = I + Fc*dt (local/body orientation error).
    Eigen::Matrix<double, kN, kN> F = Eigen::Matrix<double, kN, kN>::Identity();
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;          // dp  <- dv
    F.block<3, 3>(3, 3) += -Skew(w_hat) * dt;                        // dtheta <- dtheta
    F.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity() * dt;         // dtheta <- dbg
    F.block<3, 3>(6, 3) = -R * Skew(a_hat) * dt;                     // dv <- dtheta
    F.block<3, 3>(6, 12) = -R * dt;                                  // dv <- dba

    // Process noise (diagonal), continuous densities integrated over dt.
    Eigen::Matrix<double, kN, kN> Q = Eigen::Matrix<double, kN, kN>::Zero();
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (opt_.n_gyro * opt_.n_gyro * dt);
    Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (opt_.n_acc * opt_.n_acc * dt);
    Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (opt_.n_bg * opt_.n_bg * dt);
    Q.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (opt_.n_ba * opt_.n_ba * dt);
    // td treated as constant (no process noise).

    P_ = F * P_ * F.transpose() + Q;

    // Nominal state propagation (Euler).
    p_ += v_ * dt + 0.5 * a_world * dt * dt;
    v_ += a_world * dt;
    q_ = (q_ * DeltaQ(w_hat * dt)).normalized();

    last_w_body_ = w_hat;
}

void EkfPoseRate::UpdatePose(const VisionOnly::Pose& meas) {
    // Residual r = z - h(x): position and body-frame orientation.
    Eigen::Matrix<double, 6, 1> r;
    r.segment<3>(0) = meas.p - p_;
    Eigen::Quaterniond dq = q_.conjugate() * meas.q;
    if (dq.w() < 0) dq.coeffs() *= -1.0;
    r.segment<3>(3) = 2.0 * dq.vec();

    // Measurement Jacobian H (6 x kN).
    Eigen::Matrix<double, 6, kN> H = Eigen::Matrix<double, 6, kN>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();   // position wrt dp
    H.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();   // orientation wrt dtheta
    // td column: measurement occurs at t_cam + td, so d(meas)/d(td) = pose rate.
    H.block<3, 1>(0, 15) = v_;             // position rate = world velocity
    H.block<3, 1>(3, 15) = last_w_body_;   // body orientation rate = body angular rate

    Eigen::Matrix<double, 6, 6> Rm = Eigen::Matrix<double, 6, 6>::Zero();
    Rm.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (opt_.m_pos * opt_.m_pos);
    Rm.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (opt_.m_rot * opt_.m_rot);

    const Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + Rm;
    const Eigen::Matrix<double, kN, 6> K = P_ * H.transpose() * S.inverse();
    Eigen::Matrix<double, kN, 1> dx = K * r;

    if (!opt_.estimate_td) dx(15) = 0.0;

    // Inject the correction into the nominal state.
    p_ += dx.segment<3>(0);
    q_ = (q_ * DeltaQ(dx.segment<3>(3))).normalized();
    v_ += dx.segment<3>(6);
    bg_ += dx.segment<3>(9);
    ba_ += dx.segment<3>(12);
    td_ += dx(15);

    Eigen::Matrix<double, kN, kN> I = Eigen::Matrix<double, kN, kN>::Identity();
    P_ = (I - K * H) * P_;
    P_ = 0.5 * (P_ + P_.transpose());   // keep symmetric
}

double EkfPoseRate::td_sigma() const { return std::sqrt(P_(15, 15)); }

std::vector<VisionOnly::Pose> EkfPoseRate::Run(
    const std::vector<MotionData>& imu,
    const std::vector<VisionOnly::Pose>& cam_body, double td_init) {
    std::vector<VisionOnly::Pose> out;
    if (cam_body.size() < 2 || imu.size() < 2) return out;

    // Initialize from the first camera pose; seed velocity by finite difference.
    p_ = cam_body[0].p;
    q_ = cam_body[0].q;
    const double dt0 = cam_body[1].timestamp - cam_body[0].timestamp;
    v_ = Eigen::Vector3d::Zero();
    if (dt0 > 0) v_ = (cam_body[1].p - cam_body[0].p) / dt0;
    bg_.setZero();
    ba_.setZero();
    td_ = td_init;

    P_.setZero();
    P_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (opt_.sigma_p * opt_.sigma_p);
    P_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (opt_.sigma_theta * opt_.sigma_theta);
    P_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (opt_.sigma_v * opt_.sigma_v);
    P_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (opt_.sigma_bg * opt_.sigma_bg);
    P_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (opt_.sigma_ba * opt_.sigma_ba);
    P_(15, 15) = opt_.sigma_td * opt_.sigma_td;

    auto record = [&](double stamp) {
        VisionOnly::Pose s;
        s.timestamp = stamp;
        s.q = q_;
        s.p = p_;
        out.push_back(s);
    };

    // Filter runs in IMU time; a camera frame labeled t is captured at t + td.
    double filter_time = cam_body[0].timestamp + td_;
    record(cam_body[0].timestamp);

    size_t j = 0;  // index of the IMU sample at/just before filter_time
    for (size_t k = 1; k < cam_body.size(); ++k) {
        const double target = cam_body[k].timestamp + td_;

        while (filter_time < target) {
            while (j + 1 < imu.size() && imu[j + 1].timestamp <= filter_time) ++j;
            const double next_t =
                (j + 1 < imu.size()) ? imu[j + 1].timestamp : target;
            const double step_end = std::min(next_t, target);
            const double dt = step_end - filter_time;
            if (dt > 1e-12) {
                // Midpoint rates over the step avoid the ~dt/2 lag of a
                // zero-order hold (which otherwise biases td by half an IMU step).
                Eigen::Vector3d gyro = imu[j].imu_gyro;
                Eigen::Vector3d acc = imu[j].imu_acc;
                if (j + 1 < imu.size()) {
                    gyro = 0.5 * (imu[j].imu_gyro + imu[j + 1].imu_gyro);
                    acc = 0.5 * (imu[j].imu_acc + imu[j + 1].imu_acc);
                }
                Propagate(gyro, acc, dt);
            }
            filter_time = step_end;
        }

        UpdatePose(cam_body[k]);
        record(cam_body[k].timestamp);
    }

    return out;
}

} // namespace vio
