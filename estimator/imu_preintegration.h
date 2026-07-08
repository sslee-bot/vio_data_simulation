#ifndef VIO_ESTIMATOR_IMU_PREINTEGRATION_H
#define VIO_ESTIMATOR_IMU_PREINTEGRATION_H

#include <algorithm>
#include <cmath>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "../util/data_loader.h"

// Error-state ordering: [P(0:3) R(3:6) V(6:9) Ba(9:12) Bg(12:15)]. Midpoint rates
// avoid the dt/2 lag; a 15x15 covariance is propagated (VINS-Fusion style).
namespace vio {

inline Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

inline Eigen::Quaterniond DeltaQ(const Eigen::Vector3d& dtheta) {
    Eigen::Quaterniond dq;
    dq.w() = 1.0;
    dq.vec() = 0.5 * dtheta;
    return dq.normalized();
}

template <typename T>
inline void QuatConj(const T q[4], T out[4]) {
    out[0] = q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = -q[3];
}

struct Preint {
    double dt = 0.0;
    Eigen::Vector3d dp = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d Jp_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Jp_ba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Jv_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Jv_ba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Jq_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Zero();
    Eigen::Vector3d bg0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba0 = Eigen::Vector3d::Zero();
    double n_gyro = 0.015, n_acc = 0.019, n_bg = 1e-5, n_ba = 1e-4;
};

inline void IntegrateStep(Preint& s, const Eigen::Vector3d& gyro,
                          const Eigen::Vector3d& acc, double dt) {
    const Eigen::Vector3d w = gyro - s.bg0;
    const Eigen::Vector3d a = acc - s.ba0;
    const Eigen::Matrix3d R = s.dq.toRotationMatrix();
    const Eigen::Matrix3d a_skew = Skew(a);
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

    // Covariance propagation: cov = F cov F^T + Qd.
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
    F.block<3, 3>(0, 6) = I3 * dt;                 // P <- V
    F.block<3, 3>(3, 3) += -Skew(w) * dt;          // R <- R
    F.block<3, 3>(3, 12) = -I3 * dt;               // R <- Bg
    F.block<3, 3>(6, 3) = -R * a_skew * dt;        // V <- R
    F.block<3, 3>(6, 9) = -R * dt;                 // V <- Ba
    Eigen::Matrix<double, 15, 15> Qd = Eigen::Matrix<double, 15, 15>::Zero();
    Qd.block<3, 3>(3, 3) = I3 * (s.n_gyro * s.n_gyro * dt);
    Qd.block<3, 3>(6, 6) = I3 * (s.n_acc * s.n_acc * dt);
    Qd.block<3, 3>(9, 9) = I3 * (s.n_ba * s.n_ba * dt);
    Qd.block<3, 3>(12, 12) = I3 * (s.n_bg * s.n_bg * dt);
    s.cov = F * s.cov * F.transpose() + Qd;

    // Bias Jacobians (quantities at the start of the substep).
    s.Jp_ba += s.Jv_ba * dt - 0.5 * R * dt * dt;
    s.Jp_bg += s.Jv_bg * dt - 0.5 * R * a_skew * s.Jq_bg * dt * dt;
    s.Jv_ba += -R * dt;
    s.Jv_bg += -R * a_skew * s.Jq_bg * dt;

    // State.
    s.dp += s.dv * dt + 0.5 * R * a * dt * dt;
    s.dv += R * a * dt;
    const Eigen::Quaterniond dqi = DeltaQ(w * dt);
    s.Jq_bg = dqi.toRotationMatrix().transpose() * s.Jq_bg - I3 * dt;
    s.dq = (s.dq * dqi).normalized();
    s.dt += dt;
}

// Preintegrate over [t_a, t_b] (IMU clock) with midpoint rates.
inline Preint Preintegrate(const std::vector<MotionData>& imu, double t_a,
                           double t_b, const Eigen::Vector3d& bg0,
                           const Eigen::Vector3d& ba0, double n_gyro, double n_acc,
                           double n_bg, double n_ba) {
    Preint s;
    s.bg0 = bg0; s.ba0 = ba0;
    s.n_gyro = n_gyro; s.n_acc = n_acc; s.n_bg = n_bg; s.n_ba = n_ba;
    double t = t_a;
    size_t j = 0;
    while (j + 1 < imu.size() && imu[j + 1].timestamp <= t) ++j;
    while (t < t_b) {
        while (j + 1 < imu.size() && imu[j + 1].timestamp <= t) ++j;
        const double next_t = (j + 1 < imu.size()) ? imu[j + 1].timestamp : t_b;
        const double step_end = std::min(next_t, t_b);
        const double dt = step_end - t;
        if (dt > 1e-12) {
            Eigen::Vector3d gyro = imu[j].imu_gyro;
            Eigen::Vector3d acc = imu[j].imu_acc;
            if (j + 1 < imu.size()) {
                gyro = 0.5 * (imu[j].imu_gyro + imu[j + 1].imu_gyro);
                acc = 0.5 * (imu[j].imu_acc + imu[j + 1].imu_acc);
            }
            IntegrateStep(s, gyro, acc, dt);
        }
        t = step_end;
    }
    return s;
}

inline Eigen::Vector3d GyroAt(const std::vector<MotionData>& imu, double t) {
    size_t best = 0;
    double bd = 1e18;
    for (size_t i = 0; i < imu.size(); ++i) {
        const double d = std::fabs(imu[i].timestamp - t);
        if (d < bd) { bd = d; best = i; }
    }
    return imu[best].imu_gyro;
}

// IMU preintegration factor between consecutive frames, weighted by the propagated
// covariance ([P,R,V] 9x9 marginal). Residual order: [P, R, V].
struct ImuFactor {
    ImuFactor(const Preint& pre, double gravity) : pre_(pre), g_(gravity) {
        Eigen::Matrix<double, 9, 9> cov = pre_.cov.block<9, 9>(0, 0);
        cov += Eigen::Matrix<double, 9, 9>::Identity() * 1e-12;
        const Eigen::Matrix<double, 9, 9> info = cov.inverse();
        sqrt_info_ = Eigen::LLT<Eigen::Matrix<double, 9, 9>>(info).matrixL().transpose();
    }

    template <typename T>
    bool operator()(const T* qi, const T* pi, const T* vi, const T* qj,
                    const T* pj, const T* vj, const T* bg, const T* ba,
                    T* res) const {
        const T dt = T(pre_.dt);
        T dbg[3] = {bg[0] - T(pre_.bg0.x()), bg[1] - T(pre_.bg0.y()), bg[2] - T(pre_.bg0.z())};
        T dba[3] = {ba[0] - T(pre_.ba0.x()), ba[1] - T(pre_.ba0.y()), ba[2] - T(pre_.ba0.z())};

        T dp[3], dv[3];
        for (int r = 0; r < 3; ++r) {
            dp[r] = T(pre_.dp[r]);
            dv[r] = T(pre_.dv[r]);
            for (int c = 0; c < 3; ++c) {
                dp[r] += T(pre_.Jp_bg(r, c)) * dbg[c] + T(pre_.Jp_ba(r, c)) * dba[c];
                dv[r] += T(pre_.Jv_bg(r, c)) * dbg[c] + T(pre_.Jv_ba(r, c)) * dba[c];
            }
        }
        T corr[3] = {T(0), T(0), T(0)};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) corr[r] += T(pre_.Jq_bg(r, c)) * dbg[c];
        T dq_corr[4] = {T(1), T(0.5) * corr[0], T(0.5) * corr[1], T(0.5) * corr[2]};
        T dq0[4] = {T(pre_.dq.w()), T(pre_.dq.x()), T(pre_.dq.y()), T(pre_.dq.z())};
        T dq_c[4];
        ceres::QuaternionProduct(dq0, dq_corr, dq_c);

        const T g[3] = {T(0), T(0), T(-g_)};
        T tp[3], tv[3];
        for (int r = 0; r < 3; ++r) {
            tp[r] = pj[r] - pi[r] - vi[r] * dt - T(0.5) * g[r] * dt * dt;
            tv[r] = vj[r] - vi[r] - g[r] * dt;
        }
        T qi_inv[4]; QuatConj(qi, qi_inv);
        T Rp[3], Rv[3];
        ceres::QuaternionRotatePoint(qi_inv, tp, Rp);
        ceres::QuaternionRotatePoint(qi_inv, tv, Rv);

        T qij[4]; ceres::QuaternionProduct(qi_inv, qj, qij);
        T dq_c_inv[4]; QuatConj(dq_c, dq_c_inv);
        T qerr[4]; ceres::QuaternionProduct(dq_c_inv, qij, qerr);

        T raw[9] = {Rp[0] - dp[0], Rp[1] - dp[1], Rp[2] - dp[2],
                    T(2) * qerr[1], T(2) * qerr[2], T(2) * qerr[3],
                    Rv[0] - dv[0], Rv[1] - dv[1], Rv[2] - dv[2]};
        for (int i = 0; i < 9; ++i) {
            res[i] = T(0);
            for (int c = 0; c < 9; ++c) res[i] += T(sqrt_info_(i, c)) * raw[c];
        }
        return true;
    }

    Preint pre_;
    double g_;
    Eigen::Matrix<double, 9, 9> sqrt_info_ = Eigen::Matrix<double, 9, 9>::Identity();
};

} // namespace vio

#endif // VIO_ESTIMATOR_IMU_PREINTEGRATION_H
