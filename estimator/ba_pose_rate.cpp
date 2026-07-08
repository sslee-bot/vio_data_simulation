#include "ba_pose_rate.h"
#include "imu_preintegration.h"

#include <array>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vio {
namespace {

// Pose-measurement factor: state pose at t_k+td must match the measured body pose.
struct PoseFactor {
    PoseFactor(const Eigen::Vector3d& z_p, const Eigen::Quaterniond& z_q,
               const Eigen::Vector3d& gyro, double m_pos, double m_rot)
        : z_p_(z_p), z_q_(z_q), gyro_(gyro), m_pos_(m_pos), m_rot_(m_rot) {}

    template <typename T>
    bool operator()(const T* qk, const T* pk, const T* vk, const T* bg,
                    const T* td, T* res) const {
        res[0] = (pk[0] + vk[0] * td[0] - T(z_p_.x())) / T(m_pos_);
        res[1] = (pk[1] + vk[1] * td[0] - T(z_p_.y())) / T(m_pos_);
        res[2] = (pk[2] + vk[2] * td[0] - T(z_p_.z())) / T(m_pos_);

        T w[3] = {T(gyro_.x()) - bg[0], T(gyro_.y()) - bg[1], T(gyro_.z()) - bg[2]};
        T dth[4] = {T(1), T(0.5) * w[0] * td[0], T(0.5) * w[1] * td[0], T(0.5) * w[2] * td[0]};
        T q_pred[4];
        ceres::QuaternionProduct(qk, dth, q_pred);
        T q_pred_inv[4]; QuatConj(q_pred, q_pred_inv);
        T zq[4] = {T(z_q_.w()), T(z_q_.x()), T(z_q_.y()), T(z_q_.z())};
        T qerr[4]; ceres::QuaternionProduct(q_pred_inv, zq, qerr);
        res[3] = (T(2) * qerr[1]) / T(m_rot_);
        res[4] = (T(2) * qerr[2]) / T(m_rot_);
        res[5] = (T(2) * qerr[3]) / T(m_rot_);
        return true;
    }

    Eigen::Vector3d z_p_;
    Eigen::Quaterniond z_q_;
    Eigen::Vector3d gyro_;
    double m_pos_, m_rot_;
};

} // namespace

std::vector<VisionOnly::Pose> BaPoseRate::Run(
    const std::vector<MotionData>& imu,
    const std::vector<VisionOnly::Pose>& cam_body, double td_init) {
    std::vector<VisionOnly::Pose> out;
    const size_t K = cam_body.size();
    if (K < 2 || imu.size() < 2) return out;

    std::vector<std::array<double, 4>> qs(K);
    std::vector<std::array<double, 3>> ps(K), vs(K);
    double bg[3] = {0, 0, 0};
    double ba[3] = {0, 0, 0};
    double td[1] = {td_init};

    for (size_t k = 0; k < K; ++k) {
        qs[k] = {cam_body[k].q.w(), cam_body[k].q.x(), cam_body[k].q.y(), cam_body[k].q.z()};
        ps[k] = {cam_body[k].p.x(), cam_body[k].p.y(), cam_body[k].p.z()};
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        if (k + 1 < K) {
            const double dt = cam_body[k + 1].timestamp - cam_body[k].timestamp;
            if (dt > 0) v = (cam_body[k + 1].p - cam_body[k].p) / dt;
        }
        vs[k] = {v.x(), v.y(), v.z()};
    }
    if (K >= 2) vs[K - 1] = vs[K - 2];

    ceres::Problem problem;
    ceres::LocalParameterization* qparam = new ceres::QuaternionParameterization();
    for (size_t k = 0; k < K; ++k) {
        problem.AddParameterBlock(qs[k].data(), 4, qparam);
        problem.AddParameterBlock(ps[k].data(), 3);
        problem.AddParameterBlock(vs[k].data(), 3);
    }
    problem.AddParameterBlock(bg, 3);
    problem.AddParameterBlock(ba, 3);
    problem.AddParameterBlock(td, 1);
    if (!opt_.estimate_td) problem.SetParameterBlockConstant(td);

    for (size_t k = 0; k < K; ++k) {
        const Eigen::Vector3d gyro = GyroAt(imu, cam_body[k].timestamp);
        auto* f = new PoseFactor(cam_body[k].p, cam_body[k].q, gyro, opt_.m_pos, opt_.m_rot);
        auto* cost = new ceres::AutoDiffCostFunction<PoseFactor, 6, 4, 3, 3, 3, 1>(f);
        problem.AddResidualBlock(cost, nullptr, qs[k].data(), ps[k].data(),
                                 vs[k].data(), bg, td);
    }

    for (size_t k = 0; k + 1 < K; ++k) {
        Preint pre = Preintegrate(imu, cam_body[k].timestamp, cam_body[k + 1].timestamp,
                                  Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                  opt_.n_gyro, opt_.n_acc, opt_.n_bg, opt_.n_ba);
        auto* f = new ImuFactor(pre, opt_.gravity);
        auto* cost = new ceres::AutoDiffCostFunction<ImuFactor, 9, 4, 3, 3, 4, 3, 3, 3, 3>(f);
        problem.AddResidualBlock(cost, nullptr, qs[k].data(), ps[k].data(),
                                 vs[k].data(), qs[k + 1].data(), ps[k + 1].data(),
                                 vs[k + 1].data(), bg, ba);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = opt_.max_iterations;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    td_ = td[0];
    bg_ = Eigen::Vector3d(bg[0], bg[1], bg[2]);
    ba_ = Eigen::Vector3d(ba[0], ba[1], ba[2]);

    out.reserve(K);
    for (size_t k = 0; k < K; ++k) {
        VisionOnly::Pose s;
        s.timestamp = cam_body[k].timestamp;
        s.q = Eigen::Quaterniond(qs[k][0], qs[k][1], qs[k][2], qs[k][3]).normalized();
        s.p = Eigen::Vector3d(ps[k][0], ps[k][1], ps[k][2]);
        out.push_back(s);
    }
    return out;
}

} // namespace vio
