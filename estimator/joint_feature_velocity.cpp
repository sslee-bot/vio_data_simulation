#include "joint_feature_velocity.h"
#include "imu_preintegration.h"

#include <array>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vio {
namespace {

// Reprojection factor for a known landmark. td shifts the observation along its
// image-plane velocity (VINS-Fusion feature-velocity mechanism): the pose (q,p)
// projects the world landmark and is compared to the td-corrected observation.
struct ReprojTdFactor {
    ReprojTdFactor(const Eigen::Vector3d& pw, const Eigen::Vector2d& obs,
                   const Eigen::Vector2d& vel, const Eigen::Matrix3d& R_bc,
                   const Eigen::Vector3d& t_bc, double m_pix)
        : pw_(pw), obs_(obs), vel_(vel), R_bc_(R_bc), t_bc_(t_bc), m_pix_(m_pix) {}

    template <typename T>
    bool operator()(const T* qk, const T* pk, const T* td, T* res) const {
        // Landmark in body frame: R_wb^T (pw - p).
        T d[3] = {T(pw_.x()) - pk[0], T(pw_.y()) - pk[1], T(pw_.z()) - pk[2]};
        T qk_inv[4]; QuatConj(qk, qk_inv);
        T pb[3];
        ceres::QuaternionRotatePoint(qk_inv, d, pb);

        // Landmark in camera frame: R_bc^T (pb - t_bc).
        T e[3] = {pb[0] - T(t_bc_.x()), pb[1] - T(t_bc_.y()), pb[2] - T(t_bc_.z())};
        T pc[3];
        for (int r = 0; r < 3; ++r)
            pc[r] = T(R_bc_(0, r)) * e[0] + T(R_bc_(1, r)) * e[1] + T(R_bc_(2, r)) * e[2];

        // td-corrected observation: obs - vel * td.
        res[0] = (pc[0] / pc[2] - (T(obs_.x()) - T(vel_.x()) * td[0])) / T(m_pix_);
        res[1] = (pc[1] / pc[2] - (T(obs_.y()) - T(vel_.y()) * td[0])) / T(m_pix_);
        return true;
    }

    Eigen::Vector3d pw_;
    Eigen::Vector2d obs_, vel_;
    Eigen::Matrix3d R_bc_;
    Eigen::Vector3d t_bc_;
    double m_pix_;
};

// Alternative reprojection factor where td shifts the predicted body pose by the
// pose rate [v, w] instead of the observation by its feature velocity. Same
// tightly-coupled problem; only the td parameterization differs.
struct ReprojPoseRateFactor {
    ReprojPoseRateFactor(const Eigen::Vector3d& pw, const Eigen::Vector2d& obs,
                         const Eigen::Vector3d& gyro, const Eigen::Matrix3d& R_bc,
                         const Eigen::Vector3d& t_bc, double m_pix)
        : pw_(pw), obs_(obs), gyro_(gyro), R_bc_(R_bc), t_bc_(t_bc), m_pix_(m_pix) {}

    template <typename T>
    bool operator()(const T* qk, const T* pk, const T* vk, const T* bg,
                    const T* td, T* res) const {
        // Body state at t_k + td (shift by pose rate).
        T w[3] = {T(gyro_.x()) - bg[0], T(gyro_.y()) - bg[1], T(gyro_.z()) - bg[2]};
        T dth[4] = {T(1), T(0.5) * w[0] * td[0], T(0.5) * w[1] * td[0], T(0.5) * w[2] * td[0]};
        T q_s[4];
        ceres::QuaternionProduct(qk, dth, q_s);
        T p_s[3] = {pk[0] + vk[0] * td[0], pk[1] + vk[1] * td[0], pk[2] + vk[2] * td[0]};

        T d[3] = {T(pw_.x()) - p_s[0], T(pw_.y()) - p_s[1], T(pw_.z()) - p_s[2]};
        T q_s_inv[4]; QuatConj(q_s, q_s_inv);
        T pb[3];
        ceres::QuaternionRotatePoint(q_s_inv, d, pb);
        T e[3] = {pb[0] - T(t_bc_.x()), pb[1] - T(t_bc_.y()), pb[2] - T(t_bc_.z())};
        T pc[3];
        for (int r = 0; r < 3; ++r)
            pc[r] = T(R_bc_(0, r)) * e[0] + T(R_bc_(1, r)) * e[1] + T(R_bc_(2, r)) * e[2];
        res[0] = (pc[0] / pc[2] - T(obs_.x())) / T(m_pix_);
        res[1] = (pc[1] / pc[2] - T(obs_.y())) / T(m_pix_);
        return true;
    }

    Eigen::Vector3d pw_;
    Eigen::Vector2d obs_;
    Eigen::Vector3d gyro_;
    Eigen::Matrix3d R_bc_;
    Eigen::Vector3d t_bc_;
    double m_pix_;
};

using Key = std::tuple<long, long, long>;
Key LmKey(const Eigen::Vector3d& p) {
    return {std::lround(p.x() * 1e4), std::lround(p.y() * 1e4), std::lround(p.z() * 1e4)};
}

} // namespace

std::vector<VisionOnly::Pose> JointFeatureVelocity::Run(
    const std::vector<MotionData>& imu,
    const std::vector<FrameObservations>& frames,
    const std::vector<VisionOnly::Pose>& cam_body_init, const Eigen::Matrix3d& R_bc,
    const Eigen::Vector3d& t_bc, double td_init) {
    std::vector<VisionOnly::Pose> out;
    const size_t K = std::min(frames.size(), cam_body_init.size());
    if (K < 2 || imu.size() < 2) return out;

    // Precompute per-feature image-plane velocities (match landmarks across
    // adjacent frames by their known 3D position).
    std::vector<std::map<Key, Eigen::Vector2d>> obs_map(K);
    for (size_t k = 0; k < K; ++k)
        for (size_t i = 0; i < frames[k].landmarks_w.size(); ++i)
            obs_map[k][LmKey(frames[k].landmarks_w[i])] = frames[k].obs_norm[i];

    std::vector<std::vector<Eigen::Vector2d>> vel(K);
    for (size_t k = 0; k < K; ++k) {
        const size_t nbr = (k == 0) ? 1 : k - 1;
        const double dtk = cam_body_init[k].timestamp - cam_body_init[nbr].timestamp;
        vel[k].assign(frames[k].landmarks_w.size(), Eigen::Vector2d::Zero());
        if (std::fabs(dtk) < 1e-9) continue;
        for (size_t i = 0; i < frames[k].landmarks_w.size(); ++i) {
            auto it = obs_map[nbr].find(LmKey(frames[k].landmarks_w[i]));
            if (it != obs_map[nbr].end())
                vel[k][i] = (frames[k].obs_norm[i] - it->second) / dtk;
        }
    }

    std::vector<std::array<double, 4>> qs(K);
    std::vector<std::array<double, 3>> ps(K), vs(K);
    double bg[3] = {0, 0, 0};
    double ba[3] = {0, 0, 0};
    double td[1] = {td_init};

    for (size_t k = 0; k < K; ++k) {
        qs[k] = {cam_body_init[k].q.w(), cam_body_init[k].q.x(),
                 cam_body_init[k].q.y(), cam_body_init[k].q.z()};
        ps[k] = {cam_body_init[k].p.x(), cam_body_init[k].p.y(), cam_body_init[k].p.z()};
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        if (k + 1 < K) {
            const double dt = cam_body_init[k + 1].timestamp - cam_body_init[k].timestamp;
            if (dt > 0) v = (cam_body_init[k + 1].p - cam_body_init[k].p) / dt;
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

    // Reprojection factors (raw known-landmark observations) — tight coupling.
    // td enters either via feature velocity (obs shift) or via pose rate.
    const bool feat_vel = (opt_.td_mode == TdMode::kFeatureVelocity);
    for (size_t k = 0; k < K; ++k) {
        const auto& f = frames[k];
        const Eigen::Vector3d gyro = GyroAt(imu, cam_body_init[k].timestamp);
        for (size_t i = 0; i < f.landmarks_w.size(); ++i) {
            if (feat_vel) {
                auto* fac = new ReprojTdFactor(f.landmarks_w[i], f.obs_norm[i],
                                               vel[k][i], R_bc, t_bc, opt_.m_pix);
                auto* cost = new ceres::AutoDiffCostFunction<ReprojTdFactor, 2, 4, 3, 1>(fac);
                problem.AddResidualBlock(cost, nullptr, qs[k].data(), ps[k].data(), td);
            } else {
                auto* fac = new ReprojPoseRateFactor(f.landmarks_w[i], f.obs_norm[i],
                                                     gyro, R_bc, t_bc, opt_.m_pix);
                auto* cost = new ceres::AutoDiffCostFunction<ReprojPoseRateFactor, 2, 4, 3, 3, 3, 1>(fac);
                problem.AddResidualBlock(cost, nullptr, qs[k].data(), ps[k].data(),
                                         vs[k].data(), bg, td);
            }
        }
    }

    // IMU preintegration factors between consecutive frames.
    for (size_t k = 0; k + 1 < K; ++k) {
        Preint pre = Preintegrate(imu, cam_body_init[k].timestamp,
                                  cam_body_init[k + 1].timestamp,
                                  Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                  opt_.n_gyro, opt_.n_acc, opt_.n_bg, opt_.n_ba);
        auto* fac = new ImuFactor(pre, opt_.gravity);
        auto* cost = new ceres::AutoDiffCostFunction<ImuFactor, 9, 4, 3, 3, 4, 3, 3, 3, 3>(fac);
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
        s.timestamp = cam_body_init[k].timestamp;
        s.q = Eigen::Quaterniond(qs[k][0], qs[k][1], qs[k][2], qs[k][3]).normalized();
        s.p = Eigen::Vector3d(ps[k][0], ps[k][1], ps[k][2]);
        out.push_back(s);
    }
    return out;
}

} // namespace vio
