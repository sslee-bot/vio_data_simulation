#include "estimator.h"
#include "../src/utilities.h"   // save_Pose_asTUM

#include <algorithm>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vio {
namespace {

// Reprojection residual: project a known world point with the camera-from-world
// pose (q_cw, t_cw) and compare against the normalized image observation.
struct ReprojectionError {
    ReprojectionError(const Eigen::Vector3d& pw, const Eigen::Vector2d& obs)
        : pw_(pw), obs_(obs) {}

    template <typename T>
    bool operator()(const T* const q_cw, const T* const t_cw, T* residual) const {
        T pw[3] = {T(pw_.x()), T(pw_.y()), T(pw_.z())};
        T pc[3];
        // pc <- R(q_cw) * pw
        // pc += t_cw
        ceres::QuaternionRotatePoint(q_cw, pw, pc);   // q_cw is [w, x, y, z]
        pc[0] += t_cw[0];
        pc[1] += t_cw[1];
        pc[2] += t_cw[2];
        residual[0] = pc[0] / pc[2] - T(obs_.x());
        residual[1] = pc[1] / pc[2] - T(obs_.y());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pw,
                                       const Eigen::Vector2d& obs) {
        // <functor, residual dim = 2, q_cw size = 4, t_cw size = 3>
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3>(
            new ReprojectionError(pw, obs));
    }

    Eigen::Vector3d pw_;
    Eigen::Vector2d obs_;
};

// world<-cam (R_wc, t_wc) -> cam<-world (q_cw as [w,x,y,z], t_cw).
void WorldCamToCamWorld(const Eigen::Matrix3d& R_wc, const Eigen::Vector3d& t_wc,
                        double* q_cw, double* t_cw) {
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    Eigen::Quaterniond q(R_cw);
    q.normalize();
    q_cw[0] = q.w();
    q_cw[1] = q.x();
    q_cw[2] = q.y();
    q_cw[3] = q.z();
    const Eigen::Vector3d t = -R_cw * t_wc;
    t_cw[0] = t.x();
    t_cw[1] = t.y();
    t_cw[2] = t.z();
}

} // namespace

void Estimator::Run() {
    trajectory_.clear();
    if (data_.frames.empty()) return;

    // Camera-from-world pose, initialized from the previous frame's estimate.
    // The first frame is anchored to its known pose.
    double q_cw[4] = {1.0, 0.0, 0.0, 0.0};
    double t_cw[3] = {0.0, 0.0, 0.0};
    bool seeded = false;

    const size_t n = std::min(data_.frames.size(), data_.cam.size());
    for (size_t i = 0; i < n; ++i) {
        const FrameObservations& f = data_.frames[i];
        if (f.landmarks_w.size() < 4) continue;  // need enough 3D-2D correspondences

        // init at first frame
        if (!seeded) {
            WorldCamToCamWorld(data_.cam[i].Rwb, data_.cam[i].twb, q_cw, t_cw);
            seeded = true;
        }

        ceres::Problem problem;
        problem.AddParameterBlock(q_cw, 4, new ceres::QuaternionParameterization());
        problem.AddParameterBlock(t_cw, 3);
        for (size_t k = 0; k < f.landmarks_w.size(); ++k) {
            problem.AddResidualBlock(
                ReprojectionError::Create(f.landmarks_w[k], f.obs_norm[k]),
                nullptr, q_cw, t_cw);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 20;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Store the estimated world<-cam pose.
        const Eigen::Quaterniond q(q_cw[0], q_cw[1], q_cw[2], q_cw[3]); // w,x,y,z
        const Eigen::Matrix3d R_wc = q.toRotationMatrix().transpose();
        const Eigen::Vector3d t_wc =
            -R_wc * Eigen::Vector3d(t_cw[0], t_cw[1], t_cw[2]);

        Pose p;
        p.timestamp = data_.cam[i].timestamp;
        p.q = Eigen::Quaterniond(R_wc);
        p.p = t_wc;
        trajectory_.push_back(p);
    }
}

bool Estimator::SaveTrajectoryTUM(const std::string& path) const {
    std::vector<MotionData> poses;
    poses.reserve(trajectory_.size());
    for (const Pose& s : trajectory_) {
        MotionData d;
        d.timestamp = s.timestamp;
        d.Rwb = s.q.toRotationMatrix();
        d.twb = s.p;
        poses.push_back(d);
    }
    save_Pose_asTUM(path, poses);
    return true;
}

} // namespace vio
