#ifndef VIO_ESTIMATOR_JOINT_FEATURE_VELOCITY_H
#define VIO_ESTIMATOR_JOINT_FEATURE_VELOCITY_H

#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "../util/data_loader.h"
#include "vision_only.h"

namespace vio {

// Tightly-coupled joint estimator (VINS-Fusion style): camera poses, IMU state
// (velocity, biases) and the time offset td are all obtained in ONE optimization.
//
// Unlike the decoupled two-stage estimators (ekf_pose_rate / ba_pose_rate), the
// camera poses are NOT precomputed and fed in as measurements. Instead the raw
// reprojection residuals of the (known) landmarks and the IMU preintegration
// factors are solved together, so vision and inertial information are fused
// jointly. td shifts each observation's frame via the pose rate (v, w).
class JointFeatureVelocity {
public:
    // How td enters the reprojection factor:
    //  kFeatureVelocity : shift each observation by td * its image-plane velocity
    //                     (VINS-Fusion mechanism).
    //  kPoseRate        : shift the predicted body pose by td * pose rate [v, w].
    // Same tightly-coupled batch backbone; switching this isolates the effect of
    // the td parameterization alone.
    enum class TdMode { kFeatureVelocity, kPoseRate };

    struct Options {
        double n_gyro = 0.015, n_acc = 0.019, n_bg = 1e-5, n_ba = 1e-4;
        double m_pix = 1.0 / 460.0;   // reprojection std in normalized coords (~1 px)
        double gravity = 9.81;
        bool estimate_td = true;
        int max_iterations = 50;
        TdMode td_mode = TdMode::kFeatureVelocity;
    };

    JointFeatureVelocity() = default;
    explicit JointFeatureVelocity(const Options& opt) : opt_(opt) {}

    // imu: IMU stream; frames: per-frame landmark observations; cam_body_init:
    // stage-1 body poses used ONLY as initial values; (R_bc, t_bc): cam->body
    // extrinsics; td_init: initial time offset.
    std::vector<VisionOnly::Pose> Run(const std::vector<MotionData>& imu,
                                     const std::vector<FrameObservations>& frames,
                                     const std::vector<VisionOnly::Pose>& cam_body_init,
                                     const Eigen::Matrix3d& R_bc,
                                     const Eigen::Vector3d& t_bc,
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

#endif // VIO_ESTIMATOR_JOINT_FEATURE_VELOCITY_H
