#ifndef VIO_ESTIMATOR_VISION_ONLY_H
#define VIO_ESTIMATOR_VISION_ONLY_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "../util/data_loader.h"

namespace vio {

// Vision-only camera-pose estimator: per-frame reprojection minimization
// (motion-only bundle adjustment / PnP) using the known landmark positions. IMU is
// not used here. The output camera trajectory is the input for the stage-2
// IMU-state / time-offset estimators.
class VisionOnly {
public:
    struct Pose {
        double timestamp = 0.0;
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // world <- camera
        Eigen::Vector3d p = Eigen::Vector3d::Zero();           // camera position in world
    };

    void SetData(const SimData& data) { data_ = data; }

    // Solve the per-frame PnP problems and populate trajectory_.
    void Run();

    // Write the estimated trajectory in TUM format, for later comparison against
    // ground truth and the dead-reckoning baseline.
    bool SaveTrajectoryTUM(const std::string& path) const;

    const std::vector<Pose>& trajectory() const { return trajectory_; }

    // Convert the estimated camera trajectory (world<-cam) to body poses
    // (world<-body) using the known camera-IMU extrinsics (R_bc, t_bc = cam->body).
    // These body poses at the camera timestamps are the measurements for stage 2.
    std::vector<Pose> BodyTrajectory(const Eigen::Matrix3d& R_bc,
                                     const Eigen::Vector3d& t_bc) const;

private:
    SimData data_;
    std::vector<Pose> trajectory_;
};

} // namespace vio

#endif // VIO_ESTIMATOR_VISION_ONLY_H
