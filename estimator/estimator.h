#ifndef VIO_ESTIMATOR_ESTIMATOR_H
#define VIO_ESTIMATOR_ESTIMATOR_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "../util/data_loader.h"

namespace vio {

// Consumes the simulated data and produces an estimated body trajectory.
//
// The estimation method itself — graph optimization or a filter — is implemented
// in Run(). This class only fixes the input/output seam (data in, trajectory out)
// so either approach can be dropped in later without further restructuring.
class Estimator {
public:
    struct Pose {
        double timestamp = 0.0;
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // world <- camera
        Eigen::Vector3d p = Eigen::Vector3d::Zero();           // camera position in world
    };

    void SetData(const SimData& data) { data_ = data; }

    // Run the estimation and populate trajectory_. Currently a stub.
    void Run();

    // Write the estimated trajectory in TUM format, for later comparison against
    // ground truth and the dead-reckoning baseline.
    bool SaveTrajectoryTUM(const std::string& path) const;

    const std::vector<Pose>& trajectory() const { return trajectory_; }

private:
    SimData data_;
    std::vector<Pose> trajectory_;
};

} // namespace vio

#endif // VIO_ESTIMATOR_ESTIMATOR_H
