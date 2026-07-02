#include "estimator.h"
#include "../src/utilities.h"   // save_Pose_asTUM

namespace vio {

void Estimator::Run() {
    // TODO: implement the estimation method (graph optimization or filtering)
    // and fill trajectory_ with the estimated poses.
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
