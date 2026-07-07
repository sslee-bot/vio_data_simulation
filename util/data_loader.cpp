#include "data_loader.h"
#include "../src/utilities.h"   // LoadPose

#include <fstream>
#include <sstream>

namespace vio {

bool LoadFrameObservations(const std::string& path, FrameObservations& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    out.landmarks_w.clear();
    out.obs_norm.clear();

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        double x, y, z, w, u, v;
        if (!(ss >> x >> y >> z >> w >> u >> v)) continue;
        out.landmarks_w.emplace_back(x, y, z);
        out.obs_norm.emplace_back(u, v);
    }
    return true;
}

bool LoadSimData(const std::string& data_dir, SimData& out) {
    const std::string imu_file = data_dir + "/imu_pose_noise.txt";
    const std::string cam_file = data_dir + "/cam_pose.txt";

    out.imu.clear();
    out.cam.clear();
    out.frames.clear();

    LoadPose(imu_file, out.imu);
    if (out.imu.empty()) return false;

    LoadPose(cam_file, out.cam);
    if (out.cam.empty()) return false;

    // Read consecutive keyframe files (all_points_0.txt, all_points_1.txt, ...)
    // until the next index is missing.
    for (int n = 0;; ++n) {
        std::stringstream path;
        path << data_dir << "/keyframe/all_points_" << n << ".txt";
        FrameObservations frame;
        if (!LoadFrameObservations(path.str(), frame)) break;
        frame.frame_id = n;
        out.frames.push_back(std::move(frame));
    }

    return true;
}

} // namespace vio
