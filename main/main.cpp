#include <climits>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../util/data_loader.h"
#include "../estimator/estimator.h"

namespace {
// Resolve a path to its absolute form for display; fall back to the input if it
// cannot be resolved.
std::string AbsolutePath(const std::string& p) {
    char buf[PATH_MAX];
    if (realpath(p.c_str(), buf)) return std::string(buf);
    return p;
}
} // namespace

// Self-contained entry point for the VIO estimator. Loads the data produced by
// the simulator (bin/data_gen) and hands it to the estimator.
//
// Run from the bin/ directory (where data_gen writes its output), same as data_gen.
// Usage: vio_estimator [data_dir]   (data_dir defaults to the current directory)
int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : ".";

    vio::SimData data;
    if (!vio::LoadSimData(data_dir, data)) {
        std::cerr << "Failed to load simulator data from '" << data_dir
                  << "'. Run bin/data_gen first (see README)." << std::endl;
        return 1;
    }

    std::cout << "Loaded simulator data from '" << AbsolutePath(data_dir) << "':\n"
              << "  IMU samples     : " << data.imu.size() << "\n"
              << "  Camera poses    : " << data.cam.size() << "\n"
              << "  Keyframe frames : " << data.frames.size() << std::endl;

    vio::Estimator estimator;
    estimator.SetData(data);
    estimator.Run();

    const std::string out = data_dir + "/vio_estimated_tum.txt";
    estimator.SaveTrajectoryTUM(out);
    std::cout << "Estimated poses  : " << estimator.trajectory().size() << "\n"
              << "Trajectory saved : " << AbsolutePath(out) << std::endl;

    return 0;
}
