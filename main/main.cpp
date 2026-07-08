#include <climits>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../util/data_loader.h"
#include "../estimator/vision_only.h"
#include "../estimator/ekf_pose_rate.h"
#include "../estimator/ba_pose_rate.h"
#include "../src/param.h"
#include "../src/utilities.h"

namespace {
// Resolve a path to its absolute form for display; fall back to the input if it
// cannot be resolved.
std::string AbsolutePath(const std::string& p) {
    char buf[PATH_MAX];
    if (realpath(p.c_str(), buf)) return std::string(buf);
    return p;
}

// Write a vector of poses (world<-body) to a TUM-format file.
void SavePosesTUM(const std::string& path,
                  const std::vector<vio::VisionOnly::Pose>& poses) {
    std::vector<MotionData> md;
    md.reserve(poses.size());
    for (const auto& s : poses) {
        MotionData d;
        d.timestamp = s.timestamp;
        d.Rwb = s.q.toRotationMatrix();
        d.twb = s.p;
        md.push_back(d);
    }
    save_Pose_asTUM(path, md);
}
} // namespace

// Run from the bin/ directory (where data_gen writes its output), same as data_gen.
// Usage: vio_estimator [data_dir] [td_init] [noisy]
//   data_dir defaults to the current directory; td_init defaults to 0;
//   pass "noisy" as the 3rd arg to use imu_pose_noise.txt (default: clean imu_pose.txt).
int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : ".";
    const double td_init = (argc > 2) ? atof(argv[2]) : 0.0;
    const bool use_noisy_imu = (argc > 3) && (std::string(argv[3]) == "noisy");

    vio::SimData data;
    if (!vio::LoadSimData(data_dir, data, use_noisy_imu)) {
        std::cerr << "Failed to load simulator data from '" << data_dir
                  << "'. Run bin/data_gen first (see README)." << std::endl;
        return 1;
    }

    std::cout << "Loaded simulator data from '" << AbsolutePath(data_dir) << "':\n"
              << "  IMU samples     : " << data.imu.size()
              << (use_noisy_imu ? "  (noisy)" : "  (clean)") << "\n"
              << "  Camera poses    : " << data.cam.size() << "\n"
              << "  Keyframe frames : " << data.frames.size() << std::endl;

    // Stage 1: vision-only camera poses (PnP with known landmarks).
    vio::VisionOnly vision;
    vision.SetData(data);
    vision.Run();
    const std::string out1 = data_dir + "/vio_estimated_tum.txt";
    vision.SaveTrajectoryTUM(out1);
    std::cout << "\n[Stage 1] camera poses (vision-only)\n"
              << "  Estimated poses : " << vision.trajectory().size() << "\n"
              << "  Trajectory saved: " << AbsolutePath(out1) << std::endl;

    // Stage 2: use the stage-1 body poses as measurements to estimate the IMU
    // state and the camera-IMU time offset td (td shifts by the pose rate).
    Param params;
    const auto cam_body = vision.BodyTrajectory(params.R_bc, params.t_bc);

    vio::EkfPoseRate ekf;
    const auto body_ekf = ekf.Run(data.imu, cam_body, td_init);
    SavePosesTUM(data_dir + "/vio_stage2_ekf_tum.txt", body_ekf);
    std::cout << "\n[Stage 2a] IMU state + td  (loosely-coupled EKF, pose-rate)\n"
              << "  Estimated td    : " << ekf.td() << " s  (+/- " << ekf.td_sigma()
              << "),  td_init = " << td_init << "\n"
              << "  Gyro bias       : " << ekf.gyro_bias().transpose() << "\n"
              << "  Accel bias      : " << ekf.acc_bias().transpose() << std::endl;

    vio::BaPoseRate ba;
    const auto body_ba = ba.Run(data.imu, cam_body, td_init);
    SavePosesTUM(data_dir + "/vio_stage2_ba_tum.txt", body_ba);
    std::cout << "\n[Stage 2b] IMU state + td  (factor-graph optimization, pose-rate)\n"
              << "  Estimated td    : " << ba.td() << " s,  td_init = " << td_init << "\n"
              << "  Gyro bias       : " << ba.gyro_bias().transpose() << "\n"
              << "  Accel bias      : " << ba.acc_bias().transpose() << std::endl;

    return 0;
}
