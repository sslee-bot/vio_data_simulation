#ifndef VIO_ESTIMATOR_DATA_LOADER_H
#define VIO_ESTIMATOR_DATA_LOADER_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include "../src/imu.h"   // MotionData

namespace vio {

// One camera frame's feature observations, parsed from keyframe/all_points_<n>.txt.
// Each row of that file is: x y z w u v
//   -> landmark world position (x, y, z) and normalized image observation (u, v).
struct FrameObservations {
    int frame_id = -1;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> landmarks_w; // ground-truth 3D position
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> obs_norm;    // normalized image obs (u, v)
};

// All inputs the estimator consumes, loaded from the simulator's text output.
struct SimData {
    std::vector<MotionData> imu;           // noisy IMU stream (imu_pose_noise.txt)
    std::vector<MotionData> cam;           // camera poses (cam_pose.txt) — used for reference / evaluation
    std::vector<FrameObservations> frames; // per-camera-frame feature observations (keyframe/all_points_*.txt)
};

// Load IMU + camera poses + per-frame observations from a data directory
// (the simulator's bin/ output). Returns false if a required file is missing.
bool LoadSimData(const std::string& data_dir, SimData& out);

// Parse a single keyframe/all_points_<n>.txt file (x y z w u v per line).
bool LoadFrameObservations(const std::string& path, FrameObservations& out);

} // namespace vio

#endif // VIO_ESTIMATOR_DATA_LOADER_H
