
Forked repository for purposes:
- Simplified test (no real sensor data & feature depth estimation)
- Analysis of Computation time along with parameters
- Effectiveness of cam-IMU time offset estimation

Strategies which are considered (not fully implemented yet):
- `TODO: write`

> The content below is from the original author.

---

# imusim
IMU and camera data simulation, used for testing VIO algorithms. Feedback on any code issues is welcome: heyijia_2013@163.com.

We also provide a ROS version in the `ros_version` branch.

![demo pic](https://github.com/HeYijia/vio_data_simulation/blob/master/bin/demo.png?raw=true)

## Coordinate frames
- **B**ody frame: the IMU frame.

- **C**am frame: the camera frame.

- **W**orld frame: the pose of the first IMU frame.

- **N**avigation frame: NED (North-East-Down) or ENU (East-North-Up). This code uses ENU, in which the gravity vector is $(0,0,-9.81)$.

Currently the IMU z-axis points up, the body performs an elliptical motion in the xy-plane, a sinusoidal motion along the z-axis, and the x-axis points radially outward along the circle. The extrinsic transform Tbc rotates the camera frame so that the camera faces the feature points.

## Code structure
main/gener_alldata.cpp: generates IMU data, the camera trajectory, feature-point pixel coordinates, and the 3D coordinates of feature points.

src/param.h: IMU noise parameters, IMU frequency, camera intrinsics, etc.

src/camera_model.cpp: the camera model, taken from svo. This file has been removed from the current code.

python_tool/: visualization tools. draw_points.py dynamically plots the camera trajectory and the observed feature points. On Ubuntu no extra installation is needed; on Windows you must install Python, matplotlib, and other dependencies.

## Data storage format
### Feature points
> x, y, z, 1, u, v

The order in which each feature appears in the file is its unique id, which can be used to look up feature matches.

### IMU data
> timestamp (1), imu quaternion (4), imu position (3), imu gyro (3), imu acc (3)

### Cam data
> timestamp (1), cam quaternion (4), cam position (3), imu gyro (3), imu acc (3)

Note: because the IMU and camera use the same function for storage, the camera file also stores some gyro/acc data. These are unused and stored redundantly.
