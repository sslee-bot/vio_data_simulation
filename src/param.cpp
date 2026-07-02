//
// Created by hyj on 17-6-22.
//

#include "param.h"

Param::Param()
{
    Eigen::Matrix3d R;   // Rotate the body-frame orientation to obtain the camera frame so it can see the landmarks; the columns express the camera-frame axes in the body frame.
    // The camera looks toward the inside of the trajectory while the feature points are outside; this is the configuration we use.
    R << 0, 0, -1,
            -1, 0, 0,
            0, 1, 0;
    R_bc = R;
    t_bc = Eigen::Vector3d(0.05,0.04,0.03);

}