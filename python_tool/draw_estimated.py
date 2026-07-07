#!/usr/bin/env python3
# Overlay the estimator's output trajectory against the ground-truth camera
# trajectory. Both files are in TUM format: timestamp tx ty tz qx qy qz qw.
#
# Run after generating data (bin/data_gen) and estimating (bin/vio_estimator):
#   python3 python_tool/draw_estimated.py

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')


def load_tum_xyz(path):
    data = np.loadtxt(path)          # ts tx ty tz qx qy qz qw
    return data[:, 1:4]


gt = load_tum_xyz(os.path.join(bin_dir, 'cam_pose_tum.txt'))
est = load_tum_xyz(os.path.join(bin_dir, 'vio_estimated_tum.txt'))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], label='ground truth (cam)')
ax.plot(est[:, 0], est[:, 1], est[:, 2], '--', label='estimated (vio)')
ax.plot([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], 'r.', label='start')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Estimated vs ground-truth camera trajectory')

out = os.path.join(bin_dir, 'trajectory_compare.png')
plt.savefig(out, dpi=120)
print('saved', out)
plt.show()
