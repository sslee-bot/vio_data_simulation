#!/usr/bin/env python3
# Verify the decoupled Stage-2 time-offset (td) estimation.
#
# For each injected td, this runs the simulator (data_gen <td>) and the estimator
# (vio_estimator), parses the recovered td, and plots injected vs. estimated td for
# both the clean and noisy IMU streams. A perfect estimator lies on the y = x line.
#
#   python3 python_tool/draw_td_sweep.py

import os
import re
import subprocess

import numpy as np
import matplotlib.pyplot as plt

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
injected = [-0.020, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.020]

_td_re = re.compile(r'Estimated td\s*:\s*([-0-9.eE]+)')


def estimate_td(td_true, noisy):
    subprocess.run(['./data_gen', str(td_true)], cwd=bin_dir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    cmd = ['./vio_estimator', '.', '0'] + (['noisy'] if noisy else [])
    out = subprocess.run(cmd, cwd=bin_dir, capture_output=True, text=True, check=True).stdout
    return float(_td_re.search(out).group(1))


clean = [estimate_td(t, False) for t in injected]
noisy = [estimate_td(t, True) for t in injected]

inj = np.array(injected) * 1000.0   # ms
clean = np.array(clean) * 1000.0
noisy = np.array(noisy) * 1000.0

fig, (ax, axe) = plt.subplots(1, 2, figsize=(11, 4.5))

lim = [inj.min() - 2, inj.max() + 2]
ax.plot(lim, lim, 'k--', linewidth=1, label='ideal (y = x)')
ax.plot(inj, clean, 'o-', label='clean IMU')
ax.plot(inj, noisy, 's-', label='noisy IMU')
ax.set_xlabel('injected td [ms]')
ax.set_ylabel('estimated td [ms]')
ax.set_title('Decoupled Stage-2 td estimation')
ax.legend()
ax.grid(True, alpha=0.3)

axe.axhline(0, color='k', linewidth=1)
axe.plot(inj, clean - inj, 'o-', label='clean IMU')
axe.plot(inj, noisy - inj, 's-', label='noisy IMU')
axe.set_xlabel('injected td [ms]')
axe.set_ylabel('estimation error [ms]')
axe.set_title('td error (estimated - injected)')
axe.legend()
axe.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(bin_dir, 'td_sweep.png')
plt.savefig(out, dpi=120)
print('saved', out)
plt.show()
