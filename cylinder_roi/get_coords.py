import os.path as op
from itertools import product

import numpy as np
from paths import root_dir
from paths_funcs import glob_sub_dir

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No subdirectories found in root directory.")

for sub_dir in sub_dirs:
    coords = np.loadtxt(op.join(sub_dir, "mid.txt"), dtype=int, delimiter=",")

    dist = np.zeros(len(coords) ** 2, dtype=int)

    for idx, (coord1, coord2) in enumerate(product(coords, coords)):
        dist[idx] = np.linalg.norm(coord1 - coord2)

    dist = dist.reshape(len(coords), len(coords))

    anode = coords[np.argmin(np.sum(dist, axis=1))]
