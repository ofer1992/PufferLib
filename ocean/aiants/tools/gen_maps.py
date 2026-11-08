"""Generate small symmetric maps with the official aichallenge map generator.

The official mapgen/cell_maze.py rejects maps under 900 squares per player and
hills under 20 steps apart. This wrapper runs the same generator and the same
symmetry / accessibility checks, relaxing only those two limits.

Usage: python3 gen_maps.py AICHALLENGE_ANTS_DIR OUT_DIR COUNT [rows] [players] [min_hill_dist]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(sys.argv[1], 'mapgen'))
from cell_maze import CellMazeMap  # noqa: E402

out_dir, count = sys.argv[2], int(sys.argv[3])
size = int(sys.argv[4]) if len(sys.argv) > 4 else 32
players = int(sys.argv[5]) if len(sys.argv) > 5 else 2
min_dist = int(sys.argv[6]) if len(sys.argv) > 6 else 12
os.makedirs(out_dir, exist_ok=True)


def check(m):
    if m.get_map_symmetry() is None:
        return 'asymmetric'
    rows, cols = len(m.map), len(m.map[0])
    if (rows, cols) != (size, size):
        return 'size'
    hills = [(r, c) for r in range(rows) for c in range(cols) if 0 <= m.map[r][c] < 10]
    if len(hills) != players:
        return 'hills'
    for h in hills:
        dists = dict(m.get_distances(h, hills, (rows, cols)))
        for h2 in hills:
            if h2 == h:
                continue
            if m.euclidean_distance2(h, h2, (rows, cols)) <= 25 or dists[h2] < min_dist:
                return 'hills too close'
    if len(m.section(0)) > 1:
        return 'not accessible'
    return None


made, seed = 0, 0
while made < count:
    seed += 1
    random.seed(seed)
    m = CellMazeMap({'players': players, 'area': size * size, 'aspect_ratio': 1.0,
                     'hills': 1, 'seed': seed})
    try:
        if m.generate() or check(m):
            continue
    except Exception:
        continue
    path = os.path.join(out_dir, 'cell_maze_%dx%d_p%02d_%03d.map' % (size, size, players, made))
    with open(path, 'w') as f:
        m.toText(f)
    made += 1
print('generated %d maps (%d seeds tried)' % (made, seed))
