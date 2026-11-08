"""Render one game of a trained policy vs a bot as a GIF.

  python3 ocean/aiants/train/replay.py runs/poc/model.pt --bot gatherer --out replay.gif

Cyan = learner, red = bot. Squares outlined = hills. White dots = food.
Darkened squares are outside the learner's view.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402

LAND, WATER, FOOD = (70, 60, 45), (25, 60, 120), (240, 240, 240)
TEAM = [(0, 200, 200), (220, 40, 40)]


def frame(snap, turn, scale=12):
    rows, cols = snap.shape
    img = Image.new('RGB', (cols * scale, rows * scale + 18), (6, 24, 24))
    d = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            code = int(snap[r, c])
            base = code & 7
            col = WATER if base == 1 else LAND
            if code & 32:
                col = tuple(x // 2 for x in col)
            x, y = c * scale, r * scale + 18
            d.rectangle([x, y, x + scale - 1, y + scale - 1], fill=col)
            if code & 8 or code & 16:
                d.rectangle([x, y, x + scale - 1, y + scale - 1],
                    outline=TEAM[0] if code & 8 else TEAM[1], width=2)
            if base == 2:
                d.ellipse([x + 4, y + 4, x + scale - 5, y + scale - 5], fill=FOOD)
            elif base in (3, 4):
                d.ellipse([x + 2, y + 2, x + scale - 3, y + scale - 3], fill=TEAM[base - 3])
    d.text((4, 3), 'turn %d   learner (cyan) vs bot (red)' % turn, fill=(230, 230, 230))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--bot', default='gatherer', choices=T.BOTS)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default='replay.gif')
    ap.add_argument('--every', type=int, default=1, help='keep every k-th turn')
    a = ap.parse_args()
    net = T.GridNet().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(a.model)['net'])
    net.eval()
    v = T.Vec(1, a.seed, [T.BOTS.index(a.bot)])
    frames, turn = [frame(v.snapshot(0), 0)], 0
    while True:
        with torch.no_grad():
            mask = torch.from_numpy(v.mask).view(1, -1, 5).bool()
            logp, _ = T.dist(net, T.preprocess(v.obs, v.grid), mask)
        snap_before = v.snapshot(0)
        v.step(T.sample(logp).numpy())
        turn += 1
        if v.terminals[0]:
            frames.append(frame(snap_before, turn))
            break
        if turn % a.every == 0:
            frames.append(frame(v.snapshot(0), turn))
    result = {1.0: 'learner wins', -1.0: 'bot wins', 0.0: 'draw'}[float(v.outcome[0])]
    frames += [frames[-1]] * 15
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print('%s after %d turns -> %s' % (result, turn, a.out))


if __name__ == '__main__':
    main()
