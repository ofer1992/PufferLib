"""Proof-of-concept PPO for aiants on CPU: a small fully-convolutional gridnet.

The learner plays agent slot 0 of each game; slot 1 is a scripted bot
(random / gatherer / raider) seeing only its own fog-of-war view.

  python3 ocean/aiants/train/train.py --steps 2_000_000 --run runs/poc
  python3 ocean/aiants/train/train.py --baselines      # bot-vs-bot table
"""
import argparse
import csv
import ctypes
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LIB = os.path.join(ROOT, 'build', 'libaiants.so')
BOTS = ['random', 'gatherer', 'raider']
CHANNELS, GLOBALS = 9, 4
GLOBAL_SCALE = np.array([255.0, 10.0, 50.0, 1.0], dtype=np.float32)


def build_lib():
    src = os.path.join(ROOT, 'ocean', 'aiants', 'train', 'aiants_vec.c')
    deps = [src] + [os.path.join(ROOT, 'ocean', 'aiants', f) for f in ('aiants.h', 'ants_engine.h')]
    if os.path.exists(LIB) and all(os.path.getmtime(LIB) > os.path.getmtime(d) for d in deps):
        return
    ray = os.path.join(ROOT, 'raylib-5.5_linux_amd64')
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    subprocess.check_call(['clang', '-O3', '-march=native', '-shared', '-fPIC', '-fopenmp',
        '-Wno-unused-function', '-I' + ray + '/include', '-I' + ROOT + '/src',
        '-I' + ROOT + '/vendor', src, ray + '/lib/libraylib.a', '-lGL', '-lm', '-lpthread',
        '-ldl', '-o', LIB])


class Vec:
    """N games; numpy views over the learner's buffers."""

    def __init__(self, n, seed, opponents, turns=500, rewards=(1.0, 0.1, 0.01, 0.01)):
        build_lib()
        lib = self.lib = ctypes.CDLL(LIB)
        lib.vec_create.restype = ctypes.c_void_p
        lib.vec_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int] \
            + [ctypes.c_float] * 4
        for name in ('vec_ep_outcome', 'vec_ep_length', 'vec_ep_food'):
            getattr(lib, name).restype = ctypes.POINTER(ctypes.c_float)
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        for name in ('vec_reset', 'vec_step'):
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        lib.vec_bind.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
        lib.vec_set_opponent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.vec_learner_bot.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.vec_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.grid = lib.vec_grid()
        self.cells = self.grid * self.grid
        self.n = n
        maps = os.path.join(ROOT, 'resources', 'aiants', 'maps').encode()
        self.h = ctypes.c_void_p(lib.vec_create(n, seed, maps, turns, *rewards))
        self.obs = np.zeros((n, lib.vec_obs_size()), dtype=np.uint8)
        self.mask = np.zeros((n, self.cells * 5), dtype=np.uint8)
        self.actions = np.zeros((n, self.cells), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.terminals = np.zeros(n, dtype=np.float32)
        lib.vec_bind(self.h, *(a.ctypes.data for a in
            (self.obs, self.mask, self.actions, self.rewards, self.terminals)))
        self.outcome = np.ctypeslib.as_array(lib.vec_ep_outcome(self.h), (n,))
        self.length = np.ctypeslib.as_array(lib.vec_ep_length(self.h), (n,))
        self.food = np.ctypeslib.as_array(lib.vec_ep_food(self.h), (n,))
        self.opponents = list(opponents)
        for i in range(n):
            lib.vec_set_opponent(self.h, i, self.opponents[i])
        lib.vec_reset(self.h)

    def step(self, actions=None, learner_bot=None):
        if learner_bot is not None:
            self.lib.vec_learner_bot(self.h, learner_bot)
        else:
            self.actions[:] = actions
        self.lib.vec_step(self.h)

    def snapshot(self, i):
        out = np.zeros(self.cells, dtype=np.uint8)
        rc = self.lib.vec_snapshot(self.h, i, out.ctypes.data)
        rows, cols = rc // 1000, rc % 1000
        return out[:rows * cols].reshape(rows, cols)


def wrap(x):
    """Circular pad by 1 (the map is a torus). Much faster on CPU than
    padding_mode='circular'."""
    x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], 2)
    return torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], 3)


class WrapConv(nn.Conv2d):
    def __init__(self, i, o):
        super().__init__(i, o, 3, padding=0)

    def forward(self, x):
        return super().forward(wrap(x))


class GridNet(nn.Module):
    """Fully convolutional gridnet on the torus. Space-to-depth folds each 2x2
    block of cells into channels, so all convs run at 16x16 (plus an 8x8
    branch with global context); pixel-shuffle unfolds 5 logits per cell.
    Value from pooled features."""

    def __init__(self, c=32, blocks=3):
        super().__init__()
        self.stem = WrapConv(4 * (CHANNELS + GLOBALS), c)
        self.mid = nn.ModuleList([WrapConv(c, c) for _ in range(blocks)])
        self.down = nn.Conv2d(c, c, 2, stride=2)
        self.low = WrapConv(c, c)
        self.ctx = nn.Linear(c, c)
        self.head = nn.Conv2d(c, 4 * 5, 1)
        self.value = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        b = F.relu(self.stem(F.pixel_unshuffle(x, 2)))
        for m in self.mid:
            b = F.relu(m(b)) + b
        c = F.relu(self.low(F.relu(self.down(b))))
        c = c + self.ctx(c.mean((2, 3)))[:, :, None, None]
        b = b + F.interpolate(c, scale_factor=2)
        value = self.value(torch.cat([b.mean((2, 3)), b.amax((2, 3))], 1)).squeeze(1)
        return F.pixel_shuffle(self.head(b), 2), value


def preprocess(obs, grid):
    n = obs.shape[0]
    planes = torch.from_numpy(obs[:, :CHANNELS * grid * grid]).view(n, CHANNELS, grid, grid)
    glob = torch.from_numpy(obs[:, CHANNELS * grid * grid:].astype(np.float32) / GLOBAL_SCALE)
    x = torch.cat([planes.float(), glob[:, :, None, None].expand(n, GLOBALS, grid, grid)], 1)
    return x.contiguous(memory_format=torch.channels_last)


def dist(net, x, mask):
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits, value = net(x)
    n = logits.shape[0]
    logits = logits.float().permute(0, 2, 3, 1).reshape(n, -1, 5)
    logits = logits.masked_fill(~mask, -1e9)
    return torch.log_softmax(logits, -1), value.float()


def sample(logp):
    """Gumbel-max: same distribution as Categorical(logits).sample(), ~8x faster."""
    return (logp - torch.log(-torch.log(torch.rand_like(logp)))).argmax(-1)


def evaluate(net, grid, games_per_bot, seed, greedy=False):
    """Win/draw/loss of the learner vs each bot over fresh games."""
    res = {}
    for k, bot in enumerate(BOTS):
        v = Vec(games_per_bot, seed + 1000 * k, [k] * games_per_bot)
        done = np.zeros(v.n, dtype=bool)
        out = np.zeros(v.n)
        length = np.zeros(v.n)
        while not done.all():
            with torch.no_grad():
                mask = torch.from_numpy(v.mask).view(v.n, -1, 5).bool()
                logp, _ = dist(net, preprocess(v.obs, grid), mask)
                a = logp.argmax(-1) if greedy else sample(logp)
            v.step(a.numpy())
            fin = (v.terminals > 0) & ~done
            out[fin] = v.outcome[fin]
            length[fin] = v.length[fin]
            done |= fin
        res[bot] = {'win': float((out > 0).mean()), 'draw': float((out == 0).mean()),
                    'loss': float((out < 0).mean()), 'length': float(length.mean())}
    return res


def baselines(games):
    print('learner-side bot vs opponent bot (%d games each): win / draw / loss' % games)
    for a, name_a in enumerate(BOTS):
        for b, name_b in enumerate(BOTS):
            v = Vec(games, 99 + a * 10 + b, [b] * games)
            done = np.zeros(games, dtype=bool)
            out = np.zeros(games)
            while not done.all():
                v.step(learner_bot=a)
                fin = (v.terminals > 0) & ~done
                out[fin] = v.outcome[fin]
                done |= fin
            print('  %-9s vs %-9s %.2f / %.2f / %.2f' % (name_a, name_b, (out > 0).mean(),
                (out == 0).mean(), (out < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=float, default=2e6)
    ap.add_argument('--envs', type=int, default=128)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--minibatches', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.003)
    ap.add_argument('--vf', type=float, default=0.5)
    ap.add_argument('--opponents', default='random,gatherer,raider',
        help='bots assigned round-robin to training games')
    ap.add_argument('--eval_every', type=int, default=80, help='updates between evals')
    ap.add_argument('--eval_games', type=int, default=128,
        help='per bot; batch 64 hits a ~4x slower oneDNN path on CPU')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--run', default='runs/poc')
    ap.add_argument('--resume', default='')
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--reward_win', type=float, default=1.0)
    ap.add_argument('--reward_score', type=float, default=0.1)
    ap.add_argument('--reward_food', type=float, default=0.1)
    ap.add_argument('--reward_kill', type=float, default=0.02)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.baselines:
        baselines(a.eval_games)
        return
    os.makedirs(a.run, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.run, 'args.json'), 'w'), indent=1)

    opp = [BOTS.index(b) for b in a.opponents.split(',')]
    rewards = (a.reward_win, a.reward_score, a.reward_food, a.reward_kill)
    v = Vec(a.envs, a.seed, [opp[i % len(opp)] for i in range(a.envs)], rewards=rewards)
    G, N, T = v.grid, a.envs, a.horizon
    net = GridNet().to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    step = 0
    if a.resume:
        ck = torch.load(a.resume)
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
    print('params', sum(p.numel() for p in net.parameters()))

    obs_buf = np.zeros((T, N, v.obs.shape[1]), dtype=np.uint8)
    mask_buf = np.zeros((T, N, G * G * 5), dtype=np.uint8)
    act_buf = torch.zeros((T, N, G * G), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T + 1, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))

    log_f = open(os.path.join(a.run, 'log.csv'), 'a', newline='')
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(['step', 'time', 'sps', 'pg_loss', 'v_loss', 'entropy_per_ant', 'clipfrac',
            'games', 'food_per_game'] + ['train_win_%s' % b for b in BOTS]
            + ['eval_%s_%s' % (b, k) for b in BOTS for k in ('win', 'draw', 'loss', 'length')])
    recent = {k: [] for k in range(len(BOTS))}
    recent_food = []
    opp_arr = np.array(v.opponents)
    t_start = time.time()
    update = 0
    while step < a.steps:
        t0 = time.time()
        net.eval()
        for t in range(T):
            obs_buf[t] = v.obs
            mask_buf[t] = v.mask
            with torch.no_grad():
                mask = torch.from_numpy(v.mask).view(N, -1, 5).bool()
                logp, value = dist(net, preprocess(v.obs, G), mask)
                act = sample(logp)
                logp_buf[t] = logp.gather(-1, act[..., None]).squeeze(-1).sum(-1)
            val_buf[t] = value
            act_buf[t] = act
            v.step(act.numpy())
            rew_buf[t] = torch.from_numpy(v.rewards)
            done_buf[t] = torch.from_numpy(v.terminals)
            for i in np.nonzero(v.terminals)[0]:
                recent[opp_arr[i]].append(float(v.outcome[i] > 0))
                recent_food.append(float(v.food[i]))
        with torch.no_grad():
            mask = torch.from_numpy(v.mask).view(N, -1, 5).bool()
            _, val_buf[T] = dist(net, preprocess(v.obs, G), mask)
        adv = torch.zeros((T, N))
        last = torch.zeros(N)
        for t in reversed(range(T)):
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + a.gamma * val_buf[t + 1] * nonterm - val_buf[t]
            last = delta + a.gamma * a.lam * nonterm * last
            adv[t] = last
        ret = adv + val_buf[:T]

        t_roll = time.time() - t0
        net.train()
        B = T * N
        b_obs, b_mask = obs_buf.reshape(B, -1), mask_buf.reshape(B, -1)
        b_act, b_logp = act_buf.reshape(B, -1), logp_buf.reshape(B)
        b_adv, b_ret = adv.reshape(B), ret.reshape(B)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        stats = []
        mb = B // a.minibatches
        for _ in range(a.epochs):
            perm = torch.randperm(B)
            for k in range(a.minibatches):
                idx = perm[k * mb:(k + 1) * mb].numpy()
                mask = torch.from_numpy(b_mask[idx]).view(len(idx), -1, 5).bool()
                logp, value = dist(net, preprocess(b_obs[idx], G), mask)
                new_logp = logp.gather(-1, b_act[idx][..., None]).squeeze(-1).sum(-1)
                ants = (mask[..., 1:].any(-1)).float().sum(-1).clamp(min=1)
                ent = (-(logp.exp() * logp).sum(-1)).sum(-1) / ants
                ratio = (new_logp - b_logp[idx]).exp()
                adv_mb = b_adv[idx]
                pg = -torch.min(ratio * adv_mb,
                    ratio.clamp(1 - a.clip, 1 + a.clip) * adv_mb).mean()
                v_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()
                loss = pg + a.vf * v_loss - a.ent * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats.append([pg.item(), v_loss.item(), ent.mean().item(),
                    ((ratio - 1).abs() > a.clip).float().mean().item()])
        step += B
        update += 1
        sps = B / (time.time() - t0)
        s = np.mean(stats, 0)
        games = sum(len(r) for r in recent.values())
        train_win = [np.mean(recent[k][-200:]) if recent[k] else float('nan')
            for k in range(len(BOTS))]
        food = np.mean(recent_food[-200:]) if recent_food else float('nan')
        row = [step, round(time.time() - t_start, 1), round(sps, 1), *np.round(s, 4), games,
            round(food, 2), *np.round(train_win, 3)]
        if update % a.eval_every == 0 or step >= a.steps:
            res = evaluate(net, G, a.eval_games, seed=12345 + update)
            row += [round(res[b][k], 3) for b in BOTS for k in ('win', 'draw', 'loss', 'length')]
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'step': step},
                os.path.join(a.run, 'model.pt'))
            print('eval @%d: %s' % (step, '  '.join('%s %.2f/%.2f/%.2f' % (b, res[b]['win'],
                res[b]['draw'], res[b]['loss']) for b in BOTS)), flush=True)
        log.writerow(row)
        log_f.flush()
        if update % 10 == 0:
            print('rollout %.2fs train %.2fs' % (t_roll, time.time() - t0 - t_roll))
            print('step %d  sps %.0f  pg %.3f  v %.4f  ent/ant %.3f  clip %.3f  food %.1f  win %s'
                % (step, sps, s[0], s[1], s[2], s[3], food,
                   ' '.join('%s %.2f' % (b, w) for b, w in zip(BOTS, train_win))), flush=True)


if __name__ == '__main__':
    main()
