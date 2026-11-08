"""Parity test: C engine vs the official Python engine, turn by turn.

Clones aichallenge/aichallenge at the pinned commit (once, into build/),
compiles parity_driver.c, then runs a matrix of maps x seeds x attack/food
modes x bots. Each case plays a game on the unmodified ants.py, replays the
same orders on the C engine, and requires identical per-turn dumps.

Usage (from repo root): python3 ocean/aiants/tests/run_parity.py [--quick] [-j N]
"""
import argparse
import glob
import os
import random
import subprocess
import sys
import tempfile
os.environ['ASAN_OPTIONS'] = 'detect_leaks=0'
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = 'https://github.com/aichallenge/aichallenge.git'
COMMIT = '4237971f22767ef7f439d297e4e6ad7c458415dc'
CACHE = os.path.join(ROOT, 'build', 'aichallenge')
DRIVER = os.path.join(ROOT, 'build', 'aiants_parity')
ANTS_DIR = os.path.join(CACHE, 'ants')


def setup():
    if not os.path.isdir(CACHE):
        subprocess.check_call(['git', 'clone', '-q', REPO, CACHE])
    subprocess.check_call(['git', '-C', CACHE, 'checkout', '-q', COMMIT])
    os.makedirs(os.path.dirname(DRIVER), exist_ok=True)
    src = os.path.join(HERE, 'parity_driver.c')
    # Prefer a sanitizer build; fall back to plain if the runtime is missing
    for cc in (['gcc', '-fsanitize=address,undefined'], ['clang', '-fsanitize=address,undefined'],
            ['cc']):
        r = subprocess.run(cc + ['-O2', '-g', '-Wno-unused-result', src, '-o', DRIVER],
            capture_output=True)
        if r.returncode == 0:
            return
    raise RuntimeError('could not compile parity_driver.c')


def run_case(case):
    name, map_path, kv = case
    args = ['%s=%s' % (k, v) for k, v in kv.items()]
    with tempfile.TemporaryDirectory() as d:
        orders, ref, out = (os.path.join(d, x) for x in ('o.txt', 'ref.txt', 'c.txt'))
        r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'ref_runner.py'),
            ANTS_DIR, map_path, orders, ref] + args, capture_output=True, text=True)
        if r.returncode != 0:
            return name, 'SKIP (python engine error: %s)' % r.stderr.strip().splitlines()[-1]
        with open(out, 'w') as f:
            r = subprocess.run([DRIVER, map_path, orders] + args, stdout=f,
                stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return name, 'FAIL (driver: %s)' % r.stderr.strip()[-300:]
        a, b = open(ref).read().splitlines(), open(out).read().splitlines()
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                turn = next((l for l in reversed(a[:i]) if l.startswith('T ')), 'T 0')
                return name, 'FAIL at %s line %d:\n  py: %s\n  c:  %s\n  args: %s' % (
                    turn, i, x, y, ' '.join(args))
        if len(a) != len(b):
            return name, 'FAIL (length %d vs %d)' % (len(a), len(b))
        return name, 'OK (%d turns, %s)' % (sum(l.startswith('T ') for l in a), a[-1][4:])


def cases(quick):
    rng = random.Random(1234)
    official = sorted(glob.glob(os.path.join(ANTS_DIR, 'maps', '*', '*.map')))
    by_players = {}
    for m in official:
        p = int(next(l for l in open(m) if l.startswith('players')).split()[1])
        by_players.setdefault(p, []).append(m)
    small = sorted(glob.glob(os.path.join(ROOT, 'resources', 'aiants', 'maps', '*.map')))
    out = []
    attacks = ['focus', 'damage', 'support', 'closest']
    foods = ['symmetric', 'sections', 'random', 'none']
    per_count = 1 if quick else 3
    for p, maps in sorted(by_players.items()):
        for m in rng.sample(maps, min(per_count, len(maps))):
            for k in range(1 if quick else 2):
                seed = rng.randrange(-2**40, 2**40)
                kv = {'seed': seed, 'turns': 120 if p > 4 else 200,
                      'attack': rng.choice(attacks), 'food': 'symmetric',
                      'bots': rng.choice(['greedy', 'greedy,random', 'random,greedy,idle']),
                      'bot_seed': rng.randrange(1000)}
                out.append(('%s seed=%d %s' % (os.path.basename(m), seed, kv['attack']), m, kv))
    for m in rng.sample(small, min(4 if quick else 20, len(small))):
        for attack in attacks:
            for food in (foods if not quick else ['symmetric']):
                seed = rng.randrange(2**31)
                kv = {'seed': seed, 'turns': 500, 'attack': attack, 'food': food,
                      'bots': rng.choice(['greedy', 'greedy,random']),
                      'bot_seed': rng.randrange(1000)}
                out.append(('%s seed=%d %s %s' % (os.path.basename(m), seed, attack, food), m, kv))
    # the RL env's exact setting: default rules on the small map pool
    for k in range(10 if quick else 100):
        m = rng.choice(small)
        seed = rng.randrange(2**47)
        kv = {'seed': seed, 'turns': 500, 'bots': rng.choice(['greedy', 'greedy,random']),
              'bot_seed': rng.randrange(1000)}
        out.append(('rl %s seed=%d' % (os.path.basename(m), seed), m, kv))
    # cutoff endings: idle players let food pile up / one side dominate
    for k, bots in enumerate(['idle', 'greedy,idle', 'farmer,idle', 'idle,farmer', 'farmer']):
        m = small[k % len(small)]
        out.append(('cutoff %s bots=%s' % (os.path.basename(m), bots), m,
            {'seed': k, 'turns': 500, 'bots': bots, 'bot_seed': k}))
    for k, pct in enumerate([0.6, 0.5, 0.4, 0.3]):
        m = small[(k + 5) % len(small)]
        out.append(('cutoff %s pct=%s' % (os.path.basename(m), pct), m,
            {'seed': k, 'turns': 500, 'bots': 'farmer,idle', 'bot_seed': k,
             'cutoff_percent': pct, 'cutoff_turn': 60}))
    tut = os.path.join(ANTS_DIR, 'maps', 'example', 'tutorial1.map')
    for seed in range(3 if quick else 10):
        out.append(('tutorial1 scenario seed=%d' % seed, tut,
            {'seed': seed, 'turns': 150, 'scenario': 1, 'bots': 'greedy'}))
        out.append(('tutorial1 radii seed=%d' % seed, tut,
            {'seed': seed, 'turns': 150, 'viewradius2': 55, 'attackradius2': 10,
             'spawnradius2': 2, 'cutoff_turn': 40, 'cutoff_percent': 0.6, 'bots': 'greedy'}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('-j', type=int, default=os.cpu_count())
    ap.add_argument('--filter', default='', help='only cases whose name contains this')
    a = ap.parse_args()
    setup()
    cs = [c for c in cases(a.quick) if a.filter in c[0]]
    fails = skips = 0
    with Pool(a.j) as pool:
        for name, res in pool.imap_unordered(run_case, cs):
            print('%-60s %s' % (name, res), flush=True)
            fails += res.startswith('FAIL')
            skips += res.startswith('SKIP')
    print('\n%d cases, %d failed, %d skipped' % (len(cs), fails, skips))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
