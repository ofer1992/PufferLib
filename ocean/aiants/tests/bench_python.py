"""Turns/sec of the official Python engine (engine work only, bot time excluded).

Usage: python3 bench_python.py AICHALLENGE_ANTS_DIR MAP [games] [turns]
Random orders per ant, same engine loop as ref_runner.py.
"""
import random
import sys
import time

sys.path.insert(0, sys.argv[1])
from ants import Ants  # noqa: E402

text = open(sys.argv[2]).read()
games = int(sys.argv[3]) if len(sys.argv) > 3 else 20
turns = int(sys.argv[4]) if len(sys.argv) > 4 else 500
rng = random.Random(0)
total_turns, engine_time, ants_seen = 0, 0.0, 0
for k in range(games):
    t0 = time.perf_counter()
    g = Ants({'map': text, 'turns': turns, 'loadtime': 1, 'turntime': 1, 'viewradius2': 77,
              'attackradius2': 5, 'spawnradius2': 1, 'engine_seed': k, 'player_seed': 1})
    g.start_game()
    engine_time += time.perf_counter() - t0
    for turn in range(1, turns + 1):
        t0 = time.perf_counter()
        for b in range(g.num_players):
            if g.is_alive(b):
                g.get_player_state(b)
        g.start_turn()
        bl = [b for b in range(g.num_players) if g.is_alive(b)]
        random.shuffle(bl)
        over = g.game_over()
        engine_time += time.perf_counter() - t0
        orders = [['o %d %d %s' % (l[0], l[1], rng.choice('nesw-'))
                   for l, a in g.current_ants.items() if a.owner == b]
                  for b in range(g.num_players)]
        orders = [[o for o in os if not o.endswith('-')] for os in orders]
        t0 = time.perf_counter()
        if not over:
            for b in range(g.num_players):
                if g.is_alive(b):
                    g.do_moves(b, orders[b])
        g.finish_turn()
        over = g.game_over()
        engine_time += time.perf_counter() - t0
        total_turns += 1
        ants_seen += len(g.current_ants)
        if over:
            break
print('python engine: %.0f turns/s (%d turns, mean ants %.1f)'
      % (total_turns / engine_time, total_turns, ants_seen / total_turns))
