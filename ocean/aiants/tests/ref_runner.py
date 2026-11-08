"""Reference runner: plays a game on the official, unmodified ants.py.

Replicates the worker/engine.py turn loop (including its per-turn
random.shuffle of bots on the global RNG), drives scripted bots that read the
game state directly, and writes:
  - an orders file the C driver replays
  - a per-turn state dump (compared line by line against parity_driver)

Usage: python3 ref_runner.py AICHALLENGE_ANTS_DIR MAP ORDERS_OUT DUMP_OUT key=value...
Keys: seed, turns, attack, food, bots (comma list of random|greedy|idle),
      bot_seed, viewradius2, attackradius2, spawnradius2, cutoff_turn,
      cutoff_percent, scenario
"""
import random
import sys
from collections import deque

sys.path.insert(0, sys.argv[1])
from ants import Ants, AIM, LAND, FOOD, WATER  # noqa: E402

MASK64 = (1 << 64) - 1
DIRS = 'nesw'


def parse_kv(args):
    kv = {}
    for a in args:
        k, v = a.split('=', 1)
        kv[k] = v
    return kv


def closest_attack_py2(self):
    # ants.py's do_attack_closest kills while iterating dict.values(), which
    # raises on Python 3. Same code with the Python 2 list snapshot semantics.
    from collections import defaultdict
    ants_by_distance = {}
    for ant in list(self.current_ants.values()):
        dist_map = defaultdict(list)
        for enemy in self.nearby_ants(ant.loc, self.attackradius, ant.owner):
            dist_map[self.distance(ant.loc, enemy.loc)].append(enemy)
        ants_by_distance[ant] = dist_map
    ant_group = set()

    def find_enemy(ant, distance):
        for enemy in ants_by_distance[ant][distance]:
            if not enemy.killed and enemy not in ant_group:
                ant_group.add(enemy)
                find_enemy(enemy, distance)

    for distance in range(1, self.attackradius):
        for ant in list(self.current_ants.values()):
            if not ants_by_distance[ant] or ant.killed:
                continue
            ant_group = set([ant])
            find_enemy(ant, distance)
            if len(ant_group) > 1:
                for a in ant_group:
                    self.kill_ant(a)


class Bot:
    def __init__(self, kind, player, seed):
        self.kind = kind
        self.p = player
        self.rng = random.Random(seed)

    def moves(self, g):
        if self.kind == 'idle':
            return []
        mine = sorted(loc for loc, a in g.current_ants.items() if a.owner == self.p)
        lines = []
        taken = set()
        for loc in mine:
            r = self.rng.random()
            if self.kind == 'random' or r < 0.15:
                if self.rng.random() < 0.1:
                    continue
                d = self.rng.choice(DIRS)
            else:
                d = self.greedy_dir(g, loc, taken)
                if d is None:
                    continue
            lines.append('o %d %d %s' % (loc[0], loc[1], d))
            taken.add(g.destination(loc, AIM[d]))
        # junk orders exercise validation
        if self.rng.random() < 0.05 and mine:
            loc = self.rng.choice(mine)
            lines.append('o %d %d %s' % (loc[0], loc[1], self.rng.choice(DIRS)))
        if self.rng.random() < 0.05:
            lines.append('o %d %d %s' % (self.rng.randrange(-2, g.height + 2),
                self.rng.randrange(-2, g.width + 2), self.rng.choice(DIRS)))
        return lines

    def greedy_dir(self, g, start, taken):
        # BFS to nearest food / enemy hill / enemy ant within 12 steps
        seen = {start: None}
        q = deque([start])
        steps = 0
        while q and steps < 400:
            cur = q.popleft()
            steps += 1
            if cur != start:
                v = g.map[cur[0]][cur[1]]
                hill = g.hills.get(cur)
                if self.kind == 'farmer':
                    hit = v == FOOD
                else:
                    hit = (v == FOOD or (v >= 0 and v != self.p)
                           or (hill and hill.owner != self.p and hill.killed_by is None))
                if hit:
                    while seen[cur][0] != start:
                        cur = seen[cur][0]
                    d = seen[cur][1]
                    if g.destination(start, AIM[d]) in taken:
                        return None
                    return d
            dirs = list(DIRS)
            self.rng.shuffle(dirs)
            for d in dirs:
                n = g.destination(cur, AIM[d])
                if n in seen or g.map[n[0]][n[1]] == WATER:
                    continue
                if self.kind == 'farmer' and any(g.distance(n, h.loc) <= 16
                        for h in g.hills.values() if h.owner != self.p):
                    continue
                if g.map[n[0]][n[1]] == FOOD and cur == start:
                    seen[n] = (cur, d)
                    q.append(n)
                    continue
                seen[n] = (cur, d)
                q.append(n)
        d = self.rng.choice(DIRS)
        return None if g.destination(start, AIM[d]) in taken else d


def dump(g, turn, renders, w):
    w('T %d\n' % turn)
    for b in sorted(renders):
        w('U %d\n%s/U\n' % (b, renders[b]))
    w('S %s\n' % ' '.join(map(str, g.score)))
    w('B %s\n' % ' '.join(map(str, g.bonus)))
    w('H %s\n' % ' '.join(map(str, g.hive_food)))
    w('X %d\n' % int(g.food_extra * g.food_turn))
    w('C %d %d %d %s\n' % (g.cutoff_bot, g.cutoff_turns, int(g.hill_kill), g.cutoff))
    w('W %d %d\n' % (g.winning_turn, g.ranking_turn))
    st = random.getstate()[1]
    w('R %d\n' % (sum((i + 1) * v for i, v in enumerate(st)) & MASK64))
    for hill in g.hills.values():
        w('h %d %d %d %d %d %d %d\n' % (hill.loc[0], hill.loc[1], hill.owner,
            -1 if hill.killed_by is None else hill.killed_by, hill.last_touched,
            -1 if hill.end_turn is None else hill.end_turn, int(hill.raze_points)))
    for ant in g.current_ants.values():
        w('a %d %d %d\n' % (ant.loc[0], ant.loc[1], ant.owner))
    for r, c in sorted(g.current_food):
        w('f %d %d\n' % (r, c))
    for (r, c), n in sorted(g.pending_food.items()):
        if n:
            w('p %d %d %d\n' % (r, c, n))
    for ant in sorted(g.killed_ants, key=lambda a: (a.loc, a.owner)):
        w('d %d %d %d\n' % (ant.loc[0], ant.loc[1], ant.owner))
    W = g.width
    for p in range(g.num_players):
        vis = [r * W + c for r in range(g.height) for c in range(W) if g.vision[p][r][c] > 0]
        rev = [r * W + c for r in range(g.height) for c in range(W) if g.revealed[p][r][c]]
        w('V %d %d %d %d %d %d %d\n' % (p, len(vis), sum(vis), sum(i * i for i in vis) & MASK64,
            len(rev), sum(rev), sum(i * i for i in rev) & MASK64))
        w('Q %d %s\n' % (p, ' '.join(str(-1 if s is None else s)
            for s in g.switch[p][:g.num_players])))


def main():
    ants_dir, map_path, orders_path, dump_path = sys.argv[1:5]
    kv = parse_kv(sys.argv[5:])
    turns = int(kv.get('turns', 200))
    opts = {
        'map': open(map_path).read(),
        'turns': turns, 'loadtime': 3000, 'turntime': 1000,
        'viewradius2': int(kv.get('viewradius2', 77)),
        'attackradius2': int(kv.get('attackradius2', 5)),
        'spawnradius2': int(kv.get('spawnradius2', 1)),
        'engine_seed': int(kv.get('seed', 0)), 'player_seed': 1,
        'attack': kv.get('attack', 'focus'), 'food': kv.get('food', 'symmetric'),
        'food_rate': (5, 11), 'food_turn': (19, 37), 'food_start': (75, 175),
        'food_visible': (3, 5),
        'cutoff_turn': int(kv.get('cutoff_turn', 150)),
        'cutoff_percent': float(kv.get('cutoff_percent', 0.85)),
        'scenario': kv.get('scenario', '0') == '1',
    }
    g = Ants(opts)
    if opts['attack'] == 'closest':
        g.do_attack = closest_attack_py2.__get__(g)
    kinds = kv.get('bots', 'greedy').split(',')
    bot_seed = int(kv.get('bot_seed', 0))
    bots = [Bot(kinds[p % len(kinds)], p, bot_seed * 1000 + p) for p in range(g.num_players)]
    orders_f = open(orders_path, 'w')
    out = open(dump_path, 'w')
    for turn in range(turns + 1):
        if turn == 0:
            g.start_game()
        renders = {}
        for b in range(g.num_players):
            if g.is_alive(b) and turn > 0:
                renders[b] = g.get_player_state(b)
        if turn > 0:
            g.start_turn()
        bot_list = [b for b in range(g.num_players) if g.is_alive(b)]
        random.shuffle(bot_list)
        bot_alive = [g.is_alive(b) for b in range(g.num_players)]
        if turn > 0 and not g.game_over():
            for b in range(g.num_players):
                if g.is_alive(b):
                    lines = bots[b].moves(g)
                    orders_f.write('O %d %d %d\n' % (turn, b, len(lines)))
                    for line in lines:
                        orders_f.write(line + '\n')
                    g.do_moves(b, lines)
        if turn > 0:
            g.finish_turn()
        for b in range(g.num_players):
            if bot_alive[b] and not g.is_alive(b):
                g.get_player_state(b)
        over = g.game_over()
        if over or turn == turns:
            g.finish_game()
        dump(g, turn, renders, out.write)
        if over:
            break
    out.write('END %s\n' % g.cutoff)


if __name__ == '__main__':
    main()
