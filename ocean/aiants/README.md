# AI Ants

C port of the 2011 Google AI Challenge "Ants" game for PufferLib.
(Not to be confused with `ocean/ants`, an unrelated pheromone foraging env.)

## Files

- `ants_engine.h`: the game engine. Full rules, any player count (2-10),
  any map. No PufferLib/raylib dependency.
- `aiants.h`: the PufferLib env (1v1 gridnet). Only touches the engine
  through its public struct and functions.
- `config/aiants.ini`: env and training config.
- `resources/aiants/maps/`: 64 symmetric 32x32 2-player maps.
- `tools/gen_maps.py`: regenerates them with the official map generator.
- `tests/`: parity tests against the original engine, env checks, benchmarks.

## Source of truth

[aichallenge/aichallenge](https://github.com/aichallenge/aichallenge)
@ `4237971` (Feb 2012): `ants/ants.py` (rules) and `worker/engine.py`
(turn loop). Defaults follow `ants/playgame.py`:
- focus attack, symmetric food;
- radius² 77 / 5 / 1 for view / attack / gather;
- food rate 5-11 per 19-37 turns, randomized from the seed;
- 3-5 visible starting food;
- cutoff after 150 turns at 85%.

The engine is bit-exact with that engine running on CPython 3.8+: the same
map, `engine_seed` and orders give the same game. That needed:

- Python's `random` ported exactly: MT19937, `seed(int)`, `random()`,
  `_randbelow`, `shuffle`, `choice`.
- The engine loop's `random.shuffle(bot_list)` every turn, which uses the
  same global RNG as the food rules.
- Python dict/set iteration order wherever it reaches the RNG or the rules:
  hill order for spawn tie-breaks, insertion order of `current_ants`
  (cutoff counting), and CPython's tuple hash plus set probing for food-set
  ordering (matters with 3+ players).
- Original quirks kept as-is:
  - `support` attack ignores friendly ants.
  - `closest` attack never checks the outer radius.
  - Symmetric food sets can lie on water.
  - Food can sit under an ant spawned on a hill.
  - Hive food is counted once per alive hill in the cutoff.

Exact random streams from the 2011 servers (Python 2, hash-ordered dicts)
are not reproducible; the rules are identical.

Note: `do_attack_closest` raises on Python 3 (it deletes from a dict while
iterating it). The parity tests run that mode with the same code plus a
`list()` snapshot (Python 2 semantics), which is what the C engine does.

## Engine API

```c
Ants* g = ants_new(max_cells, max_players);   // allocate once
ants_init(g, &map, ants_default_config(), seed);
ants_start(g);                                // turn 0
while (!g->done) {
    ants_begin_turn(g, NULL, 0);              // optional: bot input text
    ants_order(g, player, row, col, dir);     // validated like the original
    ants_end_turn(g);
}
// g->score, g->cutoff, g->vision, g->revealed, g->switch_, ...
```

`ants_render(g, p, buf, n)` produces the exact text the original sends a
bot each turn, so a bot-protocol bridge would be straightforward.

## RL interface (`aiants.h`)

- One agent per player, 1v1. Both agent slots are policy slots, so
  selfplay works, and sides are randomized each episode.
- **Observation:** `uint8`, 9 planes of 32x32, channel-major, then 4
  globals.
  - Planes: visible now, ever seen, known water, visible food, own ants,
    visible enemy ants, own hills, last known enemy hills, on-map.
  - Globals: turn progress, hive food, own ant count, own hill count.
  - Maps exactly the grid size are rolled so the player's hill is at the
    center (lossless on the torus). Smaller maps are padded and marked
    off-map.
- **Action:** 1024 heads (one per cell) x 5: stay / N / E / S / W.
  - The mask allows only "stay" on cells without your ant.
  - Moves into water or food are masked, since the engine would ignore
    them.
- **Reward:** ±`reward_win` at game end by final score, plus untuned
  shaping (`reward_score`, `reward_food`, `reward_kill`). The game ends
  per the original rules, or at `turns` (500).
- **Grid size:** compile-time `AIANTS_GRID` (16, 32 or 64; default 32).

Training is out of scope. The default MLP policy is a poor fit for 1024
heads; a conv encoder/decoder (like `nmmo3`'s custom net) will be needed.

## Tests

From the repo root:

```bash
python3 ocean/aiants/tests/run_parity.py          # full matrix (~10 min on 4 cores)
python3 ocean/aiants/tests/run_parity.py --quick  # ~30 s
```

The script clones the original repo into `build/aichallenge` at the pinned
commit and compiles `parity_driver.c` (with ASan/UBSan when available).
Each case:
1. `ref_runner.py` plays a game on the unmodified `ants.py`, with scripted
   bots that also send invalid, duplicate and blocked orders.
2. The C driver replays the same orders.
3. The script diffs per-turn dumps: scores, hive, food counter, cutoff
   state, RNG state, hills, ants (in dict order), food, pending food, dead
   ants, every player's vision, explored cells and enemy numbering, and
   the exact bot input text.

The matrix covers:
- official maps for every player count 2-10;
- the 32x32 pool;
- all 4 attack modes and all 4 food modes;
- scenario mode, non-default radii and cutoff settings;
- every game ending.

Result: 503/503 cases identical.

Env checks and benchmarks:

```bash
clang -O2 -mavx2 -mfma -I./raylib-5.5_linux_amd64/include -Isrc -Ivendor \
    ocean/aiants/tests/test_aiants.c raylib-5.5_linux_amd64/lib/libraylib.a \
    -lGL -lm -lpthread -ldl -o build/aiants_test && ./build/aiants_test
python3 ocean/aiants/tests/bench_python.py build/aichallenge/ants \
    resources/aiants/maps/cell_maze_32x32_p02_000.map
```

Single thread, 32x32, default rules (4-core cloud VM):

| setting | C engine turns/s | env steps/s (agent steps/s) | ants alive |
|---|---|---|---|
| random orders | ~447k | ~140k (~280k) | 8 |
| food-seeking heuristic | ~436k | ~138k (~277k) | 10 |
| heuristic, 10x food (stress) | ~188k | ~95k (~190k) | 22 |
| original Python engine, random | ~3.1k | | 8 |

Memory is about 215 KB per env (engine buffers plus view tables).

## Maps

The official generator rejects maps under 900 squares per player and hills
under 20 steps apart, so a 1v1 map can't be smaller than about 43x43.
`tools/gen_maps.py` runs the same `cell_maze` generator and the same
symmetry and accessibility checks, relaxing only those two limits (hills
at least 12 steps apart).
