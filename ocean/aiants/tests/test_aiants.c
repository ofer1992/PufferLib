// Env sanity checks + steps/sec benchmark (random legal actions).
// Build: see ocean/aiants/README.md
#include <time.h>
#include "../aiants.h"

#define CHECK(c) do { \
        if (!(c)) { \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #c); \
            exit(1); \
        } \
    } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void random_actions(Env* env, unsigned int* seed) {
    for (int s = 0; s < AIANTS_PLAYERS; s++) {
        Agent* a = &env->agents[s];
        for (int gi = 0; gi < AIANTS_CELLS; gi++) {
            unsigned char* m = a->action_mask + 5 * gi;
            int legal[5], n = 0;
            for (int k = 0; k < 5; k++) {
                if (m[k]) {
                    legal[n++] = k;
                }
            }
            a->actions[gi] = n == 1 ? legal[0] : legal[rand_r(seed) % n];
        }
    }
}

// Step toward the nearest food within radius^2 50, else a random direction
static int bot_dir(Ants* g, int loc, unsigned int* seed) {
    int r = loc / g->cols, c = loc % g->cols, best = 1 << 30, br = 0, bc = 0;
    for (int dr = -7; dr <= 7; dr++) {
        for (int dc = -7; dc <= 7; dc++) {
            int d = dr * dr + dc * dc;
            if (d == 0 || d > 50 || d >= best) {
                continue;
            }
            if (g->map[ants_dest(g, loc, dr, dc)] == ANTS_FOOD) {
                best = d;
                br = dr;
                bc = dc;
            }
        }
    }
    int dir = rand_r(seed) % 5;
    if (best < (1 << 30)) {
        dir = abs(br) > abs(bc) ? (br < 0 ? 0 : 2) : (bc < 0 ? 3 : 1);
    }
    if (dir < 4) {
        int v = g->map[ants_dest(g, r * g->cols + c, ANTS_AIM[dir][0], ANTS_AIM[dir][1])];
        if (v == ANTS_WATER || v == ANTS_FOOD) {
            dir = rand_r(seed) % 5;
        }
    }
    return dir;
}

static void heuristic_actions(Env* env, unsigned int* seed) {
    Ants* g = env->game;
    for (int s = 0; s < AIANTS_PLAYERS; s++) {
        memset(env->agents[s].actions, 0, sizeof(float) * NUM_ATNS);
    }
    for (int i = 0; i < g->n_ants; i++) {
        int p = g->ant_owner[i], loc = g->ant_loc[i], d = bot_dir(g, loc, seed);
        env->agents[env->slot_for_player[p]].actions[env->inv[p][loc]] = d < 4 ? d + 1 : 0;
    }
}

static void check_obs(Env* env) {
    Ants* g = env->game;
    for (int s = 0; s < AIANTS_PLAYERS; s++) {
        Agent* a = &env->agents[s];
        int p = env->slot_for_player[0] == s ? 0 : 1, own = 0, mask_ants = 0;
        for (int gi = 0; gi < AIANTS_CELLS; gi++) {
            unsigned char* m = a->action_mask + 5 * gi;
            CHECK(m[0] == 1);
            int moves = m[1] + m[2] + m[3] + m[4];
            int own_ant = a->observations[CH_OWN_ANT * AIANTS_CELLS + gi];
            CHECK(own_ant || moves == 0);
            own += own_ant;
            mask_ants += moves > 0;
            for (int ch = 0; ch < AIANTS_CHANNELS; ch++) {
                CHECK(a->observations[ch * AIANTS_CELLS + gi] <= 1);
            }
            // fog of war: dynamic objects only where visible
            if (!a->observations[CH_VISIBLE * AIANTS_CELLS + gi]) {
                CHECK(!a->observations[CH_FOOD * AIANTS_CELLS + gi]);
                CHECK(!a->observations[CH_ENEMY_ANT * AIANTS_CELLS + gi]);
            }
        }
        int real = 0;
        for (int i = 0; i < g->n_ants; i++) {
            real += g->ant_owner[i] == p;
        }
        CHECK(own == real);
        CHECK(mask_ants <= own);
    }
}

int main(int argc, char** argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 200000;
    Dict* kw = calloc(1, sizeof(Dict));
    dict_set(kw, "num_agents", 2);
    dict_set_str(kw, "map_dir", "resources/aiants/maps");
    dict_set(kw, "turns", 500);
    dict_set(kw, "cutoff_turn", 150);
    dict_set(kw, "reward_win", 1.0);
    dict_set(kw, "reward_score", 0.1);
    dict_set(kw, "reward_food", 0.01);
    dict_set(kw, "reward_kill", 0.01);

    Env* env = calloc(1, sizeof(Env));
    env->rng = 7;
    puf_init(env, kw);
    static obs_t obs[AIANTS_PLAYERS][OBS_SIZE];
    static float actions[AIANTS_PLAYERS][NUM_ATNS];
    static unsigned char masks[AIANTS_PLAYERS][5 * AIANTS_CELLS];
    float rewards[AIANTS_PLAYERS], terminals[AIANTS_PLAYERS];
    for (int s = 0; s < AIANTS_PLAYERS; s++) {
        env->agents[s].observations = obs[s];
        env->agents[s].actions = actions[s];
        env->agents[s].action_mask = masks[s];
        env->agents[s].rewards = &rewards[s];
        env->agents[s].terminals = &terminals[s];
    }
    puf_reset(env);
    printf("map pool: %d maps\n", aiants_pool->n);

    // Sanity: obs/mask invariants, terminals, zero-sum outcome
    unsigned int seed = 1;
    int games = 0;
    for (int t = 0; t < 20000; t++) {
        check_obs(env);
        random_actions(env, &seed);
        puf_step(env);
        CHECK(terminals[0] == terminals[1]);
        if (terminals[0]) {
            games++;
        }
    }
    CHECK(games > 0);
    printf("sanity ok: %d games, mean length %.1f, mean food %.1f, razes %.0f\n",
        games, env->log.episode_length / env->log.n, env->log.food_gathered / env->log.n,
        env->log.hills_razed);

    // Benchmarks: random orders, and a food-seeking heuristic that keeps
    // more ants alive (closer to a trained policy's load)
    Ants* g = ants_new(AIANTS_CELLS, 2);
    AntsConfig cfg = ants_default_config();
    cfg.turns = 500;
    const char* names[3] = {"random", "heuristic", "10x food"};
    for (int mode = 0; mode < 3; mode++) {
        const char* name = names[mode];
        cfg.food_rate[0] = cfg.food_rate[1] = mode == 2 ? 80 : 0;
        if (mode < 2) {
            cfg.food_rate[0] = 5;
            cfg.food_rate[1] = 11;
        }
        long turns = 0, ant_turns = 0;
        double t0 = now(), bot = 0;
        for (int k = 0; turns < steps; k++) {
            ants_init(g, &aiants_pool->maps[k % aiants_pool->n], cfg, k);
            ants_start(g);
            while (!g->done) {
                ants_begin_turn(g, NULL, 0);
                double s0 = now();
                int n = g->n_ants, locs[AIANTS_CELLS], dirs[AIANTS_CELLS], own[AIANTS_CELLS];
                for (int i = 0; i < n; i++) {
                    locs[i] = g->ant_loc[i];
                    own[i] = g->ant_owner[i];
                    dirs[i] = mode ? bot_dir(g, locs[i], &seed) : rand_r(&seed) % 5;
                }
                bot += now() - s0;
                for (int i = 0; i < n; i++) {
                    if (dirs[i] < 4) {
                        ants_order(g, own[i], locs[i] / g->cols, locs[i] % g->cols, dirs[i]);
                    }
                }
                ants_end_turn(g);
                turns++;
                ant_turns += g->n_ants;
            }
        }
        double el = now() - t0 - bot;
        printf("engine %-9s: %8.0f turns/s  (mean %.1f ants alive)\n", name, turns / el,
            (double)ant_turns / turns);
    }
    for (int mode = 0; mode < 3; mode++) {
        const char* name = names[mode];
        if (mode == 2) {
            env->cfg.food_rate[0] = env->cfg.food_rate[1] = 80;
            puf_reset(env);
        }
        memset(&env->log, 0, sizeof(env->log));
        long ant_turns = 0;
        double a0 = 0, b0 = now();
        for (int t = 0; t < steps; t++) {
            double s0 = now();
            if (mode) {
                heuristic_actions(env, &seed);
            } else {
                random_actions(env, &seed);
            }
            a0 += now() - s0;
            puf_step(env);
            ant_turns += env->game->n_ants;
        }
        double el = now() - b0 - a0;
        printf("env    %-9s: %8.0f env steps/s = %8.0f agent steps/s  (mean %.1f ants alive)\n",
            name, steps / el, 2 * steps / el, (double)ant_turns / steps);
    }
    return 0;
}
