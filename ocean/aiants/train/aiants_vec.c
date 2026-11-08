// Vectorized aiants for a Python trainer (ctypes). Agent slot 0 is the
// learner; slot 1 is a scripted bot playing from its own fog-of-war view.
// Build: see ocean/aiants/train/README.md
#include "../aiants.h"
#include <omp.h>

enum {BOT_RANDOM, BOT_GATHERER, BOT_RAIDER, NUM_BOTS};

typedef struct {
    int n;
    Env* envs;
    int* opponent;          // per env bot kind
    unsigned int* bot_rng;
    obs_t* bot_obs;         // bot slot buffers (unused by bots except mask)
    float* bot_actions;
    unsigned char* bot_mask;
    float* bot_rewards;
    float* bot_terminals;
    // per-env episode results, written when the learner's terminal fires
    float* ep_outcome;      // +1 win, -1 loss, 0 draw (learner)
    float* ep_length;
    float* ep_food;         // food gathered by the learner
} Vec;

static int toward(Ants* g, int loc, int dr, int dc, unsigned int* rng) {
    int dirs[2], n = 0;
    if (dr) {
        dirs[n++] = dr < 0 ? 0 : 2;
    }
    if (dc) {
        dirs[n++] = dc < 0 ? 3 : 1;
    }
    if (n == 2 && (abs(dc) > abs(dr) || (abs(dc) == abs(dr) && rand_r(rng) & 1))) {
        int t = dirs[0];
        dirs[0] = dirs[1];
        dirs[1] = t;
    }
    for (int k = 0; k < n; k++) {
        int v = g->map[ants_dest(g, loc, ANTS_AIM[dirs[k]][0], ANTS_AIM[dirs[k]][1])];
        if (v != ANTS_WATER && v != ANTS_FOOD) {
            return dirs[k];
        }
    }
    return -1;
}

// Nearest visible target of a kind within radius^2 77 (view radius)
static int nearest(Ants* g, int p, int loc, int kind, int* odr, int* odc) {
    int best = 1 << 30;
    for (int dr = -8; dr <= 8; dr++) {
        for (int dc = -8; dc <= 8; dc++) {
            int d = dr * dr + dc * dc;
            if (d == 0 || d > 77 || d >= best) {
                continue;
            }
            int cell = ants_dest(g, loc, dr, dc);
            if (!ants_visible(g, p, cell)) {
                continue;
            }
            int v = g->map[cell], h = g->hill_at[cell];
            int hit = kind == 0 ? v == ANTS_FOOD
                : kind == 1 ? (h >= 0 && g->hill_owner[h] != p && g->hill_killed_by[h] < 0)
                : (v >= 0 && v != p);
            if (hit) {
                best = d;
                *odr = dr;
                *odc = dc;
            }
        }
    }
    return best;
}

static int bot_move(Ants* g, int p, int i, int kind, unsigned int* rng) {
    int loc = g->ant_loc[i], dr = 0, dc = 0, dir = -1;
    if (kind == BOT_RANDOM) {
        return rand_r(rng) % 5;
    }
    int food = nearest(g, p, loc, 0, &dr, &dc);
    if (kind == BOT_RAIDER && food > 16) {
        int fr = dr, fc = dc;
        if (nearest(g, p, loc, 1, &dr, &dc) < (1 << 30)) {
            dir = toward(g, loc, dr, dc, rng);
        } else if (nearest(g, p, loc, 2, &dr, &dc) <= 25 && g->n_ants > 1) {
            dir = toward(g, loc, dr, dc, rng);
        }
        dr = fr;
        dc = fc;
    }
    if (dir < 0 && food < (1 << 30)) {
        dir = toward(g, loc, dr, dc, rng);
    }
    return dir < 0 ? rand_r(rng) % 5 : dir;
}

static void bot_actions(Env* env, int slot, int kind, unsigned int* rng) {
    Ants* g = env->game;
    int p = env->slot_for_player[0] == slot ? 0 : 1;
    float* act = env->agents[slot].actions;
    memset(act, 0, sizeof(float) * NUM_ATNS);
    // avoid stepping onto a square another own ant is moving to
    static __thread int taken_stamp[AIANTS_CELLS];
    static __thread int stamp = 0;
    stamp++;
    for (int i = 0; i < g->n_ants; i++) {
        if (g->ant_owner[i] != p) {
            continue;
        }
        int loc = g->ant_loc[i], d = bot_move(g, p, i, kind, rng);
        int dest = d < 4 ? ants_dest(g, loc, ANTS_AIM[d][0], ANTS_AIM[d][1]) : loc;
        if (kind != BOT_RANDOM && taken_stamp[dest] == stamp) {
            d = 4;
            dest = loc;
        }
        taken_stamp[dest] = stamp;
        act[env->inv[p][loc]] = d < 4 ? d + 1 : 0;
    }
}

Vec* vec_create(int n, int seed, const char* map_dir, int turns, float reward_win,
        float reward_score, float reward_food, float reward_kill) {
    Vec* v = calloc(1, sizeof(Vec));
    v->n = n;
    v->envs = calloc(n, sizeof(Env));
    v->opponent = calloc(n, sizeof(int));
    v->bot_rng = calloc(n, sizeof(unsigned int));
    v->bot_obs = calloc((size_t)n * OBS_SIZE, 1);
    v->bot_actions = calloc((size_t)n * NUM_ATNS, sizeof(float));
    v->bot_mask = calloc((size_t)n * 5 * AIANTS_CELLS, 1);
    v->bot_rewards = calloc(n, sizeof(float));
    v->bot_terminals = calloc(n, sizeof(float));
    v->ep_outcome = calloc(n, sizeof(float));
    v->ep_length = calloc(n, sizeof(float));
    v->ep_food = calloc(n, sizeof(float));
    Dict kw = {0};
    dict_set(&kw, "num_agents", 2);
    dict_set_str(&kw, "map_dir", map_dir);
    dict_set(&kw, "turns", turns);
    dict_set(&kw, "cutoff_turn", 150);
    dict_set(&kw, "reward_win", reward_win);
    dict_set(&kw, "reward_score", reward_score);
    dict_set(&kw, "reward_food", reward_food);
    dict_set(&kw, "reward_kill", reward_kill);
    for (int i = 0; i < n; i++) {
        v->envs[i].rng = seed * 7919 + i * 104729 + 1;
        v->bot_rng[i] = seed * 31 + i * 17 + 3;
        puf_init(&v->envs[i], &kw);
    }
    return v;
}

int vec_obs_size(void) {
    return OBS_SIZE;
}

int vec_grid(void) {
    return AIANTS_GRID;
}

void vec_set_opponent(Vec* v, int i, int kind) {
    assert(kind >= 0 && kind < NUM_BOTS);
    v->opponent[i] = kind;
}

// Learner buffers are owned by the caller (numpy)
void vec_bind(Vec* v, obs_t* obs, unsigned char* mask, float* actions, float* rewards,
        float* terminals) {
    for (int i = 0; i < v->n; i++) {
        Agent* a = &v->envs[i].agents[0];
        a->observations = obs + (size_t)i * OBS_SIZE;
        a->action_mask = mask + (size_t)i * 5 * AIANTS_CELLS;
        a->actions = actions + (size_t)i * NUM_ATNS;
        a->rewards = rewards + i;
        a->terminals = terminals + i;
        Agent* b = &v->envs[i].agents[1];
        b->observations = v->bot_obs + (size_t)i * OBS_SIZE;
        b->action_mask = v->bot_mask + (size_t)i * 5 * AIANTS_CELLS;
        b->actions = v->bot_actions + (size_t)i * NUM_ATNS;
        b->rewards = v->bot_rewards + i;
        b->terminals = v->bot_terminals + i;
    }
}

void vec_reset(Vec* v) {
    for (int i = 0; i < v->n; i++) {
        puf_reset(&v->envs[i]);
    }
}

void vec_step(Vec* v) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v->n; i++) {
        Env* env = &v->envs[i];
        bot_actions(env, 1, v->opponent[i], &v->bot_rng[i]);
        float score = env->log.score, len = env->log.episode_length;
        int learner = env->slot_for_player[0] == 0 ? 0 : 1;
        int food = env->game->stat_gathered[learner];
        puf_step(env);
        if (env->agents[0].terminals[0]) {
            v->ep_outcome[i] = env->log.score - score;
            v->ep_length[i] = env->log.episode_length - len;
            v->ep_food[i] = food;
        }
    }
}

// Scripted bot on the learner's side too (reference win rates)
void vec_learner_bot(Vec* v, int kind) {
    for (int i = 0; i < v->n; i++) {
        bot_actions(&v->envs[i], 0, kind, &v->bot_rng[i]);
    }
}

float* vec_ep_outcome(Vec* v) {
    return v->ep_outcome;
}

float* vec_ep_length(Vec* v) {
    return v->ep_length;
}

float* vec_ep_food(Vec* v) {
    return v->ep_food;
}

// Full-map snapshot of env i in the map frame for rendering.
// Codes: 0 land, 1 water, 2 food, 3 learner ant, 4 bot ant, +8 learner hill,
// +16 bot hill, +32 fogged for the learner. Returns rows * 1000 + cols.
int vec_snapshot(Vec* v, int i, unsigned char* out) {
    Env* env = &v->envs[i];
    Ants* g = env->game;
    int learner = env->slot_for_player[0] == 0 ? 0 : 1;
    for (int c = 0; c < g->cells; c++) {
        int m = g->map[c], code = m == ANTS_WATER ? 1 : m == ANTS_FOOD ? 2 : 0;
        if (m >= 0) {
            code = m == learner ? 3 : 4;
        }
        int h = g->hill_at[c];
        if (h >= 0 && g->hill_killed_by[h] < 0) {
            code += g->hill_owner[h] == learner ? 8 : 16;
        }
        if (!ants_visible(g, learner, c)) {
            code += 32;
        }
        out[c] = code;
    }
    return g->rows * 1000 + g->cols;
}
