/* AI Ants: 1v1 gridnet RL interface over the 2011 AI Challenge Ants engine.
 *
 * Rules live in ants_engine.h (bit-exact port of the official engine); this
 * file only maps them to PufferLib. One agent per player. Observation is the
 * player's fog-of-war view as channel-major uint8 planes over a fixed
 * AIANTS_GRID x AIANTS_GRID grid, plus a few global scalars. Action is one
 * 5-way head per grid cell (stay/N/E/S/W), masked to cells with own ants.
 * Maps that fill the grid are rolled so the player's hill sits at the center
 * (lossless on the torus); smaller maps are padded and flagged off-map.
 */
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "ants_engine.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#ifndef AIANTS_GRID
#define AIANTS_GRID 32
#endif
#define AIANTS_CELLS (AIANTS_GRID * AIANTS_GRID)
#define AIANTS_PLAYERS 2
#define AIANTS_CHANNELS 9
#define AIANTS_GLOBALS 4
#define AIANTS_MAX_MAPS 4096

// Observation planes
#define CH_VISIBLE 0
#define CH_SEEN 1
#define CH_WATER 2
#define CH_FOOD 3
#define CH_OWN_ANT 4
#define CH_ENEMY_ANT 5
#define CH_OWN_HILL 6
#define CH_ENEMY_HILL 7     // last known alive enemy hills
#define CH_ON_MAP 8

#define OBS_SIZE (AIANTS_CHANNELS * AIANTS_CELLS + AIANTS_GLOBALS)
#define NUM_ATNS AIANTS_CELLS
#define AIANTS_R4(x) x, x, x, x
#define AIANTS_R16(x) AIANTS_R4(x), AIANTS_R4(x), AIANTS_R4(x), AIANTS_R4(x)
#define AIANTS_R64(x) AIANTS_R16(x), AIANTS_R16(x), AIANTS_R16(x), AIANTS_R16(x)
#define AIANTS_R256(x) AIANTS_R64(x), AIANTS_R64(x), AIANTS_R64(x), AIANTS_R64(x)
#define AIANTS_R1024(x) AIANTS_R256(x), AIANTS_R256(x), AIANTS_R256(x), AIANTS_R256(x)
#if AIANTS_GRID == 16
#define ACT_SIZES {AIANTS_R256(5)}
#elif AIANTS_GRID == 32
#define ACT_SIZES {AIANTS_R1024(5)}
#elif AIANTS_GRID == 64
#define ACT_SIZES {AIANTS_R1024(5), AIANTS_R1024(5), AIANTS_R1024(5), AIANTS_R1024(5)}
#else
#error "AIANTS_GRID must be 16, 32 or 64"
#endif

// Action per cell: 0 stay, 1-4 = engine dirs n, e, s, w
#define AIANTS_STAY 0

static const Color AIANTS_BG = {6, 24, 24, 255};
static const Color AIANTS_LAND = {40, 34, 26, 255};
static const Color AIANTS_WATER = {16, 40, 80, 255};
static const Color AIANTS_FOOD = {230, 230, 230, 255};
static const Color AIANTS_COLORS[AIANTS_PLAYERS] = {{0, 187, 187, 255}, {187, 0, 0, 255}};

struct Log {
    float perf;
    float score;
    float policy_0_score;
    float draw_rate;
    float episode_return;
    float episode_length;
    float food_gathered;
    float ants_lost;
    float max_ants;
    float hills_razed;
    float cutoff_lone_survivor;
    float cutoff_rank_stabilized;
    float cutoff_food;
    float cutoff_turn_limit;
    float n;
};

typedef struct {
    int n;
    AntsMap maps[AIANTS_MAX_MAPS];
} AiAntsPool;

static AiAntsPool* aiants_pool = NULL;

static int aiants_cmp_str(const void* a, const void* b) {
    return strcmp(*(char* const*)a, *(char* const*)b);
}

// Loads every 2-player map that fits the grid, sorted by file name
static AiAntsPool* aiants_load_pool(const char* dir) {
    AiAntsPool* pool = (AiAntsPool*)calloc(1, sizeof(AiAntsPool));
    DIR* d = opendir(dir);
    if (!d) {
        fprintf(stderr, "aiants: cannot open map_dir %s\n", dir);
        exit(1);
    }
    char* names[AIANTS_MAX_MAPS];
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d)) && n < AIANTS_MAX_MAPS) {
        int len = strlen(e->d_name);
        if (len > 4 && !strcmp(e->d_name + len - 4, ".map")) {
            names[n++] = strdup(e->d_name);
        }
    }
    closedir(d);
    qsort(names, n, sizeof(char*), aiants_cmp_str);
    for (int i = 0; i < n; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        FILE* f = fopen(path, "rb");
        assert(f);
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* text = (char*)malloc(size + 1);
        size_t got = fread(text, 1, size, f);
        text[got] = 0;
        fclose(f);
        AntsMap* m = &pool->maps[pool->n];
        if (!ants_parse_map(m, text) && m->players == AIANTS_PLAYERS
                && m->rows <= AIANTS_GRID && m->cols <= AIANTS_GRID) {
            pool->n++;
        }
        free(text);
        free(names[i]);
    }
    if (pool->n == 0) {
        fprintf(stderr, "aiants: no %d-player maps <= %dx%d in %s\n",
            AIANTS_PLAYERS, AIANTS_GRID, AIANTS_GRID, dir);
        exit(1);
    }
    return pool;
}

struct Env {
    Log log;
    Agent agents[AIANTS_PLAYERS];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;

    Ants* game;
    AntsConfig cfg;
    int map_index;
    int slot_for_player[AIANTS_PLAYERS];
    // grid cell -> map cell (-1 off map) and inverse, per player
    int view[AIANTS_PLAYERS][AIANTS_CELLS];
    int inv[AIANTS_PLAYERS][AIANTS_CELLS];
    uint8_t known_hill[AIANTS_PLAYERS][AIANTS_CELLS];   // per hill index

    float reward_win, reward_score, reward_food, reward_kill;
    float episode_return[AIANTS_PLAYERS];
    int max_ants;
};
typedef Env AiAnts;

static void aiants_build_view(AiAnts* env, int p) {
    Ants* g = env->game;
    int roll = g->rows == AIANTS_GRID && g->cols == AIANTS_GRID;
    int hill = 0;
    for (int h = 0; h < g->n_hills; h++) {
        if (g->hill_owner[h] == p) {
            hill = g->hill_loc[h];
            break;
        }
    }
    int dr = roll ? hill / g->cols - AIANTS_GRID / 2 : 0;
    int dc = roll ? hill % g->cols - AIANTS_GRID / 2 : 0;
    for (int i = 0; i < AIANTS_CELLS; i++) {
        env->inv[p][i] = -1;
    }
    for (int gi = 0; gi < AIANTS_CELLS; gi++) {
        int r = gi / AIANTS_GRID, c = gi % AIANTS_GRID;
        if (!roll && (r >= g->rows || c >= g->cols)) {
            env->view[p][gi] = -1;
            continue;
        }
        int mc = ants_dest(g, r * g->cols + c, dr, dc);
        env->view[p][gi] = mc;
        env->inv[p][mc] = gi;
    }
}

static void aiants_observe(AiAnts* env) {
    Ants* g = env->game;
    for (int p = 0; p < AIANTS_PLAYERS; p++) {
        Agent* a = &env->agents[env->slot_for_player[p]];
        obs_t* obs = a->observations;
        memset(obs, 0, OBS_SIZE);
        const uint16_t* vis = g->vision + (size_t)p * g->cells;
        const uint8_t* rev = g->revealed + (size_t)p * g->cells;
        for (int gi = 0; gi < AIANTS_CELLS; gi++) {
            int mc = env->view[p][gi];
            if (mc < 0) {
                continue;
            }
            obs[CH_ON_MAP * AIANTS_CELLS + gi] = 1;
            obs[CH_SEEN * AIANTS_CELLS + gi] = rev[mc];
            obs[CH_WATER * AIANTS_CELLS + gi] = rev[mc] && g->map[mc] == ANTS_WATER;
            if (!vis[mc]) {
                continue;
            }
            int v = g->map[mc];
            obs[CH_VISIBLE * AIANTS_CELLS + gi] = 1;
            obs[CH_FOOD * AIANTS_CELLS + gi] = v == ANTS_FOOD;
            obs[CH_ENEMY_ANT * AIANTS_CELLS + gi] = v >= 0 && v != p;
        }
        int own_hills = 0;
        for (int h = 0; h < g->n_hills; h++) {
            int loc = g->hill_loc[h], alive = g->hill_killed_by[h] < 0;
            if (g->hill_owner[h] == p) {
                own_hills += alive;
                obs[CH_OWN_HILL * AIANTS_CELLS + env->inv[p][loc]] = alive;
                continue;
            }
            if (vis[loc]) {
                env->known_hill[p][h] = alive;
            }
            if (env->known_hill[p][h]) {
                obs[CH_ENEMY_HILL * AIANTS_CELLS + env->inv[p][loc]] = 1;
            }
        }
        // mask: every head may stay; own ants may also take unblocked moves
        unsigned char* mask = a->action_mask;
        for (int gi = 0; gi < AIANTS_CELLS; gi++) {
            mask[5 * gi] = 1;
            memset(mask + 5 * gi + 1, 0, 4);
        }
        int own = 0;
        for (int i = 0; i < g->n_ants; i++) {
            if (g->ant_owner[i] != p) {
                continue;
            }
            own++;
            int loc = g->ant_loc[i], gi = env->inv[p][loc];
            obs[CH_OWN_ANT * AIANTS_CELLS + gi] = 1;
            for (int d = 0; d < 4; d++) {
                int v = g->map[ants_dest(g, loc, ANTS_AIM[d][0], ANTS_AIM[d][1])];
                mask[5 * gi + 1 + d] = v != ANTS_WATER && v != ANTS_FOOD;
            }
        }
        obs_t* glob = obs + AIANTS_CHANNELS * AIANTS_CELLS;
        glob[0] = g->turn * 255 / (g->cfg.turns > 0 ? g->cfg.turns : 1);
        glob[1] = g->hive[p] < 255 ? g->hive[p] : 255;
        glob[2] = own < 255 ? own : 255;
        glob[3] = own_hills;
    }
}

void puf_reset(AiAnts* env) {
    int idx = rand_r(&env->rng) % aiants_pool->n;
    long long seed = ((long long)rand_r(&env->rng) << 16) ^ rand_r(&env->rng);
    env->map_index = idx;
    ants_init(env->game, &aiants_pool->maps[idx], env->cfg, seed);
    ants_start(env->game);
    assert(!env->game->done);
    int swap = rand_r(&env->rng) & 1;
    env->slot_for_player[0] = swap;
    env->slot_for_player[1] = !swap;
    memset(env->known_hill, 0, sizeof(env->known_hill));
    env->max_ants = 0;
    for (int p = 0; p < AIANTS_PLAYERS; p++) {
        env->episode_return[p] = 0;
        aiants_build_view(env, p);
    }
    aiants_observe(env);
}

void puf_step(AiAnts* env) {
    Ants* g = env->game;
    int prev_score[AIANTS_PLAYERS], prev_food[AIANTS_PLAYERS], prev_lost[AIANTS_PLAYERS];
    for (int p = 0; p < AIANTS_PLAYERS; p++) {
        Agent* a = &env->agents[env->slot_for_player[p]];
        a->rewards[0] = 0;
        a->terminals[0] = 0;
        prev_score[p] = g->score[p];
        prev_food[p] = g->stat_gathered[p];
        prev_lost[p] = g->stat_lost[p];
    }
    ants_begin_turn(g, NULL, 0);
    for (int p = 0; p < AIANTS_PLAYERS; p++) {
        if (!ants_can_order(g, p)) {
            continue;
        }
        float* act = env->agents[env->slot_for_player[p]].actions;
        for (int i = 0; i < g->n_ants; i++) {
            if (g->ant_owner[i] != p) {
                continue;
            }
            int loc = g->ant_loc[i], a = (int)act[env->inv[p][loc]];
            if (a > AIANTS_STAY && a <= 4) {
                ants_order(g, p, loc / g->cols, loc % g->cols, a - 1);
            }
        }
    }
    ants_end_turn(g);
    env->max_ants = g->n_ants > env->max_ants ? g->n_ants : env->max_ants;
    float r[AIANTS_PLAYERS];
    for (int p = 0; p < AIANTS_PLAYERS; p++) {
        int q = 1 - p;
        r[p] = env->reward_score * ((g->score[p] - prev_score[p]) - (g->score[q] - prev_score[q]))
            + env->reward_food * (g->stat_gathered[p] - prev_food[p])
            + env->reward_kill * ((g->stat_lost[q] - prev_lost[q]) - (g->stat_lost[p] - prev_lost[p]));
    }
    if (!g->done) {
        for (int p = 0; p < AIANTS_PLAYERS; p++) {
            env->agents[env->slot_for_player[p]].rewards[0] = r[p];
            env->episode_return[p] += r[p];
        }
        aiants_observe(env);
        return;
    }
    // Terminal: rank by final score (after the original's end-of-game bonus)
    float outcome = g->score[0] > g->score[1] ? 1.0f : g->score[0] < g->score[1] ? -1.0f : 0.0f;
    for (int p = 0; p < AIANTS_PLAYERS; p++) {
        Agent* a = &env->agents[env->slot_for_player[p]];
        r[p] += env->reward_win * (p == 0 ? outcome : -outcome);
        a->rewards[0] = r[p];
        a->terminals[0] = 1;
        env->episode_return[p] += r[p];
    }
    // Logged from the perspective of agent slot 0
    int p0 = env->slot_for_player[0] == 0 ? 0 : 1;
    float o0 = p0 == 0 ? outcome : -outcome;
    env->log.perf += (o0 + 1.0f) / 2.0f;
    env->log.score += o0;
    env->log.policy_0_score += (o0 + 1.0f) / 2.0f;
    env->log.draw_rate += outcome == 0.0f;
    env->log.episode_return += env->episode_return[p0];
    env->log.episode_length += g->turn;
    env->log.food_gathered += g->stat_gathered[0] + g->stat_gathered[1];
    env->log.ants_lost += g->stat_lost[0] + g->stat_lost[1];
    env->log.max_ants += env->max_ants;
    for (int h = 0; h < g->n_hills; h++) {
        env->log.hills_razed += g->hill_end_turn[h] >= 0;
    }
    env->log.cutoff_lone_survivor += !strcmp(g->cutoff, "lone survivor");
    env->log.cutoff_rank_stabilized += !strcmp(g->cutoff, "rank stabilized");
    env->log.cutoff_food += !strcmp(g->cutoff, "food not being gathered");
    env->log.cutoff_turn_limit += !strcmp(g->cutoff, "turn limit reached");
    env->log.n += 1;
    if (env->tag > 0) {
        env->boundary_reached = 1;
    }
    puf_reset(env);
}

void puf_render(AiAnts* env) {
    Ants* g = env->game;
    int cell = 20, top = 40;
    if (!IsWindowReady()) {
        InitWindow(AIANTS_GRID * cell, AIANTS_GRID * cell + top, "PufferLib AI Ants");
        SetTargetFPS(15);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    // Full map from the map's own frame, fog of player 0 shaded
    ants_count_ants(g);
    BeginDrawing();
    ClearBackground(AIANTS_BG);
    for (int mc = 0; mc < g->cells; mc++) {
        int x = (mc % g->cols) * cell, y = (mc / g->cols) * cell + top, v = g->map[mc];
        Color c = v == ANTS_WATER ? AIANTS_WATER : AIANTS_LAND;
        if (!ants_visible(g, 0, mc)) {
            c = (Color){c.r / 2, c.g / 2, c.b / 2, 255};
        }
        DrawRectangle(x, y, cell, cell, c);
        int h = g->hill_at[mc];
        if (h >= 0 && g->hill_killed_by[h] < 0) {
            DrawRectangleLinesEx((Rectangle){x + 1, y + 1, cell - 2, cell - 2}, 3,
                AIANTS_COLORS[g->hill_owner[h]]);
        }
        if (v == ANTS_FOOD) {
            DrawCircle(x + cell / 2, y + cell / 2, cell / 5, AIANTS_FOOD);
        } else if (v >= 0) {
            DrawCircle(x + cell / 2, y + cell / 2, cell / 3, AIANTS_COLORS[v]);
        }
    }
    DrawText(TextFormat("turn %d   score %d : %d   ants %d : %d   hive %d : %d",
        g->turn, g->score[0], g->score[1], g->ant_count[0], g->ant_count[1],
        g->hive[0], g->hive[1]), 10, 10, 20, RAYWHITE);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(AiAnts* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    assert(env->num_agents == AIANTS_PLAYERS && "aiants is 1v1 selfplay: num_agents = 2");
    if (!aiants_pool) {
        aiants_pool = aiants_load_pool(dict_get_str(kwargs, "map_dir"));
    }
    env->cfg = ants_default_config();
    env->cfg.turns = dict_get(kwargs, "turns");
    env->cfg.cutoff_turn = dict_get(kwargs, "cutoff_turn");
    env->reward_win = dict_get(kwargs, "reward_win");
    env->reward_score = dict_get(kwargs, "reward_score");
    env->reward_food = dict_get(kwargs, "reward_food");
    env->reward_kill = dict_get(kwargs, "reward_kill");
    env->game = ants_new(AIANTS_CELLS, AIANTS_PLAYERS);
    for (int slot = 0; slot < AIANTS_PLAYERS; slot++) {
        env->agents[slot].policy = slot;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "food_gathered", log->food_gathered);
    dict_set(out, "ants_lost", log->ants_lost);
    dict_set(out, "max_ants", log->max_ants);
    dict_set(out, "hills_razed", log->hills_razed);
    dict_set(out, "cutoff_lone_survivor", log->cutoff_lone_survivor);
    dict_set(out, "cutoff_rank_stabilized", log->cutoff_rank_stabilized);
    dict_set(out, "cutoff_food", log->cutoff_food);
    dict_set(out, "cutoff_turn_limit", log->cutoff_turn_limit);
    dict_set(out, "n", log->n);
}
