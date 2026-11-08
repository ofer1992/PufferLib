// Ants: C port of the 2011 Google AI Challenge game engine.
//
// Source of truth: aichallenge/aichallenge @ 4237971 (ants/ants.py rules,
// worker/engine.py turn loop). Behavior is bit-exact with that engine run
// under CPython 3.8+: same map + engine_seed + orders -> same game, including
// the Python Mersenne Twister stream, dict/set iteration orders that feed it,
// and the engine loop's per-turn random.shuffle of bots. Known quirks of the
// original (support attack ignores friends, closest attack skips the outer
// radius, food sets over water, ...) are kept on purpose.
//
// No PufferLib or raylib dependency. Any player count (2-10), any map size.

#pragma once
#include <assert.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANTS_LAND -2
#define ANTS_FOOD -3
#define ANTS_WATER -4
#define ANTS_MAXP 10
#define ANTS_STAY 4
#define ANTS_NEW -1
#define ANTS_HILL_POINTS 2
#define ANTS_RAZE_POINTS -1

enum {ANTS_ATTACK_FOCUS, ANTS_ATTACK_CLOSEST, ANTS_ATTACK_SUPPORT, ANTS_ATTACK_DAMAGE};
enum {ANTS_FOOD_NONE, ANTS_FOOD_RANDOM, ANTS_FOOD_SECTIONS, ANTS_FOOD_SYMMETRIC};
enum {ANTS_ORDER_VALID, ANTS_ORDER_IGNORED, ANTS_ORDER_INVALID};

// AIM dict order in ants.py: n, e, s, w
static const int ANTS_AIM[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
static const char ANTS_DIR_CHARS[] = "nesw-";

// CPython's random module: MT19937 + seed(int) + random/_randbelow/shuffle
typedef struct {
    uint32_t mt[624];
    int index;
} PyRand;

void pyrand_init_genrand(PyRand* r, uint32_t s) {
    r->mt[0] = s;
    for (int i = 1; i < 624; i++) {
        r->mt[i] = 1812433253U * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + i;
    }
    r->index = 624;
}

void pyrand_seed(PyRand* r, long long seed) {
    uint64_t a = seed < 0 ? -(uint64_t)seed : (uint64_t)seed;
    uint32_t key[2] = {(uint32_t)a, (uint32_t)(a >> 32)};
    int n = (a >> 32) ? 2 : 1;
    pyrand_init_genrand(r, 19650218U);
    uint32_t* mt = r->mt;
    int i = 1, j = 0;
    for (int k = 624 > n ? 624 : n; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + j;
        i++;
        j++;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
        if (j >= n) {
            j = 0;
        }
    }
    for (int k = 623; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - i;
        i++;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
    }
    mt[0] = 0x80000000U;
}

uint32_t pyrand_u32(PyRand* r) {
    uint32_t* mt = r->mt;
    if (r->index >= 624) {
        for (int k = 0; k < 624; k++) {
            uint32_t y = (mt[k] & 0x80000000U) | (mt[(k + 1) % 624] & 0x7fffffffU);
            mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0);
        }
        r->index = 0;
    }
    uint32_t y = mt[r->index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;
    return y;
}

double pyrand_random(PyRand* r) {
    uint32_t a = pyrand_u32(r) >> 5, b = pyrand_u32(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// random._randbelow_with_getrandbits, n in [1, 2^31)
int pyrand_below(PyRand* r, int n) {
    assert(n > 0);
    int k = 0;
    while ((1u << k) <= (unsigned)n && k < 32) {
        k++;
    }
    uint32_t v = pyrand_u32(r) >> (32 - k);
    while (v >= (uint32_t)n) {
        v = pyrand_u32(r) >> (32 - k);
    }
    return (int)v;
}

// Same weighting as sum((i + 1) * x for i, x in enumerate(random.getstate()[1]))
uint64_t pyrand_hash(PyRand* r) {
    uint64_t h = 0;
    for (int i = 0; i < 624; i++) {
        h += (uint64_t)(i + 1) * r->mt[i];
    }
    return h + 625ull * (uint64_t)r->index;
}

// list(set(locs)) for (row, col) int tuples: CPython 3.8+ tuple hash + set
// probing. Writes unique locs in iteration order, returns the count.
uint64_t pytuple2_hash(int a, int b) {
    const uint64_t p1 = 11400714785074694791ull, p2 = 14029467366897019727ull;
    const uint64_t p5 = 2870177450012600261ull;
    uint64_t acc = p5;
    acc += (uint64_t)a * p2;
    acc = (acc << 31) | (acc >> 33);
    acc *= p1;
    acc += (uint64_t)b * p2;
    acc = (acc << 31) | (acc >> 33);
    acc *= p1;
    acc += 2 ^ (p5 ^ 3527539ull);
    return acc == (uint64_t)-1 ? 1546275796ull : acc;
}

int pyset_order(const int* rows, const int* cols, int n, int ncols, int* out) {
    assert(n <= 64);
    int table[256];
    uint64_t hashes[256];
    size_t mask = 7;
    int fill = 0;
    for (size_t i = 0; i <= mask; i++) {
        table[i] = -1;
    }
    for (int k = 0; k < n; k++) {
        int key = rows[k] * ncols + cols[k];
        uint64_t h = pytuple2_hash(rows[k], cols[k]);
        size_t i = h & mask, perturb = h;
        int done = 0;
        while (!done) {
            int probes = (i + 9 <= mask) ? 9 : 0;
            size_t e = i;
            do {
                if (table[e] < 0) {
                    table[e] = key;
                    hashes[e] = h;
                    fill++;
                    done = 2;
                    break;
                }
                if (table[e] == key) {
                    done = 1;
                    break;
                }
                e++;
            } while (probes--);
            if (!done) {
                perturb >>= 5;
                i = (i * 5 + 1 + perturb) & mask;
            }
        }
        if (done == 2 && (size_t)fill * 5 >= mask * 3) {
            size_t newsize = 8;
            while (newsize <= (size_t)fill * 4) {
                newsize <<= 1;
            }
            assert(newsize <= 256);
            int old[256];
            uint64_t oldh[256];
            size_t oldmask = mask;
            memcpy(old, table, sizeof(int) * (oldmask + 1));
            memcpy(oldh, hashes, sizeof(uint64_t) * (oldmask + 1));
            mask = newsize - 1;
            for (size_t j = 0; j <= mask; j++) {
                table[j] = -1;
            }
            for (size_t j = 0; j <= oldmask; j++) {
                if (old[j] < 0) {
                    continue;
                }
                size_t p = oldh[j], q = oldh[j] & mask;
                while (1) {
                    if (table[q] < 0) {
                        break;
                    }
                    int found = 0;
                    if (q + 9 <= mask) {
                        for (int l = 1; l <= 9; l++) {
                            if (table[q + l] < 0) {
                                q += l;
                                found = 1;
                                break;
                            }
                        }
                    }
                    if (found) {
                        break;
                    }
                    p >>= 5;
                    q = (q * 5 + 1 + p) & mask;
                }
                table[q] = old[j];
                hashes[q] = oldh[j];
            }
        }
    }
    int m = 0;
    for (size_t i = 0; i <= mask; i++) {
        if (table[i] >= 0) {
            out[m++] = table[i];
        }
    }
    return m;
}

int ants_pydiv(int a, int b) {
    int q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int ants_pymod(int a, int b) {
    return a - ants_pydiv(a, b) * b;
}

typedef struct {
    int rows, cols, players;
    char* grid;
} AntsMap;

// Official .map text format. Returns 0 on success.
int ants_parse_map(AntsMap* m, const char* text) {
    m->rows = m->cols = m->players = -1;
    m->grid = NULL;
    int row = 0;
    const char* p = text;
    char line[4096];
    while (*p) {
        const char* end = strchr(p, '\n');
        int len = end ? (int)(end - p) : (int)strlen(p);
        assert(len < (int)sizeof(line));
        memcpy(line, p, len);
        line[len] = 0;
        p += len + (end ? 1 : 0);
        char* s = line;
        while (*s && isspace((unsigned char)*s)) {
            s++;
        }
        int sl = strlen(s);
        while (sl > 0 && isspace((unsigned char)s[sl - 1])) {
            s[--sl] = 0;
        }
        if (!*s || *s == '#') {
            continue;
        }
        char* sp = strchr(s, ' ');
        if (!sp) {
            return 1;
        }
        *sp = 0;
        char* val = sp + 1;
        for (char* c = s; *c; c++) {
            *c = tolower((unsigned char)*c);
        }
        if (!strcmp(s, "cols")) {
            m->cols = atoi(val);
        } else if (!strcmp(s, "rows")) {
            m->rows = atoi(val);
        } else if (!strcmp(s, "players")) {
            m->players = atoi(val);
            if (m->players < 2 || m->players > ANTS_MAXP) {
                return 2;
            }
        } else if (!strcmp(s, "m")) {
            if (m->players < 0 || m->rows <= 0 || m->cols <= 0 || row >= m->rows) {
                return 3;
            }
            if ((int)strlen(val) != m->cols) {
                return 4;
            }
            if (!m->grid) {
                m->grid = (char*)malloc(m->rows * m->cols);
            }
            for (int c = 0; c < m->cols; c++) {
                char ch = val[c];
                int ok = ch == '.' || ch == '%' || ch == '*'
                    || (ch >= 'a' && ch < 'a' + m->players)
                    || (ch >= 'A' && ch < 'A' + m->players)
                    || (ch >= '0' && ch < '0' + m->players);
                if (!ok) {
                    return 5;
                }
                m->grid[row * m->cols + c] = ch;
            }
            row++;
        }
    }
    return (m->grid && row == m->rows) ? 0 : 6;
}

typedef struct {
    int turns, viewradius2, attackradius2, spawnradius2;
    int attack, food;
    int food_rate[2], food_turn[2], food_start[2], food_visible[2];
    int cutoff_turn;
    double cutoff_percent;
    int scenario;
} AntsConfig;

// Defaults of ants/playgame.py
AntsConfig ants_default_config(void) {
    return (AntsConfig){
        .turns = 1000, .viewradius2 = 77, .attackradius2 = 5, .spawnradius2 = 1,
        .attack = ANTS_ATTACK_FOCUS, .food = ANTS_FOOD_SYMMETRIC,
        .food_rate = {5, 11}, .food_turn = {19, 37}, .food_start = {75, 175},
        .food_visible = {3, 5}, .cutoff_turn = 150, .cutoff_percent = 0.85,
        .scenario = 0,
    };
}

// collections.deque of food sets, -1 is the None sentinel
typedef struct {
    int n_sets;
    int* locs;      // n_sets * players
    int* len;
    int* buf;
    int head, size, cap;
} AntsFoodDeque;

typedef struct {
    AntsConfig cfg;
    int rows, cols, cells, players;
    int cap_cells, cap_players;
    PyRand rng;
    int food_rate, food_turn, food_start, food_visible;
    long long food_extra;   // Fraction numerator over food_turn
    int turn, land_area, done, accept_orders;
    const char* cutoff;
    int8_t* map;

    // live ants in Python dict (insertion) order
    int n_ants;
    int* ant_loc;
    int* ant_prev;
    int8_t* ant_owner;
    int8_t* ant_dir;    // ANTS_NEW, 0-3 moved, ANTS_STAY
    uint8_t* ant_dead;
    int* ant_at;        // cell -> ant index, -1
    int* ant_loc2;
    int* ant_prev2;
    int8_t* ant_owner2;
    int8_t* ant_dir2;

    int n_killed;       // ants killed this turn
    int* killed_loc;
    int* killed_prev;
    int8_t* killed_owner;

    int n_hills;        // hills in dict order
    int* hill_loc;
    int8_t* hill_owner;
    int8_t* hill_killed_by;     // -1 if alive
    int* hill_last_touched;
    int* hill_end_turn;
    uint8_t* hill_raze_points;
    int* hill_at;
    int n_hills_p0;

    int n_food;
    int* food_loc;
    int* food_at;
    int* pending;
    int n_pending;
    int* pending_list;

    uint16_t* vision;   // players * cells, count of own ants seeing each cell
    uint8_t* revealed;  // players * cells
    int* rwater;        // players * cells, water revealed this turn
    int rwater_n[ANTS_MAXP];
    int* cand;          // newly visible candidates, p * cells + cell
    int n_cand;
    int8_t switch_[ANTS_MAXP][ANTS_MAXP];
    int n_switch[ANTS_MAXP];

    int score[ANTS_MAXP], bonus[ANTS_MAXP], hive[ANTS_MAXP], ant_count[ANTS_MAXP];
    uint8_t killed[ANTS_MAXP], alive_begin[ANTS_MAXP];
    int stat_gathered[ANTS_MAXP], stat_lost[ANTS_MAXP];  // stats only, not rules
    int cutoff_bot, cutoff_turns, hill_kill;
    int ranking_turn, winning_turn, have_ranking;
    int ranking[ANTS_MAXP];
    uint8_t winning[ANTS_MAXP];

    int n_orders[ANTS_MAXP];
    int* order_loc;     // players * cells
    int8_t* order_dir;
    int* seen;
    int seen_stamp;

    int view_n, att_n, spawn_n;
    int* view;          // stencils as normalized (dr, dc) pairs
    int* att;
    int* spawn;
    int vadd_n[4], vrem_n[4];
    int* vadd[4];
    int* vrem[4];
    int stencil_cap;

    int have_sym;
    int sym_loc[ANTS_MAXP], sym_aim[ANTS_MAXP];
    AntsFoodDeque vis_dq, sym_dq;
    int vis_init, sym_init;
    int* access;        // sections food: per player square lists
    int access_start[ANTS_MAXP + 1];

    int* tmp_a;
    int* tmp_b;
    int* tmp_c;
    int* queue;
    long long* dmg;
} Ants;

static void* ants_xalloc(size_t n) {
    void* p = calloc(n ? n : 1, 1);
    assert(p);
    return p;
}

void ants_dq_alloc(AntsFoodDeque* d, int cells, int players) {
    d->locs = (int*)ants_xalloc(sizeof(int) * cells * players);
    d->len = (int*)ants_xalloc(sizeof(int) * cells);
    d->buf = (int*)ants_xalloc(sizeof(int) * (cells + 1));
    d->cap = cells + 1;
}

Ants* ants_new(int cap_cells, int cap_players) {
    Ants* g = (Ants*)ants_xalloc(sizeof(Ants));
    int C = cap_cells, P = cap_players;
    g->cap_cells = C;
    g->cap_players = P;
    g->map = (int8_t*)ants_xalloc(C);
    g->ant_loc = (int*)ants_xalloc(sizeof(int) * C);
    g->ant_prev = (int*)ants_xalloc(sizeof(int) * C);
    g->ant_owner = (int8_t*)ants_xalloc(C);
    g->ant_dir = (int8_t*)ants_xalloc(C);
    g->ant_dead = (uint8_t*)ants_xalloc(C);
    g->ant_at = (int*)ants_xalloc(sizeof(int) * C);
    g->ant_loc2 = (int*)ants_xalloc(sizeof(int) * C);
    g->ant_prev2 = (int*)ants_xalloc(sizeof(int) * C);
    g->ant_owner2 = (int8_t*)ants_xalloc(C);
    g->ant_dir2 = (int8_t*)ants_xalloc(C);
    g->killed_loc = (int*)ants_xalloc(sizeof(int) * C);
    g->killed_prev = (int*)ants_xalloc(sizeof(int) * C);
    g->killed_owner = (int8_t*)ants_xalloc(C);
    g->hill_loc = (int*)ants_xalloc(sizeof(int) * C);
    g->hill_owner = (int8_t*)ants_xalloc(C);
    g->hill_killed_by = (int8_t*)ants_xalloc(C);
    g->hill_last_touched = (int*)ants_xalloc(sizeof(int) * C);
    g->hill_end_turn = (int*)ants_xalloc(sizeof(int) * C);
    g->hill_raze_points = (uint8_t*)ants_xalloc(C);
    g->hill_at = (int*)ants_xalloc(sizeof(int) * C);
    g->food_loc = (int*)ants_xalloc(sizeof(int) * C);
    g->food_at = (int*)ants_xalloc(sizeof(int) * C);
    g->pending = (int*)ants_xalloc(sizeof(int) * C);
    g->pending_list = (int*)ants_xalloc(sizeof(int) * C);
    g->vision = (uint16_t*)ants_xalloc(sizeof(uint16_t) * P * C);
    g->revealed = (uint8_t*)ants_xalloc(P * C);
    g->rwater = (int*)ants_xalloc(sizeof(int) * P * C);
    g->cand = (int*)ants_xalloc(sizeof(int) * P * C);
    g->order_loc = (int*)ants_xalloc(sizeof(int) * P * C);
    g->order_dir = (int8_t*)ants_xalloc(P * C);
    g->seen = (int*)ants_xalloc(sizeof(int) * C);
    g->access = (int*)ants_xalloc(sizeof(int) * C);
    g->tmp_a = (int*)ants_xalloc(sizeof(int) * C);
    g->tmp_b = (int*)ants_xalloc(sizeof(int) * C);
    g->tmp_c = (int*)ants_xalloc(sizeof(int) * C);
    g->queue = (int*)ants_xalloc(sizeof(int) * 4 * C + 64);
    g->dmg = (long long*)ants_xalloc(sizeof(long long) * C);
    ants_dq_alloc(&g->vis_dq, C, P);
    ants_dq_alloc(&g->sym_dq, C, P);
    return g;
}

static inline int ants_dest(Ants* g, int loc, int dr, int dc) {
    int r = (loc / g->cols + dr) % g->rows, c = (loc % g->cols + dc) % g->cols;
    r += r < 0 ? g->rows : 0;
    c += c < 0 ? g->cols : 0;
    return r * g->cols + c;
}

// Stencil offsets are pre-normalized to [0, rows) x [0, cols)
static inline int ants_off(Ants* g, int r, int c, const int* off) {
    int rr = r + off[0], cc = c + off[1];
    rr -= rr >= g->rows ? g->rows : 0;
    cc -= cc >= g->cols ? g->cols : 0;
    return rr * g->cols + cc;
}

int ants_distance(Ants* g, int a, int b) {
    int dr = abs(a / g->cols - b / g->cols), dc = abs(a % g->cols - b % g->cols);
    dr = dr < g->rows - dr ? dr : g->rows - dr;
    dc = dc < g->cols - dc ? dc : g->cols - dc;
    return dr * dr + dc * dc;
}

static int ants_isqrt(int x) {
    int m = 0;
    while ((m + 1) * (m + 1) <= x) {
        m++;
    }
    return m;
}

// neighbourhood_offsets(): 0 < d <= max_dist, same order, may hold duplicate
// cells on maps smaller than the radius (as in Python)
static int ants_nbhd(Ants* g, int max_dist, int* out) {
    int mx = ants_isqrt(max_dist), n = 0;
    for (int dr = -mx; dr <= mx; dr++) {
        for (int dc = -mx; dc <= mx; dc++) {
            int d = dr * dr + dc * dc;
            if (d > 0 && d <= max_dist) {
                out[2 * n] = ((dr % g->rows) + g->rows) % g->rows;
                out[2 * n + 1] = ((dc % g->cols) + g->cols) % g->cols;
                n++;
            }
        }
    }
    return n;
}

static void ants_build_stencils(Ants* g) {
    int mxv = ants_isqrt(g->cfg.viewradius2), mxa = ants_isqrt(g->cfg.attackradius2);
    int mxs = ants_isqrt(g->cfg.spawnradius2), mx = mxv > mxa ? mxv : mxa;
    mx = mx > mxs ? mx : mxs;
    int need = (2 * mx + 1) * (2 * mx + 1) + 1;
    if (need > g->stencil_cap) {
        free(g->view);
        free(g->att);
        free(g->spawn);
        g->view = (int*)ants_xalloc(sizeof(int) * 2 * need);
        g->att = (int*)ants_xalloc(sizeof(int) * 2 * need);
        g->spawn = (int*)ants_xalloc(sizeof(int) * 2 * need);
        for (int d = 0; d < 4; d++) {
            free(g->vadd[d]);
            free(g->vrem[d]);
            g->vadd[d] = (int*)ants_xalloc(sizeof(int) * 2 * need);
            g->vrem[d] = (int*)ants_xalloc(sizeof(int) * 2 * need);
        }
        g->stencil_cap = need;
    }
    g->att_n = ants_nbhd(g, g->cfg.attackradius2, g->att);
    g->spawn_n = ants_nbhd(g, g->cfg.spawnradius2, g->spawn);
    // vision uses set(offsets) + (0, 0): unique normalized cells
    int* tmp = (int*)ants_xalloc(sizeof(int) * 2 * need);
    int n = ants_nbhd(g, g->cfg.viewradius2, tmp);
    tmp[2 * n] = tmp[2 * n + 1] = 0;
    n++;
    uint8_t* mark = (uint8_t*)ants_xalloc(g->cells);
    g->view_n = 0;
    for (int i = 0; i < n; i++) {
        int cell = tmp[2 * i] * g->cols + tmp[2 * i + 1];
        if (!mark[cell]) {
            mark[cell] = 1;
            g->view[2 * g->view_n] = tmp[2 * i];
            g->view[2 * g->view_n + 1] = tmp[2 * i + 1];
            g->view_n++;
        }
    }
    uint8_t* pmark = (uint8_t*)ants_xalloc(g->cells);
    for (int d = 0; d < 4; d++) {
        memset(pmark, 0, g->cells);
        int pr = (-ANTS_AIM[d][0] + g->rows) % g->rows;
        int pc = (-ANTS_AIM[d][1] + g->cols) % g->cols;
        for (int i = 0; i < g->view_n; i++) {
            pmark[ants_off(g, pr, pc, &g->view[2 * i])] = 1;
        }
        g->vadd_n[d] = g->vrem_n[d] = 0;
        for (int cell = 0; cell < g->cells; cell++) {
            int r = cell / g->cols, c = cell % g->cols;
            if (mark[cell] && !pmark[cell]) {
                g->vadd[d][2 * g->vadd_n[d]] = r;
                g->vadd[d][2 * g->vadd_n[d] + 1] = c;
                g->vadd_n[d]++;
            } else if (pmark[cell] && !mark[cell]) {
                g->vrem[d][2 * g->vrem_n[d]] = r;
                g->vrem[d][2 * g->vrem_n[d] + 1] = c;
                g->vrem_n[d]++;
            }
        }
    }
    free(tmp);
    free(mark);
    free(pmark);
}

static void ants_vision_apply(Ants* g, int p, int loc, const int* off, int n, int delta) {
    uint16_t* v = g->vision + (size_t)p * g->cells;
    int r = loc / g->cols, c = loc % g->cols;
    for (int i = 0; i < n; i++) {
        int cell = ants_off(g, r, c, &off[2 * i]);
        if (delta > 0) {
            if (v[cell] == 0 && !g->revealed[(size_t)p * g->cells + cell]) {
                assert(g->n_cand < g->players * g->cells);
                g->cand[g->n_cand++] = p * g->cells + cell;
            }
            v[cell]++;
        } else {
            assert(v[cell] > 0);
            v[cell]--;
        }
    }
}

static void ants_update_vision(Ants* g) {
    g->n_cand = 0;
    for (int i = 0; i < g->n_ants; i++) {
        int d = g->ant_dir[i], o = g->ant_owner[i], loc = g->ant_loc[i];
        if (d == ANTS_NEW) {
            ants_vision_apply(g, o, loc, g->view, g->view_n, 1);
        } else if (d < 4) {
            ants_vision_apply(g, o, loc, g->vadd[d], g->vadd_n[d], 1);
            ants_vision_apply(g, o, loc, g->vrem[d], g->vrem_n[d], -1);
        }
    }
    for (int k = 0; k < g->n_killed; k++) {
        ants_vision_apply(g, g->killed_owner[k], g->killed_prev[k], g->view, g->view_n, -1);
    }
}

static inline int ants_visible(Ants* g, int p, int cell) {
    return g->vision[(size_t)p * g->cells + cell] > 0;
}

static void ants_assign_switch(Ants* g, int p, int owner) {
    if (g->switch_[p][owner] < 0) {
        g->switch_[p][owner] = g->n_switch[p]++;
    }
}

static void ants_update_revealed(Ants* g) {
    for (int k = 0; k < g->n_cand; k++) {
        int p = g->cand[k] / g->cells, cell = g->cand[k] % g->cells;
        size_t idx = (size_t)p * g->cells + cell;
        if (g->vision[idx] && !g->revealed[idx]) {
            g->revealed[idx] = 1;
            if (g->map[cell] == ANTS_WATER) {
                g->rwater[idx - cell + g->rwater_n[p]++] = cell;
            }
        }
    }
    // switch: first sighting in row-major scan order of visible squares
    for (int p = 0; p < g->players; p++) {
        if (g->n_switch[p] >= g->players) {
            continue;
        }
        int first[ANTS_MAXP];
        for (int o = 0; o < g->players; o++) {
            first[o] = g->cells;
        }
        for (int i = 0; i < g->n_ants; i++) {
            int o = g->ant_owner[i], loc = g->ant_loc[i];
            if (g->switch_[p][o] < 0 && loc < first[o] && g->map[loc] == o
                    && ants_visible(g, p, loc)) {
                first[o] = loc;
            }
        }
        while (1) {
            int best = -1;
            for (int o = 0; o < g->players; o++) {
                if (g->switch_[p][o] < 0 && first[o] < g->cells
                        && (best < 0 || first[o] < first[best])) {
                    best = o;
                }
            }
            if (best < 0) {
                break;
            }
            ants_assign_switch(g, p, best);
        }
    }
}

static int ants_cmp_int(const void* a, const void* b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}

// render_changes(player): the text the original sends a bot each turn. Also
// applies the side effect of numbering enemies first seen as hills or dead
// ants. buf may be NULL (side effects only). Returns text length.
int ants_render(Ants* g, int p, char* buf, int buf_size) {
    int len = 0;
#define ANTS_EMIT(...) do { \
        if (buf) { \
            len += snprintf(buf + len, buf_size - len, __VA_ARGS__); \
            assert(len < buf_size); \
        } \
    } while (0)
    if (!buf && g->n_switch[p] >= g->players) {
        return 0;
    }
    int* tmp = g->tmp_c;
    if (buf) {
        int nw = g->rwater_n[p];
        memcpy(tmp, g->rwater + (size_t)p * g->cells, sizeof(int) * nw);
        qsort(tmp, nw, sizeof(int), ants_cmp_int);
        for (int i = 0; i < nw; i++) {
            ANTS_EMIT("w %d %d\n", tmp[i] / g->cols, tmp[i] % g->cols);
        }
    }
    int n = 0;
    for (int h = 0; h < g->n_hills; h++) {
        if (g->hill_killed_by[h] < 0) {
            tmp[n++] = g->hill_loc[h] * 16 + g->hill_owner[h];
        }
    }
    qsort(tmp, n, sizeof(int), ants_cmp_int);
    for (int i = 0; i < n; i++) {
        int loc = tmp[i] / 16, o = tmp[i] % 16;
        if (ants_visible(g, p, loc)) {
            ants_assign_switch(g, p, o);
            ANTS_EMIT("h %d %d %d\n", loc / g->cols, loc % g->cols, g->switch_[p][o]);
        }
    }
    n = 0;
    for (int i = 0; i < g->n_ants; i++) {
        tmp[n++] = g->ant_loc[i] * 16 + g->ant_owner[i];
    }
    qsort(tmp, n, sizeof(int), ants_cmp_int);
    for (int i = 0; i < n; i++) {
        int loc = tmp[i] / 16, o = tmp[i] % 16;
        if (ants_visible(g, p, loc)) {
            ants_assign_switch(g, p, o);
            ANTS_EMIT("a %d %d %d\n", loc / g->cols, loc % g->cols, g->switch_[p][o]);
        }
    }
    if (buf) {
        memcpy(tmp, g->food_loc, sizeof(int) * g->n_food);
        qsort(tmp, g->n_food, sizeof(int), ants_cmp_int);
        for (int i = 0; i < g->n_food; i++) {
            if (ants_visible(g, p, tmp[i])) {
                ANTS_EMIT("f %d %d\n", tmp[i] / g->cols, tmp[i] % g->cols);
            }
        }
    }
    n = 0;
    for (int k = 0; k < g->n_killed; k++) {
        tmp[n++] = g->killed_loc[k] * 16 + g->killed_owner[k];
    }
    qsort(tmp, n, sizeof(int), ants_cmp_int);
    for (int i = 0; i < n; i++) {
        int loc = tmp[i] / 16, o = tmp[i] % 16;
        if (ants_visible(g, p, loc) || o == p) {
            ants_assign_switch(g, p, o);
            ANTS_EMIT("d %d %d %d\n", loc / g->cols, loc % g->cols, g->switch_[p][o]);
        }
    }
#undef ANTS_EMIT
    return len;
}

static void ants_add_ant(Ants* g, int loc, int owner) {
    int i = g->n_ants++;
    g->ant_loc[i] = g->ant_prev[i] = loc;
    g->ant_owner[i] = owner;
    g->ant_dir[i] = ANTS_NEW;
    g->ant_dead[i] = 0;
    g->ant_at[loc] = i;
    g->map[loc] = owner;
}

static void ants_spawn_on_hill(Ants* g, int h) {
    ants_add_ant(g, g->hill_loc[h], g->hill_owner[h]);
    g->hill_last_touched[h] = g->turn;
}

static void ants_kill(Ants* g, int loc, int prev, int owner) {
    g->map[loc] = ANTS_LAND;
    int k = g->n_killed++;
    g->killed_loc[k] = loc;
    g->killed_prev[k] = prev;
    g->killed_owner[k] = owner;
    g->stat_lost[owner]++;
    int h = g->hill_at[loc];
    if (h >= 0 && g->hill_owner[h] != g->cutoff_bot && g->hill_killed_by[h] < 0) {
        g->hill_kill = 1;
    }
}

// current_food is a dict keyed by loc: food can stay listed under an ant
// spawned onto it (a hill), and re-adding the loc later overwrites the entry
static void ants_add_food(Ants* g, int loc) {
    assert(g->map[loc] == ANTS_LAND);
    g->map[loc] = ANTS_FOOD;
    if (g->food_at[loc] < 0) {
        g->food_at[loc] = g->n_food;
        g->food_loc[g->n_food++] = loc;
    }
}

static void ants_remove_food(Ants* g, int loc, int owner) {
    g->map[loc] = ANTS_LAND;
    int i = g->food_at[loc], last = g->food_loc[--g->n_food];
    g->food_loc[i] = last;
    g->food_at[last] = i;
    g->food_at[loc] = -1;
    if (owner >= 0) {
        g->hive[owner]++;
        g->stat_gathered[owner]++;
    }
}

static void ants_pend(Ants* g, int loc) {
    if (g->pending[loc]++ == 0) {
        g->pending_list[g->n_pending++] = loc;
    }
}

static void ants_place_food(Ants* g) {
    int m = 0;
    for (int k = 0; k < g->n_pending; k++) {
        int loc = g->pending_list[k];
        if (g->map[loc] == ANTS_LAND) {
            ants_add_food(g, loc);
            g->pending[loc]--;
        }
        if (g->pending[loc]) {
            g->pending_list[m++] = loc;
        }
    }
    g->n_pending = m;
}

static void ants_offset_aim(int r, int c, int aim, int* orow, int* ocol) {
    static const int tbl[8][4] = {
        {1, 0, 0, 1}, {-1, 0, 0, 1}, {1, 0, 0, -1}, {-1, 0, 0, -1},
        {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, -1, 0}, {0, -1, -1, 0},
    };
    *orow = tbl[aim][0] * r + tbl[aim][1] * c;
    *ocol = tbl[aim][2] * r + tbl[aim][3] * c;
}

static int ants_map_similar(Ants* g, int loc1, int loc2, int aim, int player) {
    for (int row = 0; row < g->rows; row++) {
        for (int col = 0; col < g->cols; col++) {
            int orow, ocol;
            ants_offset_aim(row, col, aim, &orow, &ocol);
            int ilk0 = g->map[ants_dest(g, loc1, row, col)];
            int ilk1 = g->map[ants_dest(g, loc2, orow, ocol)];
            if (ilk0 == 0 && ilk1 != player) {
                return 0;
            } else if (ilk0 > 0 && (ilk1 < 0 || ilk1 == player)) {
                return 0;
            } else if (ilk0 < 0 && ilk1 != ilk0) {
                return 0;
            }
        }
    }
    return 1;
}

// get_map_symmetry() + choice(): sets sym_loc/sym_aim, one entry per player
static void ants_choose_symmetry(Ants* g) {
    int P = g->players, h0 = -1;
    for (int h = 0; h < g->n_hills && h0 < 0; h++) {
        if (g->hill_owner[h] == 0) {
            h0 = h;
        }
    }
    assert(h0 >= 0);
    int cap = 64, n = 1;
    int* ori = (int*)malloc(sizeof(int) * cap * P);   // hill index * 8 + aim
    ori[0] = h0 * 8;
    for (int p = 1; p < P; p++) {
        int ncap = 64, nn = 0;
        int* next = (int*)malloc(sizeof(int) * ncap * P);
        for (int h = 0; h < g->n_hills; h++) {
            if (g->hill_owner[h] != p) {
                continue;
            }
            for (int aim = 0; aim < 8; aim++) {
                if (!ants_map_similar(g, g->hill_loc[h0], g->hill_loc[h], aim, p)) {
                    continue;
                }
                for (int k = 0; k < n; k++) {
                    if (nn == ncap) {
                        ncap *= 2;
                        next = (int*)realloc(next, sizeof(int) * ncap * P);
                    }
                    memcpy(next + nn * P, ori + k * P, sizeof(int) * p);
                    next[nn * P + p] = h * 8 + aim;
                    nn++;
                }
            }
        }
        free(ori);
        ori = next;
        n = nn;
        cap = ncap;
        assert(n > 0 && "map is not symmetric");
    }
    int nv = 0;
    int* valid = (int*)malloc(sizeof(int) * (n ? n : 1));
    int fix_loc[ANTS_MAXP], fix_ilk[ANTS_MAXP];
    for (int k = 0; k < n; k++) {
        int* o = ori + k * P;
        for (int e = 0; e < P; e++) {
            int r, c;
            ants_offset_aim(1, 2, o[e] % 8, &r, &c);
            int loc = ants_dest(g, g->hill_loc[o[e] / 8], r, c);
            fix_loc[e] = loc;
            fix_ilk[e] = g->map[loc];
            g->map[loc] = ANTS_FOOD;
        }
        int ok = 1;
        for (int e = 0; e < P && ok; e++) {
            ok = ants_map_similar(g, g->hill_loc[o[0] / 8], g->hill_loc[o[e] / 8],
                o[e] % 8, g->hill_owner[o[e] / 8]);
        }
        if (ok) {
            valid[nv++] = k;
        }
        for (int e = P - 1; e >= 0; e--) {
            g->map[fix_loc[e]] = fix_ilk[e];
        }
    }
    assert(nv > 0 && "no valid orientation sets");
    int* o = ori + valid[pyrand_below(&g->rng, nv)] * P;
    for (int e = 0; e < P; e++) {
        g->sym_loc[e] = g->hill_loc[o[e] / 8];
        g->sym_aim[e] = o[e] % 8;
    }
    g->have_sym = 1;
    free(ori);
    free(valid);
}

// get_symmetric_food_sets(starting) into a fresh deque + None sentinel
static void ants_build_food_sets(Ants* g, AntsFoodDeque* d, int starting) {
    if (!g->have_sym) {
        ants_choose_symmetry(g);
    }
    int P = g->players;
    uint8_t* visited = (uint8_t*)ants_xalloc(g->cells);
    uint8_t* vsq = (uint8_t*)ants_xalloc(g->cells);
    if (starting) {
        int vr = g->cfg.viewradius2;
        for (int h = 0; h < g->n_hills; h++) {
            // BFS with duplicate queueing, only membership matters
            uint8_t* inq = (uint8_t*)ants_xalloc(g->cells);
            int head = 0, tail = 0, hl = g->hill_loc[h];
            g->queue[tail++] = hl;
            inq[hl] = 1;
            while (head < tail) {
                int cl = g->queue[head++];
                vsq[cl] = 1;
                for (int a = 0; a < 4; a++) {
                    int nl = ants_dest(g, cl, ANTS_AIM[a][0], ANTS_AIM[a][1]);
                    if (!vsq[nl] && !inq[nl] && g->map[nl] != ANTS_WATER
                            && ants_distance(g, hl, nl) <= vr) {
                        inq[nl] = 1;
                        g->queue[tail++] = nl;
                    }
                }
            }
            free(inq);
        }
    }
    d->n_sets = 0;
    int a0 = g->sym_loc[0];
    for (int cell = 0; cell < g->cells; cell++) {
        if (visited[cell] || g->hill_at[cell] >= 0 || (starting && !vsq[cell])) {
            continue;
        }
        int orow = cell / g->cols - a0 / g->cols, ocol = cell % g->cols - a0 % g->cols;
        int rr[ANTS_MAXP], cc[ANTS_MAXP], locs[ANTS_MAXP];
        for (int e = 0; e < P; e++) {
            int r, c;
            ants_offset_aim(orow, ocol, g->sym_aim[e], &r, &c);
            int loc = ants_dest(g, g->sym_loc[e], r, c);
            rr[e] = loc / g->cols;
            cc[e] = loc % g->cols;
        }
        int n = pyset_order(rr, cc, P, g->cols, locs);
        int too_close = 0;
        for (int k = 1; k < n && !too_close; k++) {
            too_close = ants_distance(g, locs[0], locs[k]) == 1;
        }
        if (too_close) {
            continue;
        }
        if (!starting || n == P) {
            for (int k = 0; k < n; k++) {
                visited[locs[k]] = 1;
                d->locs[d->n_sets * P + k] = locs[k];
            }
            d->len[d->n_sets++] = n;
        }
    }
    free(visited);
    free(vsq);
    d->head = 0;
    d->size = 0;
    for (int s = 0; s < d->n_sets; s++) {
        d->buf[d->size++] = s;
    }
    d->buf[d->size++] = -1;
}

static inline int* ants_dq_at(AntsFoodDeque* d, int i) {
    return &d->buf[(d->head + i) % d->cap];
}

static int ants_dq_pop(AntsFoodDeque* d) {
    assert(d->size > 0);
    return *ants_dq_at(d, --d->size);
}

static void ants_dq_append(AntsFoodDeque* d, int v) {
    *ants_dq_at(d, d->size++) = v;
}

static void ants_dq_appendleft(AntsFoodDeque* d, int v) {
    d->head = (d->head + d->cap - 1) % d->cap;
    d->buf[d->head] = v;
    d->size++;
}

// do_food_symmetric / do_food_visible
static int ants_food_from_sets(Ants* g, AntsFoodDeque* d, int amount) {
    while (1) {
        int s = ants_dq_pop(d);
        if (s < 0) {
            for (int i = d->size - 1; i > 0; i--) {
                int j = pyrand_below(&g->rng, i + 1);
                int t = *ants_dq_at(d, i);
                *ants_dq_at(d, i) = *ants_dq_at(d, j);
                *ants_dq_at(d, j) = t;
            }
            ants_dq_appendleft(d, -1);
            s = ants_dq_pop(d);
            assert(s >= 0);
        }
        if (d->len[s] > amount) {
            ants_dq_append(d, s);
            ants_place_food(g);
            return amount;
        }
        amount -= d->len[s];
        ants_dq_appendleft(d, s);
        for (int k = 0; k < d->len[s]; k++) {
            ants_pend(g, d->locs[s * g->players + k]);
        }
    }
}

static int ants_do_food(Ants* g, int amount) {
    int P = g->players;
    if (g->cfg.food == ANTS_FOOD_NONE) {
        return amount;
    } else if (g->cfg.food == ANTS_FOOD_RANDOM) {
        for (int k = 0; k < amount; k++) {
            while (1) {
                int r = pyrand_below(&g->rng, g->rows), c = pyrand_below(&g->rng, g->cols);
                if (g->map[r * g->cols + c] == ANTS_LAND) {
                    ants_pend(g, r * g->cols + c);
                    break;
                }
            }
        }
        ants_place_food(g);
        return 0;
    } else if (g->cfg.food == ANTS_FOOD_SECTIONS) {
        int left = ants_pymod(amount, P), rounds = ants_pydiv(amount, P);
        for (int k = 0; k < rounds; k++) {
            for (int p = 0; p < P; p++) {
                int n = g->access_start[p + 1] - g->access_start[p];
                int loc = g->access[g->access_start[p] + pyrand_below(&g->rng, n)];
                if (g->map[loc] == ANTS_LAND) {
                    ants_pend(g, loc);
                }
            }
        }
        ants_place_food(g);
        return left;
    }
    if (!g->sym_init) {
        ants_build_food_sets(g, &g->sym_dq, 0);
        g->sym_init = 1;
    }
    return ants_food_from_sets(g, &g->sym_dq, amount);
}

// access_map(): squares closest to exactly one player, in BFS dict order
static void ants_build_access_map(Ants* g) {
    int* dist = g->tmp_a;
    int* players = g->tmp_b;    // bitmask
    int* order = g->tmp_c;
    int head = 0, tail = 0, n_order = 0;
    for (int cell = 0; cell < g->cells; cell++) {
        players[cell] = 0;
        dist[cell] = g->map[cell] == ANTS_WATER ? -2 : -1;
        if (g->map[cell] >= 0) {
            dist[cell] = 0;
            players[cell] = 1 << g->map[cell];
            order[n_order++] = cell;
            g->queue[tail++] = cell;
        }
    }
    while (head < tail) {
        int cl = g->queue[head++];
        for (int a = 0; a < 4; a++) {
            int nl = ants_dest(g, cl, ANTS_AIM[a][0], ANTS_AIM[a][1]);
            if (dist[nl] == -2) {
                continue;
            }
            if (dist[nl] == -1) {
                dist[nl] = dist[cl] + 1;
                players[nl] |= players[cl];
                order[n_order++] = nl;
                g->queue[tail++] = nl;
            } else if (dist[nl] == dist[cl] + 1) {
                players[nl] |= players[cl];
            }
        }
    }
    int k = 0;
    for (int p = 0; p < g->players; p++) {
        g->access_start[p] = k;
        for (int i = 0; i < n_order; i++) {
            if (players[order[i]] == (1 << p)) {
                g->access[k++] = order[i];
            }
        }
    }
    g->access_start[g->players] = k;
}

// Ants.__init__ (+ seed). Map must fit the capacity given to ants_new.
void ants_init(Ants* g, const AntsMap* m, AntsConfig cfg, long long engine_seed) {
    assert(m->rows * m->cols <= g->cap_cells && m->players <= g->cap_players);
    g->cfg = cfg;
    g->rows = m->rows;
    g->cols = m->cols;
    g->cells = m->rows * m->cols;
    g->players = m->players;
    pyrand_seed(&g->rng, engine_seed);
    g->food_rate = cfg.food_rate[0] + pyrand_below(&g->rng, cfg.food_rate[1] - cfg.food_rate[0] + 1);
    g->food_turn = cfg.food_turn[0] + pyrand_below(&g->rng, cfg.food_turn[1] - cfg.food_turn[0] + 1);
    g->food_start = cfg.food_start[0]
        + pyrand_below(&g->rng, cfg.food_start[1] - cfg.food_start[0] + 1);
    g->food_visible = cfg.food_visible[0]
        + pyrand_below(&g->rng, cfg.food_visible[1] - cfg.food_visible[0] + 1);
    g->food_extra = 0;
    g->turn = 0;
    g->done = 0;
    g->accept_orders = 0;
    g->cutoff = NULL;
    g->cutoff_bot = ANTS_LAND;
    g->cutoff_turns = 0;
    g->hill_kill = 0;
    g->have_ranking = 0;
    g->ranking_turn = g->winning_turn = 0;
    g->n_ants = g->n_killed = g->n_hills = g->n_food = g->n_pending = 0;
    g->have_sym = g->vis_init = g->sym_init = 0;
    g->seen_stamp = 0;
    int C = g->cells, P = g->players, water = 0;
    for (int i = 0; i < C; i++) {
        g->ant_at[i] = g->hill_at[i] = g->food_at[i] = -1;
        g->pending[i] = 0;
        g->seen[i] = 0;
        g->map[i] = m->grid[i] == '%' ? ANTS_WATER : ANTS_LAND;
        water += m->grid[i] == '%';
    }
    g->land_area = C - water;
    memset(g->vision, 0, sizeof(uint16_t) * P * C);
    memset(g->revealed, 0, (size_t)P * C);
    ants_build_stencils(g);

    // map_data['hills'] defaultdict: owners by first appearance, locs row-major
    int owner_order[ANTS_MAXP], n_owner = 0, ant_order[ANTS_MAXP], n_antown = 0;
    uint8_t seen_h[ANTS_MAXP] = {0}, seen_a[ANTS_MAXP] = {0};
    int has_ants = 0;
    for (int i = 0; i < C; i++) {
        char ch = m->grid[i];
        int ho = (ch >= '0' && ch <= '9') ? ch - '0' : (ch >= 'A' && ch <= 'J') ? ch - 'A' : -1;
        int ao = (ch >= 'a' && ch <= 'j') ? ch - 'a' : (ch >= 'A' && ch <= 'J') ? ch - 'A' : -1;
        if (ho >= 0 && !seen_h[ho]) {
            seen_h[ho] = 1;
            owner_order[n_owner++] = ho;
        }
        if (ao >= 0) {
            has_ants = 1;
            if (!seen_a[ao]) {
                seen_a[ao] = 1;
                ant_order[n_antown++] = ao;
            }
        }
    }
    assert(cfg.scenario || n_owner == P);
    g->n_hills_p0 = 0;
    for (int k = 0; k < n_owner; k++) {
        for (int i = 0; i < C; i++) {
            char ch = m->grid[i];
            int ho = (ch >= '0' && ch <= '9') ? ch - '0' : (ch >= 'A' && ch <= 'J') ? ch - 'A' : -1;
            if (ho != owner_order[k]) {
                continue;
            }
            int h = g->n_hills++;
            g->hill_loc[h] = i;
            g->hill_owner[h] = ho;
            g->hill_killed_by[h] = -1;
            g->hill_last_touched[h] = 0;
            g->hill_end_turn[h] = -1;
            g->hill_raze_points[h] = 0;
            g->hill_at[i] = h;
            g->n_hills_p0 += ho == 0;
            if (!cfg.scenario || !has_ants) {
                ants_spawn_on_hill(g, h);
            }
        }
    }
    if (cfg.scenario) {
        for (int k = 0; k < n_antown; k++) {
            for (int i = 0; i < C; i++) {
                char ch = m->grid[i];
                int ao = (ch >= 'a' && ch <= 'j') ? ch - 'a' : (ch >= 'A' && ch <= 'J') ? ch - 'A' : -1;
                if (ao == ant_order[k]) {
                    ants_add_ant(g, i, ao);
                }
            }
        }
        for (int i = 0; i < C; i++) {
            if (m->grid[i] == '*') {
                ants_add_food(g, i);
            }
        }
    }
    for (int p = 0; p < P; p++) {
        g->score[p] = g->n_hills_p0;
        g->bonus[p] = g->hive[p] = 0;
        g->stat_gathered[p] = g->stat_lost[p] = 0;
        g->killed[p] = 0;
        g->n_switch[p] = 1;
        g->rwater_n[p] = 0;
        for (int q = 0; q < ANTS_MAXP; q++) {
            g->switch_[p][q] = -1;
        }
        g->switch_[p][p] = 0;
    }
    if (cfg.food == ANTS_FOOD_SECTIONS) {
        ants_build_access_map(g);
    }
    ants_update_vision(g);
    ants_update_revealed(g);
}

void ants_count_ants(Ants* g) {
    memset(g->ant_count, 0, sizeof(g->ant_count));
    for (int i = 0; i < g->n_ants; i++) {
        g->ant_count[g->ant_owner[i]]++;
    }
}

int ants_is_alive(Ants* g, int p) {
    if (g->killed[p]) {
        return 0;
    }
    for (int i = 0; i < g->n_ants; i++) {
        if (g->ant_owner[i] == p) {
            return 1;
        }
    }
    return 0;
}

int ants_has_hill(Ants* g, int p) {
    for (int h = 0; h < g->n_hills; h++) {
        if (g->hill_owner[h] == p && g->hill_killed_by[h] < 0) {
            return 1;
        }
    }
    return 0;
}

// Validate and queue one order for this turn (validate_orders semantics)
int ants_order(Ants* g, int p, int row, int col, int dir) {
    assert(dir >= 0 && dir < 4);
    if (row < 0 || col < 0 || row >= g->rows || col >= g->cols) {
        return ANTS_ORDER_INVALID;
    }
    int loc = row * g->cols + col, stamp = g->seen_stamp + p + 1;
    if (g->seen[loc] == stamp || g->map[loc] != p) {
        return ANTS_ORDER_INVALID;
    }
    int dest = ants_dest(g, loc, ANTS_AIM[dir][0], ANTS_AIM[dir][1]);
    if (g->map[dest] == ANTS_FOOD || g->map[dest] == ANTS_WATER) {
        return ANTS_ORDER_IGNORED;
    }
    g->seen[loc] = stamp;
    int k = g->n_orders[p]++;
    g->order_loc[(size_t)p * g->cap_cells + k] = loc;
    g->order_dir[(size_t)p * g->cap_cells + k] = dir;
    return ANTS_ORDER_VALID;
}

static void ants_swap_ant_buffers(Ants* g) {
    int* t = g->ant_loc;
    g->ant_loc = g->ant_loc2;
    g->ant_loc2 = t;
    t = g->ant_prev;
    g->ant_prev = g->ant_prev2;
    g->ant_prev2 = t;
    int8_t* u = g->ant_owner;
    g->ant_owner = g->ant_owner2;
    g->ant_owner2 = u;
    u = g->ant_dir;
    g->ant_dir = g->ant_dir2;
    g->ant_dir2 = u;
}

static void ants_do_orders(Ants* g) {
    int* move = g->tmp_a;
    int* ordered = g->tmp_b;
    int m = 0;
    for (int i = 0; i < g->n_ants; i++) {
        ordered[i] = 0;
    }
    for (int p = 0; p < g->players; p++) {
        for (int k = 0; k < g->n_orders[p]; k++) {
            int i = g->ant_at[g->order_loc[(size_t)p * g->cap_cells + k]];
            g->ant_dir[i] = g->order_dir[(size_t)p * g->cap_cells + k];
            ordered[i] = 1;
            move[m++] = i;
        }
    }
    for (int i = 0; i < g->n_ants; i++) {
        if (!ordered[i]) {
            g->ant_dir[i] = ANTS_STAY;
            move[m++] = i;
        }
        g->map[g->ant_loc[i]] = ANTS_LAND;
        g->ant_at[g->ant_loc[i]] = -1;
    }
    for (int k = 0; k < m; k++) {
        int i = move[k], d = g->ant_dir[i];
        g->ant_prev[i] = g->ant_loc[i];
        if (d < 4) {
            g->ant_loc[i] = ants_dest(g, g->ant_loc[i], ANTS_AIM[d][0], ANTS_AIM[d][1]);
        }
        int loc = g->ant_loc[i];
        g->ant_at[loc] = g->ant_at[loc] == -1 ? i : -2;
    }
    int n = 0;
    for (int k = 0; k < m; k++) {
        int i = move[k], loc = g->ant_loc[i];
        if (g->ant_at[loc] == -2) {
            ants_kill(g, loc, g->ant_prev[i], g->ant_owner[i]);
            continue;
        }
        g->ant_loc2[n] = loc;
        g->ant_prev2[n] = g->ant_prev[i];
        g->ant_owner2[n] = g->ant_owner[i];
        g->ant_dir2[n] = g->ant_dir[i];
        n++;
    }
    for (int k = 0; k < m; k++) {
        g->ant_at[g->ant_loc[move[k]]] = -1;
    }
    ants_swap_ant_buffers(g);
    g->n_ants = n;
    for (int i = 0; i < n; i++) {
        g->map[g->ant_loc[i]] = g->ant_owner[i];
        g->ant_at[g->ant_loc[i]] = i;
        g->ant_dead[i] = 0;
    }
}

static void ants_kill_index(Ants* g, int i) {
    if (g->ant_dead[i]) {
        return;
    }
    g->ant_dead[i] = 1;
    g->ant_at[g->ant_loc[i]] = -1;
    ants_kill(g, g->ant_loc[i], g->ant_prev[i], g->ant_owner[i]);
}

static void ants_compact(Ants* g) {
    int n = 0;
    for (int i = 0; i < g->n_ants; i++) {
        if (g->ant_dead[i]) {
            continue;
        }
        g->ant_loc[n] = g->ant_loc[i];
        g->ant_prev[n] = g->ant_prev[i];
        g->ant_owner[n] = g->ant_owner[i];
        g->ant_dir[n] = g->ant_dir[i];
        g->ant_dead[n] = 0;
        g->ant_at[g->ant_loc[n]] = n;
        n++;
    }
    g->n_ants = n;
}

// Number of nearby_ants(loc, attackradius, owner) entries (with duplicates)
static int ants_enemy_count(Ants* g, int i) {
    int r = g->ant_loc[i] / g->cols, c = g->ant_loc[i] % g->cols, o = g->ant_owner[i], n = 0;
    for (int k = 0; k < g->att_n; k++) {
        int v = g->map[ants_off(g, r, c, &g->att[2 * k])];
        n += v >= 0 && v != o;
    }
    return n;
}

static void ants_do_attack(Ants* g) {
    int* cnt = g->tmp_c;
    int N = g->n_ants;
    for (int i = 0; i < N; i++) {
        cnt[i] = ants_enemy_count(g, i);
    }
    int* kill = g->tmp_b;
    int nk = 0;
    if (g->cfg.attack == ANTS_ATTACK_FOCUS || g->cfg.attack == ANTS_ATTACK_SUPPORT) {
        for (int i = 0; i < N; i++) {
            if (cnt[i] == 0) {
                continue;
            }
            if (g->cfg.attack == ANTS_ATTACK_SUPPORT) {
                kill[nk++] = i;
                continue;
            }
            int r = g->ant_loc[i] / g->cols, c = g->ant_loc[i] % g->cols, o = g->ant_owner[i];
            int mn = 1 << 30;
            for (int k = 0; k < g->att_n; k++) {
                int cell = ants_off(g, r, c, &g->att[2 * k]);
                int v = g->map[cell];
                if (v >= 0 && v != o && cnt[g->ant_at[cell]] < mn) {
                    mn = cnt[g->ant_at[cell]];
                }
            }
            if (mn <= cnt[i]) {
                kill[nk++] = i;
            }
        }
    } else if (g->cfg.attack == ANTS_ATTACK_DAMAGE) {
        assert(g->att_n <= 40);
        long long L = 1;
        for (long long k = 2; k <= g->att_n; k++) {
            long long a = L, b = k;
            while (b) {
                long long t = a % b;
                a = b;
                b = t;
            }
            L = L / a * k;
        }
        for (int i = 0; i < N; i++) {
            g->dmg[i] = 0;
        }
        for (int i = 0; i < N; i++) {
            if (cnt[i] == 0) {
                continue;
            }
            int r = g->ant_loc[i] / g->cols, c = g->ant_loc[i] % g->cols, o = g->ant_owner[i];
            for (int k = 0; k < g->att_n; k++) {
                int cell = ants_off(g, r, c, &g->att[2 * k]);
                int v = g->map[cell];
                if (v >= 0 && v != o) {
                    g->dmg[g->ant_at[cell]] += L / cnt[i];
                }
            }
        }
        for (int i = 0; i < N; i++) {
            if (g->dmg[i] >= L) {
                kill[nk++] = i;
            }
        }
    } else {
        // closest (Python 2 semantics: values() is a list snapshot)
        int* group = g->tmp_a;
        int* stack = g->queue;
        int* mark = (int*)ants_xalloc(sizeof(int) * (N + 1));
        int stamp = 0;
        for (int d = 1; d < g->cfg.attackradius2; d++) {
            for (int i = 0; i < N; i++) {
                if (cnt[i] == 0 || g->ant_dead[i]) {
                    continue;
                }
                stamp++;
                int ng = 0, sp = 0;
                group[ng++] = i;
                mark[i] = stamp;
                stack[sp++] = i;
                while (sp) {
                    int a = stack[--sp];
                    int r = g->ant_loc[a] / g->cols, c = g->ant_loc[a] % g->cols;
                    for (int k = 0; k < g->att_n; k++) {
                        int cell = ants_off(g, r, c, &g->att[2 * k]);
                        int v = g->map[cell];
                        if (v < 0 || v == g->ant_owner[a]) {
                            continue;
                        }
                        int e = g->ant_at[cell];
                        if (e < 0 || g->ant_dead[e] || mark[e] == stamp
                                || ants_distance(g, g->ant_loc[a], cell) != d) {
                            continue;
                        }
                        mark[e] = stamp;
                        group[ng++] = e;
                        stack[sp++] = e;
                    }
                }
                if (ng > 1) {
                    for (int k = 0; k < ng; k++) {
                        int e = group[k];
                        g->ant_dead[e] = 1;
                        g->ant_at[g->ant_loc[e]] = -1;
                        ants_kill(g, g->ant_loc[e], g->ant_prev[e], g->ant_owner[e]);
                    }
                }
            }
        }
        free(mark);
    }
    for (int k = 0; k < nk; k++) {
        ants_kill_index(g, kill[k]);
    }
    ants_compact(g);
}

static void ants_raze(Ants* g, int h, int by) {
    g->hill_end_turn[h] = g->turn;
    g->hill_killed_by[h] = by;
    g->score[by] += ANTS_HILL_POINTS;
    if (!g->hill_raze_points[h]) {
        g->hill_raze_points[h] = 1;
        g->score[g->hill_owner[h]] += ANTS_RAZE_POINTS;
    }
    g->cutoff_turns = 0;
}

static void ants_do_raze_hills(Ants* g) {
    for (int h = 0; h < g->n_hills; h++) {
        int i = g->ant_at[g->hill_loc[h]];
        if (i < 0) {
            continue;
        }
        if (g->ant_owner[i] == g->hill_owner[h]) {
            g->hill_last_touched[h] = g->turn;
        } else if (g->hill_killed_by[h] < 0) {
            ants_raze(g, h, g->ant_owner[i]);
        }
    }
}

static void ants_do_spawn(Ants* g) {
    int hs[ANTS_MAXP * 8];
    int* hl = g->n_hills > ANTS_MAXP * 8 ? g->tmp_a : hs;
    double rnd_small[ANTS_MAXP * 8];
    double* rnd = g->n_hills > ANTS_MAXP * 8 ? (double*)ants_xalloc(sizeof(double) * g->n_hills) : rnd_small;
    for (int p = 0; p < g->players; p++) {
        int n = 0;
        for (int h = 0; h < g->n_hills; h++) {
            if (g->hill_owner[h] == p && g->hill_killed_by[h] < 0) {
                rnd[n] = pyrand_random(&g->rng);
                hl[n++] = h;
            }
        }
        // stable insertion sort on (last_touched, random())
        for (int i = 1; i < n; i++) {
            int h = hl[i];
            double x = rnd[i];
            int j = i - 1;
            while (j >= 0 && (g->hill_last_touched[hl[j]] > g->hill_last_touched[h]
                    || (g->hill_last_touched[hl[j]] == g->hill_last_touched[h] && rnd[j] > x))) {
                hl[j + 1] = hl[j];
                rnd[j + 1] = rnd[j];
                j--;
            }
            hl[j + 1] = h;
            rnd[j + 1] = x;
        }
        for (int i = 0; i < n; i++) {
            if (g->hive[p] > 0 && g->ant_at[g->hill_loc[hl[i]]] < 0) {
                g->hive[p]--;
                ants_spawn_on_hill(g, hl[i]);
            }
        }
    }
    if (rnd != rnd_small) {
        free(rnd);
    }
}

static void ants_do_gather(Ants* g) {
    int* snap = g->tmp_a;
    int n = g->n_food;
    memcpy(snap, g->food_loc, sizeof(int) * n);
    for (int k = 0; k < n; k++) {
        int loc = snap[k], r = loc / g->cols, c = loc % g->cols, mask = 0;
        for (int j = 0; j < g->spawn_n; j++) {
            int v = g->map[ants_off(g, r, c, &g->spawn[2 * j])];
            if (v >= 0) {
                mask |= 1 << v;
            }
        }
        if (mask && !(mask & (mask - 1))) {
            ants_remove_food(g, loc, __builtin_ctz(mask));
        } else if (mask) {
            ants_remove_food(g, loc, -1);
        }
    }
}

void ants_calc_significant_turns(Ants* g) {
    int P = g->players, mx = g->score[0], changed = !g->have_ranking;
    for (int p = 0; p < P; p++) {
        mx = g->score[p] > mx ? g->score[p] : mx;
    }
    int rank[ANTS_MAXP];
    uint8_t win[ANTS_MAXP];
    int wchanged = !g->have_ranking;
    for (int p = 0; p < P; p++) {
        rank[p] = 0;
        for (int q = 0; q < P; q++) {
            rank[p] += g->score[q] > g->score[p];
        }
        win[p] = g->score[p] == mx;
        changed |= g->have_ranking && rank[p] != g->ranking[p];
        wchanged |= g->have_ranking && win[p] != g->winning[p];
    }
    if (changed) {
        g->ranking_turn = g->turn;
    }
    if (wchanged) {
        g->winning_turn = g->turn;
    }
    memcpy(g->ranking, rank, sizeof(rank));
    memcpy(g->winning, win, sizeof(win));
    g->have_ranking = 1;
}

static void ants_finish_turn(Ants* g) {
    ants_do_orders(g);
    ants_do_attack(g);
    ants_do_raze_hills(g);
    ants_do_spawn(g);
    ants_do_gather(g);
    g->food_extra += (long long)g->food_rate * g->players;
    int food_now = (int)(g->food_extra / g->food_turn);
    int left = ants_do_food(g, food_now);
    g->food_extra -= (long long)(food_now - left) * g->food_turn;
    ants_update_vision(g);
    ants_update_revealed(g);

    // cutoff: first pop_count key (dict order) holding cutoff_percent
    int keys[ANTS_MAXP + 1], nkeys = 0;
    long long cnt[ANTS_MAXP + 1] = {0};
    int key_of[ANTS_MAXP];
    for (int p = 0; p < ANTS_MAXP; p++) {
        key_of[p] = -1;
    }
    for (int i = 0; i < g->n_ants; i++) {
        int o = g->ant_owner[i];
        if (key_of[o] < 0) {
            key_of[o] = nkeys;
            keys[nkeys++] = o;
        }
        cnt[key_of[o]]++;
    }
    for (int h = 0; h < g->n_hills; h++) {
        if (g->hill_killed_by[h] >= 0) {
            continue;
        }
        int o = g->hill_owner[h];
        if (key_of[o] < 0) {
            key_of[o] = nkeys;
            keys[nkeys++] = o;
        }
        cnt[key_of[o]] += g->hive[o];
    }
    keys[nkeys] = ANTS_FOOD;
    cnt[nkeys++] = g->n_food;
    long long total = 0;
    for (int k = 0; k < nkeys; k++) {
        total += cnt[k];
    }
    int hit = 0;
    for (int k = 0; k < nkeys && !hit; k++) {
        if ((double)cnt[k] >= (double)total * g->cfg.cutoff_percent) {
            hit = 1;
            if (g->cutoff_bot == keys[k]) {
                g->cutoff_turns += !g->hill_kill;
            } else {
                g->cutoff_bot = keys[k];
                g->cutoff_turns = 1;
            }
        }
    }
    if (!hit) {
        g->cutoff_bot = ANTS_LAND;
        g->cutoff_turns = 0;
    }
    ants_calc_significant_turns(g);
}

int ants_is_rank_stabilized(Ants* g) {
    for (int p = 0; p < g->players; p++) {
        if (!ants_is_alive(g, p) || !ants_has_hill(g, p)) {
            continue;
        }
        int max_score = g->score[p];
        for (int h = 0; h < g->n_hills; h++) {
            if (g->hill_killed_by[h] < 0 && g->hill_owner[h] != p) {
                max_score += ANTS_HILL_POINTS;
            }
        }
        for (int q = 0; q < g->players; q++) {
            if (q == p) {
                continue;
            }
            int min_score = g->score[q];
            for (int h = 0; h < g->n_hills; h++) {
                if (g->hill_killed_by[h] < 0 && g->hill_owner[h] == q) {
                    min_score += ANTS_RAZE_POINTS;
                }
            }
            if ((g->score[p] < g->score[q] && max_score >= min_score)
                    || (g->score[p] == g->score[q] && max_score > min_score)) {
                return 0;
            }
        }
    }
    return 1;
}

int ants_game_over(Ants* g) {
    int alive = 0;
    for (int p = 0; p < g->players; p++) {
        alive += ants_is_alive(g, p);
    }
    if (alive < 1) {
        g->cutoff = "extermination";
        return 1;
    }
    if (alive == 1) {
        g->cutoff = "lone survivor";
        return 1;
    }
    if (g->cutoff_turns >= g->cfg.cutoff_turn) {
        g->cutoff = g->cutoff_bot == ANTS_FOOD ? "food not being gathered" : "ants not razing hills";
        return 1;
    }
    if (ants_is_rank_stabilized(g)) {
        g->cutoff = "rank stabilized";
        return 1;
    }
    return 0;
}

// Engine kicks a player (crash/timeout in the original)
void ants_kill_player(Ants* g, int p) {
    g->killed[p] = 1;
    for (int h = 0; h < g->n_hills; h++) {
        if (g->hill_owner[h] == p && !g->hill_raze_points[h]) {
            g->hill_raze_points[h] = 1;
            g->score[p] += ANTS_RAZE_POINTS;
        }
    }
}

static void ants_finish_game(Ants* g) {
    int alive = 0, last = -1;
    for (int p = 0; p < g->players; p++) {
        if (ants_is_alive(g, p)) {
            alive++;
            last = p;
        }
    }
    if (alive == 1) {
        for (int h = 0; h < g->n_hills; h++) {
            if (g->hill_owner[h] == last) {
                continue;
            }
            if (g->hill_killed_by[h] < 0) {
                g->bonus[last] += ANTS_HILL_POINTS;
                g->hill_killed_by[h] = last;
            }
            if (!g->hill_raze_points[h]) {
                g->bonus[g->hill_owner[h]] += ANTS_RAZE_POINTS;
                g->hill_raze_points[h] = 1;
            }
        }
        for (int p = 0; p < g->players; p++) {
            g->score[p] += g->bonus[p];
        }
    }
    ants_calc_significant_turns(g);
    if (!g->cutoff) {
        g->cutoff = "turn limit reached";
    }
    g->done = 1;
}

// worker/engine.py shuffles alive bots each turn with the global RNG
static void ants_shuffle_bots(Ants* g) {
    int n = 0;
    for (int p = 0; p < g->players; p++) {
        n += ants_is_alive(g, p);
    }
    for (int i = n - 1; i > 0; i--) {
        pyrand_below(&g->rng, i + 1);
    }
}

// Turn 0 of the engine loop: start_game() and the first game_over() check
void ants_start(Ants* g) {
    if (g->cfg.food != ANTS_FOOD_NONE) {
        int starting = g->food_start
            ? g->land_area / g->food_start - g->food_visible * g->players : 0;
        if (g->food_visible > 0) {
            ants_build_food_sets(g, &g->vis_dq, 1);
            g->vis_init = 1;
            ants_food_from_sets(g, &g->vis_dq, g->food_visible * g->players);
        }
        ants_do_food(g, starting);
    }
    ants_shuffle_bots(g);
    if (ants_game_over(g) || g->cfg.turns <= 0) {
        ants_finish_game(g);
    }
}

// Engine loop up to move collection. render_text: per-player buffers for the
// bot input text (NULL to skip; enemy numbering side effects still apply).
void ants_begin_turn(Ants* g, char** render_text, int render_size) {
    assert(!g->done);
    for (int p = 0; p < g->players; p++) {
        if (ants_is_alive(g, p)) {
            ants_render(g, p, render_text ? render_text[p] : NULL, render_size);
        }
    }
    g->turn++;
    g->n_killed = 0;
    g->hill_kill = 0;
    g->seen_stamp += ANTS_MAXP + 1;
    for (int p = 0; p < g->players; p++) {
        g->rwater_n[p] = 0;
        g->n_orders[p] = 0;
        g->alive_begin[p] = ants_is_alive(g, p);
    }
    ants_shuffle_bots(g);
    g->accept_orders = !ants_game_over(g);
}

int ants_can_order(Ants* g, int p) {
    return g->accept_orders && g->alive_begin[p];
}

// finish_turn(), eliminated-bot renders, game_over()/turn limit
void ants_end_turn(Ants* g) {
    for (int p = 0; p < g->players; p++) {
        if (!ants_can_order(g, p)) {
            g->n_orders[p] = 0;
        }
    }
    ants_finish_turn(g);
    for (int p = 0; p < g->players; p++) {
        if (g->alive_begin[p] && !ants_is_alive(g, p)) {
            ants_render(g, p, NULL, 0);
        }
    }
    if (ants_game_over(g) || g->turn >= g->cfg.turns) {
        ants_finish_game(g);
    }
}
