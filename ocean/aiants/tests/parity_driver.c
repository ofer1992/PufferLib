// Replays a ref_runner.py orders file on the C engine and prints the same dump.
// Usage: parity_driver MAP ORDERS key=value...   (same keys as ref_runner.py)
#include "../ants_engine.h"

static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(n + 1);
    fread(buf, 1, n, f);
    buf[n] = 0;
    fclose(f);
    return buf;
}

static const char* kv_get(int argc, char** argv, const char* key, const char* def) {
    size_t n = strlen(key);
    for (int i = 3; i < argc; i++) {
        if (!strncmp(argv[i], key, n) && argv[i][n] == '=') {
            return argv[i] + n + 1;
        }
    }
    return def;
}

static void dump(Ants* g, int turn, char** renders, int* have_render) {
    printf("T %d\n", turn);
    for (int p = 0; p < g->players; p++) {
        if (have_render[p]) {
            printf("U %d\n%s/U\n", p, renders[p]);
        }
    }
    int P = g->players;
    printf("S");
    for (int p = 0; p < P; p++) {
        printf(" %d", g->score[p]);
    }
    printf("\nB");
    for (int p = 0; p < P; p++) {
        printf(" %d", g->bonus[p]);
    }
    printf("\nH");
    for (int p = 0; p < P; p++) {
        printf(" %d", g->hive[p]);
    }
    printf("\nX %lld\n", g->food_extra);
    printf("C %d %d %d %s\n", g->cutoff_bot, g->cutoff_turns, g->hill_kill,
        g->cutoff ? g->cutoff : "None");
    printf("W %d %d\n", g->winning_turn, g->ranking_turn);
    printf("R %llu\n", (unsigned long long)pyrand_hash(&g->rng));
    for (int h = 0; h < g->n_hills; h++) {
        printf("h %d %d %d %d %d %d %d\n", g->hill_loc[h] / g->cols, g->hill_loc[h] % g->cols,
            g->hill_owner[h], g->hill_killed_by[h], g->hill_last_touched[h],
            g->hill_end_turn[h], g->hill_raze_points[h]);
    }
    for (int i = 0; i < g->n_ants; i++) {
        printf("a %d %d %d\n", g->ant_loc[i] / g->cols, g->ant_loc[i] % g->cols, g->ant_owner[i]);
    }
    for (int cell = 0; cell < g->cells; cell++) {
        if (g->food_at[cell] >= 0) {
            printf("f %d %d\n", cell / g->cols, cell % g->cols);
        }
    }
    for (int cell = 0; cell < g->cells; cell++) {
        if (g->pending[cell]) {
            printf("p %d %d %d\n", cell / g->cols, cell % g->cols, g->pending[cell]);
        }
    }
    int* tmp = malloc(sizeof(int) * (g->n_killed + 1));
    for (int k = 0; k < g->n_killed; k++) {
        tmp[k] = g->killed_loc[k] * 16 + g->killed_owner[k];
    }
    qsort(tmp, g->n_killed, sizeof(int), ants_cmp_int);
    for (int k = 0; k < g->n_killed; k++) {
        printf("d %d %d %d\n", tmp[k] / 16 / g->cols, tmp[k] / 16 % g->cols, tmp[k] % 16);
    }
    free(tmp);
    for (int p = 0; p < P; p++) {
        unsigned long long vn = 0, vs = 0, vq = 0, rn = 0, rs = 0, rq = 0;
        for (unsigned long long c = 0; c < (unsigned long long)g->cells; c++) {
            if (g->vision[(size_t)p * g->cells + c]) {
                vn++;
                vs += c;
                vq += c * c;
            }
            if (g->revealed[(size_t)p * g->cells + c]) {
                rn++;
                rs += c;
                rq += c * c;
            }
        }
        printf("V %d %llu %llu %llu %llu %llu %llu\n", p, vn, vs, vq, rn, rs, rq);
        printf("Q %d", p);
        for (int q = 0; q < P; q++) {
            printf(" %d", g->switch_[p][q]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: parity_driver MAP ORDERS key=value...\n");
        return 1;
    }
    char* text = read_file(argv[1]);
    AntsMap m;
    int err = ants_parse_map(&m, text);
    if (err) {
        fprintf(stderr, "map parse error %d\n", err);
        return 1;
    }
    AntsConfig cfg = ants_default_config();
    cfg.turns = atoi(kv_get(argc, argv, "turns", "200"));
    cfg.viewradius2 = atoi(kv_get(argc, argv, "viewradius2", "77"));
    cfg.attackradius2 = atoi(kv_get(argc, argv, "attackradius2", "5"));
    cfg.spawnradius2 = atoi(kv_get(argc, argv, "spawnradius2", "1"));
    cfg.cutoff_turn = atoi(kv_get(argc, argv, "cutoff_turn", "150"));
    cfg.cutoff_percent = atof(kv_get(argc, argv, "cutoff_percent", "0.85"));
    cfg.scenario = atoi(kv_get(argc, argv, "scenario", "0"));
    const char* attack = kv_get(argc, argv, "attack", "focus");
    const char* food = kv_get(argc, argv, "food", "symmetric");
    const char* attacks[] = {"focus", "closest", "support", "damage"};
    const char* foods[] = {"none", "random", "sections", "symmetric"};
    for (int i = 0; i < 4; i++) {
        if (!strcmp(attack, attacks[i])) {
            cfg.attack = i;
        }
        if (!strcmp(food, foods[i])) {
            cfg.food = i;
        }
    }
    long long seed = atoll(kv_get(argc, argv, "seed", "0"));
    Ants* g = ants_new(m.rows * m.cols, m.players);
    ants_init(g, &m, cfg, seed);

    FILE* of = fopen(argv[2], "r");
    if (!of) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    int rsize = 1 << 22;
    char* renders[ANTS_MAXP];
    int have_render[ANTS_MAXP] = {0};
    for (int p = 0; p < m.players; p++) {
        renders[p] = calloc(rsize, 1);
    }
    ants_start(g);
    dump(g, 0, renders, have_render);
    char line[256];
    while (!g->done) {
        for (int p = 0; p < m.players; p++) {
            have_render[p] = ants_is_alive(g, p);
            renders[p][0] = 0;
        }
        ants_begin_turn(g, renders, rsize);
        for (int p = 0; p < m.players; p++) {
            if (!ants_can_order(g, p)) {
                continue;
            }
            int t, b, n;
            if (!fgets(line, sizeof(line), of) || sscanf(line, "O %d %d %d", &t, &b, &n) != 3
                    || t != g->turn || b != p) {
                fprintf(stderr, "orders file out of sync at turn %d player %d\n", g->turn, p);
                return 1;
            }
            for (int k = 0; k < n; k++) {
                int r, c;
                char d;
                fgets(line, sizeof(line), of);
                sscanf(line, "o %d %d %c", &r, &c, &d);
                ants_order(g, p, r, c, (int)(strchr(ANTS_DIR_CHARS, d) - ANTS_DIR_CHARS));
            }
        }
        ants_end_turn(g);
        dump(g, g->turn, renders, have_render);
    }
    printf("END %s\n", g->cutoff);
    return 0;
}
