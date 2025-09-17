// Weighted A* solver with partial best output and metrics
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <chrono>
#include <climits>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

struct Operation { int x, y, n; };
struct PairPos { int r1=-1,c1=-1,r2=-1,c2=-1; };

// Weighted factor for A*: f = g + w*h_eff
static double W_WEIGHT = 2.0;       
static double ALPHA = 0.0;          // h_eff = (1-ALPHA)*hsum + ALPHA*hcount
enum TieMode { TB_H_MIN=0, TB_H_MAX=1, TB_G_MIN=2, TB_G_MAX=3 };
static TieMode TIE_BREAK = TB_H_MIN; // tie-breaker on equal f

// -------- Parsing --------
static bool parse_input(const std::string& filename, int& size, std::vector<std::vector<int>>& grid) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    try {
        size_t pos = content.find("size");
        if (pos == std::string::npos) throw std::runtime_error("size not found");
        pos = content.find(":", pos);
        if (pos == std::string::npos) throw std::runtime_error("size colon not found");
        size_t end_pos = content.find_first_of(",}", pos);
        if (end_pos == std::string::npos) throw std::runtime_error("size value not found");
        size = std::stoi(content.substr(pos + 1, end_pos - (pos + 1)));

        pos = content.find("entities");
        if (pos == std::string::npos) throw std::runtime_error("entities not found");
        pos = content.find("[", pos);
        if (pos == std::string::npos) throw std::runtime_error("entities opening bracket not found");

        grid.assign(size, std::vector<int>(size));
        for (int i = 0; i < size; ++i) {
            pos = content.find("[", pos + 1);
            if (pos == std::string::npos) throw std::runtime_error("row opening bracket not found");
            for (int j = 0; j < size; ++j) {
                end_pos = content.find_first_of(",]", pos + 1);
                if (end_pos == std::string::npos) throw std::runtime_error("number not found");
                grid[i][j] = std::stoi(content.substr(pos + 1, end_pos - (pos + 1)));
                pos = end_pos;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// -------- Zobrist Hashing --------
struct Zobrist {
    int N, Vmax;
    std::vector<uint64_t> table; // size: N*N*(Vmax+1)
    Zobrist(int n, int vmax): N(n), Vmax(vmax), table((size_t)n*n*(vmax+1)) {
        std::mt19937_64 rng(0x9e3779b97f4a7c15ull ^ (uint64_t)n << 32 ^ (uint64_t)vmax);
        for (auto &x: table) x = rng();
    }
    inline uint64_t at(int r, int c, int v) const {
        return table[((size_t)r*N + c)*(Vmax+1) + v];
    }
    uint64_t hash_grid(const std::vector<std::vector<int>>& g) const {
        uint64_t h = 0;
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) h ^= at(r,c,g[r][c]);
        return h;
    }
};

// -------- Heuristic and helpers --------
static inline int contrib_sum(const PairPos& p){
    int d = std::abs(p.r1 - p.r2) + std::abs(p.c1 - p.c2);
    return d>1 ? (d-1) : 0;
}
static inline int contrib_cnt(const PairPos& p){
    int d = std::abs(p.r1 - p.r2) + std::abs(p.c1 - p.c2);
    return d>1 ? 1 : 0;
}

static inline bool in_sub(int r,int c,int y,int x,int n){ return (r>=y && r<y+n && c>=x && c<x+n); }
static inline std::pair<int,int> rot_map(int r,int c,int y,int x,int n){
    int i = r - y, j = c - x; return { y + j, x + (n - 1 - i) };
}

static int compute_hsum_full(const std::vector<PairPos>& pos){
    long long s=0; for (const auto& p: pos) if (p.r1>=0 && p.r2>=0) s += contrib_sum(p); return (int)s;
}
static int compute_hcnt_full(const std::vector<PairPos>& pos){
    long long c=0; for (const auto& p: pos) if (p.r1>=0 && p.r2>=0) c += contrib_cnt(p); return (int)c;
}

// -------- State --------
struct Node {
    int g; int hsum; int hcnt; double f; uint64_t hash; // minimal state for PQ
};

static inline double h_eff(const Node& n){ return (1.0-ALPHA)*(double)n.hsum + ALPHA*(double)n.hcnt; }
struct PQComp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.f != b.f) return a.f > b.f;
        // tie-breaker
        switch (TIE_BREAK){
            case TB_H_MIN: return h_eff(a) > h_eff(b);
            case TB_H_MAX: return h_eff(a) < h_eff(b);
            case TB_G_MIN: return a.g > b.g;
            case TB_G_MAX: return a.g < b.g;
        }
        return false;
    }
};

struct ParentInfo { uint64_t parent_hash; Operation op; bool has_parent=false; };

// -------- Focused candidate generation --------
static void worst_pairs(const std::vector<PairPos>& pos, int k, std::vector<int>& out_idx){
    std::vector<std::pair<int,int>> tmp; tmp.reserve(pos.size());
    for (int i=0;i<(int)pos.size();++i){ if (pos[i].r1>=0 && pos[i].r2>=0){ tmp.push_back({contrib_sum(pos[i]), i}); } }
    std::nth_element(tmp.begin(), tmp.begin()+std::min(k,(int)tmp.size()), tmp.end(), [](auto&a,auto&b){return a.first>b.first;});
    out_idx.clear(); int limit = std::min(k,(int)tmp.size());
    for (int i=0;i<limit;i++) out_idx.push_back(tmp[i].second);
}

static void gen_candidates(int N, const std::vector<std::vector<int>>& grid, const std::vector<PairPos>& pos, int maxSmallN, int cap, std::vector<Operation>& ops){
    ops.clear();
    std::set<uint64_t> seen; seen.clear();
    auto key = [](int x,int y,int n){ return ( (uint64_t)n<<40 ) ^ ( (uint64_t)x<<20 ) ^ (uint64_t)y; };

    // target worst pairs
    std::vector<int> idx; worst_pairs(pos, 4, idx);
    for (int id: idx){
        const auto& p = pos[id];
        int rmin = std::min(p.r1,p.r2), rmax = std::max(p.r1,p.r2);
        int cmin = std::min(p.c1,p.c2), cmax = std::max(p.c1,p.c2);
        for (int n=2;n<=std::min(N, maxSmallN);++n){
            // try include both endpoints if possible
            if ((rmax-rmin) < n && (cmax-cmin) < n){
                int y0 = std::max(0, std::min(rmin, rmax - n + 1));
                int x0 = std::max(0, std::min(cmin, cmax - n + 1));
                for (int dy=0; dy<2; ++dy){
                    for (int dx=0; dx<2; ++dx){
                        int y = std::min(std::max(0, y0+dy), N-n);
                        int x = std::min(std::max(0, x0+dx), N-n);
                        uint64_t k = key(x,y,n); if (seen.insert(k).second) ops.push_back({x,y,n});
                    }
                }
            }
            // include each endpoint
            for (int t=0;t<2;++t){
                int r = (t==0? p.r1: p.r2), c = (t==0? p.c1: p.c2);
                int yL = std::max(0, r - n + 1), yR = std::min(r, N - n);
                int xL = std::max(0, c - n + 1), xR = std::min(c, N - n);
                int y = (yL+yR)/2; int x=(xL+xR)/2; // center-ish
                uint64_t k = key(x,y,n); if (seen.insert(k).second) ops.push_back({x,y,n});
            }
            if ((int)ops.size() >= cap) return;
        }
        if ((int)ops.size() >= cap) break;
    }
    // fallback: a few systematic small windows
    for (int n=2;n<=std::min(N, maxSmallN) && (int)ops.size()<cap; ++n){
        for (int y=0;y<=N-n && (int)ops.size()<cap; y+=std::max(1, n-1)){
            for (int x=0;x<=N-n && (int)ops.size()<cap; x+=std::max(1, n-1)){
                uint64_t k = key(x,y,n); if (seen.insert(k).second) ops.push_back({x,y,n});
            }
        }
    }
}

// rotate grid in-place CW and update hash incrementally
static void rotate_apply(std::vector<std::vector<int>>& grid, int x,int y,int n, const Zobrist& zob, uint64_t& h){
    // apply via layer swap, updating hash by xor-out old / xor-in new
    // For simplicity, compute by copying subgrid then writing back with hash update
    std::vector<std::vector<int>> sub(n, std::vector<int>(n));
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) sub[i][j] = grid[y+i][x+j];
    for (int i=0;i<n;i++) for (int j=0;j<n;j++){
        int r=y+i,c=x+j; int v_old = grid[r][c]; h ^= zob.at(r,c,v_old);
        int v_new = sub[n-1-j][i]; grid[r][c]=v_new; h ^= zob.at(r,c,v_new);
    }
}

// helper: emit solution JSON (supports solved/partial)
static void emit_solution_json(const std::vector<Operation>& ops, bool solved, bool partial,
                              int hsum, int hcnt, double heff,
                              double time_s, long long nodes_expanded,
                              long long nodes_generated, size_t open_max,
                              double peak_rss_mb,
                              int N){
    // trivial compression: remove blocks of 4 identical rotations
    std::vector<Operation> out;
    for (size_t i=0;i<ops.size();){
        size_t j=i; while (j<ops.size() && ops[j].x==ops[i].x && ops[j].y==ops[i].y && ops[j].n==ops[i].n) j++;
        size_t keep = (j-i)%4; for (size_t k=0;k<keep;k++) out.push_back(ops[i]); i=j;
    }
    std::cout << "{\n";
    std::cout << "  \"solved\": " << (solved?"true":"false") << ",\n";
    std::cout << "  \"partial\": " << (partial?"true":"false") << ",\n";
    std::cout << "  \"board_size\": " << N << ",\n";
    std::cout << "  \"h_sum\": " << hsum << ",\n";
    std::cout << "  \"h_count\": " << hcnt << ",\n";
    std::cout << "  \"h_eff\": " << heff << ",\n";
    std::cout << "  \"ops\": [\n";
    for (size_t i=0;i<out.size();++i){ const auto& op = out[i];
        std::cout << "    {\"x\": " << op.y << ", \"y\": " << op.x << ", \"n\": " << op.n << "}";
        if (i+1<out.size()) std::cout << ","; std::cout << "\n"; }
    std::cout << "  ],\n";
    std::cout << "  \"metrics\": {\n";
    std::cout << "    \"time_s\": " << time_s << ",\n";
    std::cout << "    \"nodes_expanded\": " << nodes_expanded << ",\n";
    std::cout << "    \"nodes_generated\": " << nodes_generated << ",\n";
    std::cout << "    \"open_max\": " << open_max << ",\n";
    std::cout << "    \"peak_rss_mb\": " << peak_rss_mb << "\n";
    std::cout << "  }\n";
    std::cout << "}" << std::endl;
}

// -------- Main --------
int main(int argc, char* argv[]){
    if (argc != 2) { std::cerr << "Usage: " << argv[0] << " <problem_file.json>" << std::endl; return 1; }
    if (const char* w = std::getenv("WASTAR_W")) { try { W_WEIGHT = std::max(1.0, std::stod(w)); } catch (...) {} }
    if (const char* a = std::getenv("ALPHA")) { try { ALPHA = std::clamp(std::stod(a), 0.0, 1.0); } catch (...) {} }
    if (const char* a = std::getenv("WASTAR_ALPHA")) { try { ALPHA = std::clamp(std::stod(a), 0.0, 1.0); } catch (...) {} }
    if (const char* t = std::getenv("TIE_BREAK")) {
        std::string s(t);
        if (s == "h_min") TIE_BREAK = TB_H_MIN; else if (s=="h_max") TIE_BREAK = TB_H_MAX;
        else if (s=="g_min") TIE_BREAK = TB_G_MIN; else if (s=="g_max") TIE_BREAK = TB_G_MAX;
    }

    int N; std::vector<std::vector<int>> grid0;
    if (!parse_input(argv[1], N, grid0)) return 1;
    int vmax=0; for (auto& r: grid0) for (int v: r) vmax = std::max(vmax, v);

    // build positions for values
    std::vector<PairPos> pos0(vmax+1);
    std::vector<int> cnt(vmax+1,0);
    for (int r=0;r<N;r++) for (int c=0;c<N;c++){
        int v=grid0[r][c]; if (cnt[v]==0){ pos0[v].r1=r; pos0[v].c1=c; } else { pos0[v].r2=r; pos0[v].c2=c; } cnt[v]++; }
    int hsum0 = compute_hsum_full(pos0);
    int hcnt0 = compute_hcnt_full(pos0);
    Zobrist zob(N, vmax);
    uint64_t hash0 = zob.hash_grid(grid0);

    // Time limit handling (env TIME_LIMIT_S or WASTAR_TIME_LIMIT_S)
    double TIME_LIMIT_S = -1.0;
    if (const char* ts = std::getenv("TIME_LIMIT_S")) { try { TIME_LIMIT_S = std::stod(ts); } catch (...) {} }
    if (TIME_LIMIT_S <= 0.0) {
        if (const char* ts = std::getenv("WASTAR_TIME_LIMIT_S")) { try { TIME_LIMIT_S = std::stod(ts); } catch (...) {} }
    }
    auto t_start = std::chrono::steady_clock::now();
    auto time_exceeded = [&]() -> bool {
        if (TIME_LIMIT_S <= 0.0) return false;
        using namespace std::chrono;
        double el = duration<double>(steady_clock::now() - t_start).count();
        return el >= TIME_LIMIT_S;
    };

    // OPEN and CLOSED
    std::priority_queue<Node, std::vector<Node>, PQComp> open;
    std::unordered_map<uint64_t, int> best_g; best_g.reserve(1<<20);
    std::unordered_map<uint64_t, ParentInfo> parent; parent.reserve(1<<20);

    open.push(Node{0, hsum0, hcnt0, 0 + W_WEIGHT*((1.0-ALPHA)*hsum0 + ALPHA*hcnt0), hash0});
    best_g[hash0] = 0;

    // to store grids and positions for nodes we expand (on demand)
    std::unordered_map<uint64_t, std::vector<std::vector<int>>> grid_of;
    std::unordered_map<uint64_t, std::vector<PairPos>> pos_of;
    grid_of[hash0]=grid0; pos_of[hash0]=pos0;

    // Defaults with size-based scaling; env overrides
    int MAX_SMALL_N = (N>=12?4:5);
    int CAND_CAP = (N>=12?64: std::max(32, N*N/2));
    int TOPK = (N>=12?32: CAND_CAP);
    if (const char* s = std::getenv("FAST_MAX_SMALL_N")) { try { MAX_SMALL_N = std::max(2, std::min(N, std::stoi(s))); } catch (...) {} }
    if (const char* s = std::getenv("FAST_CAND_CAP")) { try { CAND_CAP = std::max(8, std::stoi(s)); } catch (...) {} }
    if (const char* s = std::getenv("FAST_TOPK")) { try { TOPK = std::max(8, std::stoi(s)); } catch (...) {} }
    if (const char* s = std::getenv("K_TOP_MOVES")) { try { TOPK = std::max(8, std::stoi(s)); } catch (...) {} }

    int MAX_DEPTH = INT32_MAX;
    if (const char* s = std::getenv("MAX_DEPTH")) { try { MAX_DEPTH = std::max(1, std::stoi(s)); } catch (...) {} }

    // metrics
    long long nodes_expanded = 0;
    long long nodes_generated = 0;
    size_t open_max = 1;

    // track best-so-far by h_eff (then f, then g)
    uint64_t best_hash = hash0; int best_hsum = hsum0; int best_hcnt = hcnt0; int best_gval = 0; double best_f = 0 + W_WEIGHT*((1.0-ALPHA)*hsum0 + ALPHA*hcnt0);

    auto emit_from_hash = [&](uint64_t h, bool solved, bool partial, int hsum_val, int hcnt_val, double time_s){
        std::vector<Operation> ops;
        uint64_t curh = h;
        while (true){
            auto itp = parent.find(curh);
            if (itp==parent.end() || !itp->second.has_parent) break;
            ops.push_back(itp->second.op); curh = itp->second.parent_hash;
        }
        std::reverse(ops.begin(), ops.end());
        // peak RSS
        double peak_mb = 0.0;
#if defined(__APPLE__) || defined(__linux__)
        struct rusage u; if (getrusage(RUSAGE_SELF, &u) == 0){
#  if defined(__APPLE__)
            peak_mb = (double)u.ru_maxrss / (1024.0 * 1024.0);
#  else
            peak_mb = (double)u.ru_maxrss / 1024.0; // ru_maxrss in KiB on Linux
#  endif
        }
#endif
        emit_solution_json(ops, solved, partial, hsum_val, hcnt_val, (1.0-ALPHA)*hsum_val + ALPHA*hcnt_val,
                           time_s, nodes_expanded, nodes_generated, open_max, peak_mb, N);
    };

    while (!open.empty()){
        Node cur = open.top(); open.pop();
        auto itg = best_g.find(cur.hash);
        if (itg==best_g.end() || cur.g != itg->second) continue; // stale
        nodes_expanded++;

        // update best-so-far
        double cur_heff = (1.0-ALPHA)*cur.hsum + ALPHA*cur.hcnt;
        double best_heff = (1.0-ALPHA)*best_hsum + ALPHA*best_hcnt;
        if (cur_heff < best_heff || (cur_heff == best_heff && (cur.f < best_f || (cur.f == best_f && cur.g < best_gval)))){
            best_hash = cur.hash; best_hsum = cur.hsum; best_hcnt = cur.hcnt; best_gval = cur.g; best_f = cur.f;
        }

        // time limit check before expanding
        if (time_exceeded()){
            using namespace std::chrono; double tsec = duration<double>(steady_clock::now() - t_start).count();
            emit_from_hash(best_hash, false, true, best_hsum, best_hcnt, tsec);
            return 0;
        }

        // goal?
        if (cur.hsum == 0){
            using namespace std::chrono; double tsec = duration<double>(steady_clock::now() - t_start).count();
            emit_from_hash(cur.hash, true, false, 0, 0, tsec);
            return 0;
        }

        // materialize state
        auto& gcur = grid_of[cur.hash];
        auto& pcur = pos_of[cur.hash];

        // generate candidates: exhaustive for small boards, focused for larger
        std::vector<Operation> cand;
        if (N <= 10) {
            for (int n=2; n<=N; ++n){
                for (int y=0; y<=N-n; ++y){
                    for (int x=0; x<=N-n; ++x){ cand.push_back(Operation{x,y,n}); }
                }
            }
        } else {
            gen_candidates(N, gcur, pcur, MAX_SMALL_N, CAND_CAP, cand);
        }

        // pre-evaluate candidates (Δh and fnext), then push top-K
        struct CandEval { Operation op; int hsum_next; int hcnt_next; int gnext; double fnext; std::vector<PairPos> pnext; };
        std::vector<CandEval> evals; evals.reserve(cand.size());

        // reuse mark array to avoid repeated allocations
        std::vector<int> mark(vmax+1);
        for (const auto& op: cand){
            std::fill(mark.begin(), mark.end(), 0);
            std::vector<int> affected; affected.reserve(op.n*op.n);
            for (int i=0;i<op.n;i++) for (int j=0;j<op.n;j++){
                int v = gcur[op.y+i][op.x+j]; if (!mark[v]){ mark[v]=1; affected.push_back(v);} }
            int oldSum=0, newSum=0; int oldCnt=0, newCnt=0; std::vector<PairPos> pnext = pcur;
            for (int v: affected){
                oldSum += contrib_sum(pcur[v]);
                oldCnt += contrib_cnt(pcur[v]);
                PairPos np = pcur[v];
                if (in_sub(np.r1,np.c1,op.y,op.x,op.n)){ auto rc = rot_map(np.r1,np.c1,op.y,op.x,op.n); np.r1=rc.first; np.c1=rc.second; }
                if (in_sub(np.r2,np.c2,op.y,op.x,op.n)){ auto rc = rot_map(np.r2,np.c2,op.y,op.x,op.n); np.r2=rc.first; np.c2=rc.second; }
                newSum += contrib_sum(np);
                newCnt += contrib_cnt(np);
                pnext[v]=np;
            }
            int hsum_next = cur.hsum + (newSum - oldSum);
            int hcnt_next = cur.hcnt + (newCnt - oldCnt);
            int gnext = cur.g + 1;
            double heff_next = (1.0-ALPHA)*hsum_next + ALPHA*hcnt_next;
            double fnext = gnext + W_WEIGHT*heff_next;
            // depth cap
            if (gnext <= MAX_DEPTH){
                evals.push_back(CandEval{op, hsum_next, hcnt_next, gnext, fnext, std::move(pnext)});
            }
        }
        // sort by fnext then h_eff; cap only for large boards
        std::sort(evals.begin(), evals.end(), [](const CandEval& a, const CandEval& b){ if (a.fnext!=b.fnext) return a.fnext<b.fnext; double ha=(1.0-ALPHA)*a.hsum_next + ALPHA*a.hcnt_next; double hb=(1.0-ALPHA)*b.hsum_next + ALPHA*b.hcnt_next; return ha<hb; });
        if (N > 10 && (int)evals.size() > TOPK) evals.resize(TOPK);
        for (const auto& ce: evals){
            auto gnext_grid = gcur; uint64_t hsh = cur.hash; rotate_apply(gnext_grid, ce.op.x, ce.op.y, ce.op.n, zob, hsh);
            auto it = best_g.find(hsh); if (it != best_g.end() && it->second <= ce.gnext) continue;
            best_g[hsh] = ce.gnext; parent[hsh] = ParentInfo{cur.hash, ce.op, true};
            grid_of[hsh] = std::move(gnext_grid); pos_of[hsh] = ce.pnext; // pnext already a copy, move ok
            open.push(Node{ce.gnext, ce.hsum_next, ce.hcnt_next, ce.fnext, hsh});
            nodes_generated++;
            if (open.size() > open_max) open_max = open.size();
            // periodic time check inside expansion loop (cheap)
            if (time_exceeded()){
                // update best candidate with this neighbor if better
                double heff_next = (1.0-ALPHA)*ce.hsum_next + ALPHA*ce.hcnt_next;
                double best_heff2 = (1.0-ALPHA)*best_hsum + ALPHA*best_hcnt;
                if (heff_next < best_heff2 || (heff_next == best_heff2 && (ce.fnext < best_f || (ce.fnext == best_f && ce.gnext < best_gval)))){
                    best_hash = hsh; best_hsum = ce.hsum_next; best_hcnt = ce.hcnt_next; best_gval = ce.gnext; best_f = ce.fnext;
                }
                using namespace std::chrono; double tsec = duration<double>(steady_clock::now() - t_start).count();
                emit_from_hash(best_hash, false, true, best_hsum, best_hcnt, tsec);
                return 0;
            }
        }
    }
    // No exact solution found within search; emit best-so-far if any
    using namespace std::chrono; double tsec = duration<double>(steady_clock::now() - t_start).count();
    emit_from_hash(best_hash, false, true, best_hsum, best_hcnt, tsec);
    return 0;
}
