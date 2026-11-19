// SPDX-License-Identifier: GPL-3.0-or-later
// plegma_core: field-aligned quad remeshing of one part (Plegma add-on for Blender).
//
// Method after Instant Meshes (Jakob, Tarini, Panozzo, Sorkine-Hornung 2015):
// orientation field + position field on a dense triangle mesh, solved coarse-to-fine,
// then lattice-node extraction. Plegma additions: feature lines from part segments,
// sharp creases and open borders; gap filling; triangle pairing; final all-quad split.
//
// Usage: plegma_core job.txt
//   job.txt holds "key value" lines (see read_job). Exit code 0 = ok.

#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double T0 = now_s();
static int VERBOSE = 1;
static int POS_COARSE = 60, POS_FINE = 20;
static void stamp(const char *what) { if (VERBOSE > 1) std::printf("  %-18s %.2fs\n", what, now_s() - T0); }

// ------------------------------------------------------------------ math

struct V3 {
    double x = 0, y = 0, z = 0;
    V3() = default;
    V3(double a, double b, double c) : x(a), y(b), z(c) {}
    V3 operator+(const V3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(double s) const { return {x * s, y * s, z * s}; }
    V3 operator-() const { return {-x, -y, -z}; }
    V3 &operator+=(const V3 &o) { x += o.x; y += o.y; z += o.z; return *this; }
    V3 &operator-=(const V3 &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
static inline double dot(const V3 &a, const V3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(const V3 &a, const V3 &b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline double norm(const V3 &a) { return std::sqrt(dot(a, a)); }
static inline V3 normalized(const V3 &a) { double n = norm(a); return n > 1e-20 ? a * (1.0 / n) : V3(0, 0, 0); }
static inline V3 tangent(const V3 &v, const V3 &n) { return v - n * dot(v, n); }

using Tri = std::array<int, 3>;
using Poly = std::vector<int>;

static inline uint64_t ekey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

// ------------------------------------------------------------------ io

struct Job {
    std::string dense, ref, out;
    double target = 4000, sharp_angle = 40, merge_tol = 0.75, seam_eps = 0;
    int normal_smooth = 4, fill_all = 1, verbose = 1;
    double mem_limit_mb = 2048;
    std::vector<int> seam_axes;
};

static bool read_job(const char *path, Job &j) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string k; ss >> k;
        if (k == "dense") { std::getline(ss >> std::ws, j.dense); }
        else if (k == "ref") { std::getline(ss >> std::ws, j.ref); }
        else if (k == "out") { std::getline(ss >> std::ws, j.out); }
        else if (k == "target") ss >> j.target;
        else if (k == "sharp_angle") ss >> j.sharp_angle;
        else if (k == "merge_tol") ss >> j.merge_tol;
        else if (k == "normal_smooth") ss >> j.normal_smooth;
        else if (k == "fill_all") ss >> j.fill_all;
        else if (k == "seam_eps") ss >> j.seam_eps;
        else if (k == "mem_limit_mb") ss >> j.mem_limit_mb;
        else if (k == "verbose") ss >> j.verbose;
        else if (k == "pos_iters") ss >> POS_COARSE >> POS_FINE;
        else if (k == "seam_axes") { int a; while (ss >> a) j.seam_axes.push_back(a); }
    }
    return !j.dense.empty() && !j.ref.empty() && !j.out.empty();
}

// mesh file: "PLM1", int32 nV, int32 nT, float32 V[nV*3], int32 T[nT*3], int32 label[nT]
struct TriMesh {
    std::vector<V3> V;
    std::vector<Tri> T;
    std::vector<int> L;  // label per triangle (segment / part id)
};

static bool read_mesh(const std::string &path, TriMesh &m) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t nv = 0, nt = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "PLM1", 4) == 0 &&
              std::fread(&nv, 4, 1, f) == 1 && std::fread(&nt, 4, 1, f) == 1 && nv >= 0 && nt >= 0;
    if (ok) {
        std::vector<float> v(size_t(nv) * 3);
        std::vector<int32_t> t(size_t(nt) * 3), l(nt);
        ok = std::fread(v.data(), 4, v.size(), f) == v.size() && std::fread(t.data(), 4, t.size(), f) == t.size() &&
             std::fread(l.data(), 4, l.size(), f) == l.size();
        if (ok) {
            m.V.resize(nv); m.T.resize(nt); m.L.assign(l.begin(), l.end());
            for (int i = 0; i < nv; ++i) m.V[i] = V3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            for (int i = 0; i < nt; ++i) {
                m.T[i] = {t[3 * i], t[3 * i + 1], t[3 * i + 2]};
                for (int k = 0; k < 3; ++k) if (m.T[i][k] < 0 || m.T[i][k] >= nv) ok = false;
            }
        }
    }
    std::fclose(f);
    return ok;
}

// result file: "PLQ1", int32 nV, int32 nF, float32 V[nV*3], int32 size[nF], int32 idx[sum]
static bool write_polys(const std::string &path, const std::vector<V3> &V, const std::vector<Poly> &F) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    int32_t nv = int32_t(V.size()), nf = int32_t(F.size());
    std::fwrite("PLQ1", 1, 4, f);
    std::fwrite(&nv, 4, 1, f); std::fwrite(&nf, 4, 1, f);
    std::vector<float> v(V.size() * 3);
    for (size_t i = 0; i < V.size(); ++i) { v[3 * i] = float(V[i].x); v[3 * i + 1] = float(V[i].y); v[3 * i + 2] = float(V[i].z); }
    std::fwrite(v.data(), 4, v.size(), f);
    std::vector<int32_t> sz(F.size()), idx;
    for (size_t i = 0; i < F.size(); ++i) { sz[i] = int32_t(F[i].size()); for (int x : F[i]) idx.push_back(x); }
    std::fwrite(sz.data(), 4, sz.size(), f);
    std::fwrite(idx.data(), 4, idx.size(), f);
    bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

// ------------------------------------------------------------------ mesh helpers

static void weld(TriMesh &m) {
    if (m.V.empty()) return;
    V3 lo = m.V[0], hi = m.V[0];
    for (auto &p : m.V) { lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)}; hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)}; }
    double span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-12});
    double q = span * 1e-6;
    std::map<std::array<long long, 3>, int> id;
    std::vector<int> remap(m.V.size());
    std::vector<V3> nv;
    for (size_t i = 0; i < m.V.size(); ++i) {
        std::array<long long, 3> k = {llround(m.V[i].x / q), llround(m.V[i].y / q), llround(m.V[i].z / q)};
        auto it = id.find(k);
        if (it == id.end()) { id[k] = int(nv.size()); remap[i] = int(nv.size()); nv.push_back(m.V[i]); }
        else remap[i] = it->second;
    }
    std::vector<Tri> nt; std::vector<int> nl;
    for (size_t i = 0; i < m.T.size(); ++i) {
        Tri t = {remap[m.T[i][0]], remap[m.T[i][1]], remap[m.T[i][2]]};
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
        nt.push_back(t); nl.push_back(m.L[i]);
    }
    m.V.swap(nv); m.T.swap(nt); m.L.swap(nl);
}

static double tri_area(const TriMesh &m, int i) {
    const Tri &t = m.T[i];
    return 0.5 * norm(cross(m.V[t[1]] - m.V[t[0]], m.V[t[2]] - m.V[t[0]]));
}

struct EdgeInfo {
    std::vector<std::array<int, 2>> E;     // vertex pair
    std::vector<std::vector<int>> EF;      // faces per edge
    std::unordered_map<uint64_t, int> idx;
};

static EdgeInfo build_edges(const TriMesh &m) {
    EdgeInfo ei;
    ei.idx.reserve(m.T.size() * 2);
    for (int f = 0; f < int(m.T.size()); ++f)
        for (int k = 0; k < 3; ++k) {
            int a = m.T[f][k], b = m.T[f][(k + 1) % 3];
            uint64_t key = ekey(a, b);
            auto it = ei.idx.find(key);
            int e;
            if (it == ei.idx.end()) { e = int(ei.E.size()); ei.idx[key] = e; ei.E.push_back({std::min(a, b), std::max(a, b)}); ei.EF.emplace_back(); }
            else e = it->second;
            ei.EF[e].push_back(f);
        }
    return ei;
}

static std::vector<V3> face_normals(const TriMesh &m) {
    std::vector<V3> n(m.T.size());
    for (size_t i = 0; i < m.T.size(); ++i) {
        const Tri &t = m.T[i];
        n[i] = normalized(cross(m.V[t[1]] - m.V[t[0]], m.V[t[2]] - m.V[t[0]]));
    }
    return n;
}

static std::vector<V3> vertex_normals(const TriMesh &m) {
    std::vector<V3> n(m.V.size());
    for (auto &t : m.T) {
        V3 c = cross(m.V[t[1]] - m.V[t[0]], m.V[t[2]] - m.V[t[0]]);
        for (int k = 0; k < 3; ++k) n[t[k]] += c;
    }
    for (auto &x : n) x = normalized(x);
    return n;
}

// ------------------------------------------------------------------ closest point (AABB tree)

struct Box { V3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300}; };
static inline void grow(Box &b, const V3 &p) {
    b.lo = {std::min(b.lo.x, p.x), std::min(b.lo.y, p.y), std::min(b.lo.z, p.z)};
    b.hi = {std::max(b.hi.x, p.x), std::max(b.hi.y, p.y), std::max(b.hi.z, p.z)};
}
static inline double box_d2(const Box &b, const V3 &p) {
    double d = 0;
    for (int i = 0; i < 3; ++i) {
        double v = p[i], lo = b.lo[i], hi = b.hi[i];
        if (v < lo) d += (lo - v) * (lo - v); else if (v > hi) d += (v - hi) * (v - hi);
    }
    return d;
}

static V3 closest_on_tri(const V3 &p, const V3 &a, const V3 &b, const V3 &c) {
    V3 ab = b - a, ac = c - a, ap = p - a;
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    V3 bp = p - b; double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
    V3 cp = p - c; double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    double den = 1.0 / (va + vb + vc);
    return a + ab * (vb * den) + ac * (vc * den);
}

struct BVH {
    const TriMesh *m = nullptr;
    struct Node { Box b; int left = -1, right = -1, start = 0, count = 0; };
    std::vector<Node> nodes;
    std::vector<int> order;

    void build(const TriMesh &mesh) {
        m = &mesh;
        order.resize(mesh.T.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<V3> cen(mesh.T.size());
        for (size_t i = 0; i < mesh.T.size(); ++i) { auto &t = mesh.T[i]; cen[i] = (mesh.V[t[0]] + mesh.V[t[1]] + mesh.V[t[2]]) * (1.0 / 3); }
        nodes.clear(); nodes.reserve(2 * mesh.T.size() / 4 + 2);
        std::function<int(int, int)> rec = [&](int s, int n) -> int {
            Node nd; nd.start = s; nd.count = n;
            Box cb;
            for (int i = s; i < s + n; ++i) { auto &t = mesh.T[order[i]]; for (int k = 0; k < 3; ++k) grow(nd.b, mesh.V[t[k]]); grow(cb, cen[order[i]]); }
            int id = int(nodes.size()); nodes.push_back(nd);
            if (n > 4) {
                V3 ext = cb.hi - cb.lo;
                int ax = ext.x > ext.y ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2);
                int mid = s + n / 2;
                std::nth_element(order.begin() + s, order.begin() + mid, order.begin() + s + n,
                                 [&](int a, int b) { return cen[a][ax] < cen[b][ax]; });
                int l = rec(s, mid - s), r = rec(mid, s + n - mid);
                nodes[id].left = l; nodes[id].right = r; nodes[id].count = 0;
            }
            return id;
        };
        if (!mesh.T.empty()) rec(0, int(mesh.T.size()));
    }

    V3 nearest(const V3 &p, int *face = nullptr) const {
        double best = 1e300; V3 bp = p; int bf = -1;
        if (nodes.empty()) return p;
        std::vector<int> stack = {0};
        while (!stack.empty()) {
            int id = stack.back(); stack.pop_back();
            const Node &nd = nodes[id];
            if (box_d2(nd.b, p) >= best) continue;
            if (nd.count > 0) {
                for (int i = nd.start; i < nd.start + nd.count; ++i) {
                    auto &t = m->T[order[i]];
                    V3 q = closest_on_tri(p, m->V[t[0]], m->V[t[1]], m->V[t[2]]);
                    double d = dot(q - p, q - p);
                    if (d < best) { best = d; bp = q; bf = order[i]; }
                }
            } else {
                double dl = box_d2(nodes[nd.left].b, p), dr = box_d2(nodes[nd.right].b, p);
                if (dl < dr) { stack.push_back(nd.right); stack.push_back(nd.left); }
                else { stack.push_back(nd.left); stack.push_back(nd.right); }
            }
        }
        if (face) *face = bf;
        return bp;
    }
};

// ------------------------------------------------------------------ dense base

// split every edge longer than max_edge (red-green subdivision), labels inherited
static void split_long_edges(TriMesh &m, double max_edge) {
    for (int pass = 0; pass < 8; ++pass) {
        std::unordered_map<uint64_t, int> mid;
        for (auto &t : m.T)
            for (int k = 0; k < 3; ++k) {
                int a = t[k], b = t[(k + 1) % 3];
                if (norm(m.V[a] - m.V[b]) > max_edge) {
                    uint64_t key = ekey(a, b);
                    if (!mid.count(key)) { mid[key] = int(m.V.size()); m.V.push_back((m.V[a] + m.V[b]) * 0.5); }
                }
            }
        if (mid.empty()) return;
        std::vector<Tri> nt; std::vector<int> nl;
        nt.reserve(m.T.size() * 2); nl.reserve(m.T.size() * 2);
        for (size_t f = 0; f < m.T.size(); ++f) {
            const Tri t = m.T[f];
            int lab = m.L[f];
            int ms[3];
            int cnt = 0;
            for (int k = 0; k < 3; ++k) {
                auto it = mid.find(ekey(t[k], t[(k + 1) % 3]));
                ms[k] = it == mid.end() ? -1 : it->second;
                cnt += ms[k] >= 0;
            }
            auto add = [&](int a, int b, int c) { nt.push_back({a, b, c}); nl.push_back(lab); };
            if (cnt == 0) { add(t[0], t[1], t[2]); continue; }
            if (cnt == 3) {
                add(t[0], ms[0], ms[2]); add(ms[0], t[1], ms[1]); add(ms[2], ms[1], t[2]); add(ms[0], ms[1], ms[2]);
                continue;
            }
            // rotate so that edge 0 is split
            int r = 0;
            while (ms[r] < 0) ++r;
            int a = t[r], b = t[(r + 1) % 3], c = t[(r + 2) % 3];
            int mab = ms[r], mbc = ms[(r + 1) % 3], mca = ms[(r + 2) % 3];
            if (cnt == 1) { add(a, mab, c); add(mab, b, c); }
            else if (mbc >= 0) { add(a, mab, c); add(mab, b, mbc); add(mab, mbc, c); }
            else { add(a, mab, mca); add(mab, b, c); add(mab, c, mca); }
        }
        m.T.swap(nt); m.L.swap(nl);
    }
}

// ------------------------------------------------------------------ features

struct Features {
    std::vector<char> is_feat_edge;  // per edge
    std::vector<char> feat, corner;  // per vertex
    std::vector<V3> fdir;            // per vertex
};

static std::vector<char> prune_lines(const EdgeInfo &ei, const std::vector<char> &fe, const std::vector<V3> &V, double min_len) {
    int nE = int(ei.E.size());
    std::vector<std::vector<int>> vert_edges(V.size());
    for (int e = 0; e < nE; ++e) if (fe[e]) { vert_edges[ei.E[e][0]].push_back(e); vert_edges[ei.E[e][1]].push_back(e); }
    std::vector<char> seen(nE, 0), keep(nE, 0);
    for (int e = 0; e < nE; ++e) {
        if (!fe[e] || seen[e]) continue;
        std::vector<int> comp, st = {e};
        seen[e] = 1;
        double len = 0; bool border = false;
        while (!st.empty()) {
            int x = st.back(); st.pop_back(); comp.push_back(x);
            len += norm(V[ei.E[x][0]] - V[ei.E[x][1]]);
            if (ei.EF[x].size() != 2) border = true;
            for (int vv : ei.E[x]) for (int y : vert_edges[vv]) if (!seen[y]) { seen[y] = 1; st.push_back(y); }
        }
        if (len >= min_len || border) for (int x : comp) keep[x] = 1;
    }
    return keep;
}

static Features find_features(const TriMesh &m, const EdgeInfo &ei, const std::vector<V3> &fn, double sharp_deg, double h) {
    int nE = int(ei.E.size());
    std::vector<char> fe(nE, 0);
    double lim = std::cos(sharp_deg * M_PI / 180.0);
    for (int e = 0; e < nE; ++e) {
        auto &ef = ei.EF[e];
        if (ef.size() != 2) { fe[e] = 1; continue; }
        if (m.L[ef[0]] != m.L[ef[1]]) { fe[e] = 1; continue; }
        if (sharp_deg < 180 && dot(fn[ef[0]], fn[ef[1]]) < lim) fe[e] = 1;
    }
    Features F;
    F.is_feat_edge = prune_lines(ei, fe, m.V, 2.0 * h);
    int nV = int(m.V.size());
    std::vector<std::vector<int>> ve(nV);
    for (int e = 0; e < nE; ++e) if (F.is_feat_edge[e]) { ve[ei.E[e][0]].push_back(e); ve[ei.E[e][1]].push_back(e); }
    F.feat.assign(nV, 0); F.corner.assign(nV, 0); F.fdir.assign(nV, V3());
    for (int v = 0; v < nV; ++v) {
        size_t c = ve[v].size();
        if (c == 0) continue;
        auto other = [&](int e) { return ei.E[e][0] == v ? ei.E[e][1] : ei.E[e][0]; };
        if (c == 1) { F.feat[v] = 1; F.fdir[v] = normalized(m.V[other(ve[v][0])] - m.V[v]); }
        else if (c == 2) { F.feat[v] = 1; F.fdir[v] = normalized(m.V[other(ve[v][0])] - m.V[v]) - normalized(m.V[other(ve[v][1])] - m.V[v]); }
        else F.corner[v] = 1;
    }
    return F;
}

static void relax_project(TriMesh &m, const EdgeInfo &ei, const Features &F, const BVH &bvh, double strength) {
    int nV = int(m.V.size());
    std::vector<V3> vn = vertex_normals(m), acc(nV);
    std::vector<int> cnt(nV, 0);
    for (int e = 0; e < int(ei.E.size()); ++e) {
        int a = ei.E[e][0], b = ei.E[e][1];
        bool fe = F.is_feat_edge.empty() ? false : F.is_feat_edge[e];
        if (!F.feat[a] || fe) { acc[a] += m.V[b]; cnt[a]++; }
        if (!F.feat[b] || fe) { acc[b] += m.V[a]; cnt[b]++; }
    }
    for (int v = 0; v < nV; ++v) {
        if (F.corner[v] || cnt[v] == 0) continue;
        V3 d = acc[v] * (1.0 / cnt[v]) - m.V[v];
        if (!F.feat[v]) d = tangent(d, vn[v]);
        m.V[v] = bvh.nearest(m.V[v] + d * strength);
    }
}

// ------------------------------------------------------------------ fields

static inline V3 rosy_match(const V3 &qi, const V3 &qj, const V3 &nj) {
    V3 tj = cross(nj, qj);
    V3 c[4] = {qj, tj, -qj, -tj};
    int best = 0; double bd = -1e300;
    for (int k = 0; k < 4; ++k) { double d = dot(qi, c[k]); if (d > bd) { bd = d; best = k; } }
    return c[best];
}

static inline V3 middle_point(const V3 &p0, const V3 &n0, const V3 &p1, const V3 &n1) {
    double n0p0 = dot(n0, p0), n0p1 = dot(n0, p1), n1p0 = dot(n1, p0), n1p1 = dot(n1, p1), n0n1 = dot(n0, n1);
    double den = 1.0 / (1.0 - n0n1 * n0n1 + 1e-4);
    double l0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1)) * den;
    double l1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0)) * den;
    return (p0 + p1) * 0.5 - (n0 * l0 + n1 * l1) * 0.25;
}

static inline V3 lattice_floor(const V3 &o, const V3 &q, const V3 &t, const V3 &p, double h) {
    V3 d = p - o;
    return o + q * (std::floor(dot(q, d) / h) * h) + t * (std::floor(dot(t, d) / h) * h);
}
static inline V3 lattice_round(const V3 &o, const V3 &q, const V3 &t, const V3 &p, double h) {
    V3 d = p - o;
    return o + q * (std::floor(dot(q, d) / h + 0.5) * h) + t * (std::floor(dot(t, d) / h + 0.5) * h);
}

static inline void compat_pos(const V3 &v0, const V3 &n0, const V3 &q0, const V3 &o0,
                              const V3 &v1, const V3 &n1, const V3 &q1, const V3 &o1, double h, V3 &ra, V3 &rb) {
    V3 t0 = cross(n0, q0), t1 = cross(n1, q1);
    V3 mid = middle_point(v0, n0, v1, n1);
    V3 a = lattice_floor(o0, q0, t0, mid, h), b = lattice_floor(o1, q1, t1, mid, h);
    double best = 1e300;
    for (int i = 0; i < 4; ++i) {
        V3 ai = a + (q0 * double(i & 1) + t0 * double((i & 2) >> 1)) * h;
        for (int j = 0; j < 4; ++j) {
            V3 bj = b + (q1 * double(j & 1) + t1 * double((j & 2) >> 1)) * h;
            V3 d = ai - bj;
            double c = dot(d, d);
            if (c < best) { best = c; ra = ai; rb = bj; }
        }
    }
}

struct Level {
    std::vector<V3> V, N;
    std::vector<std::array<int, 2>> E;
    std::vector<int> parent;  // fine vertex -> coarse vertex (of the next level)
};

static std::vector<Level> build_hierarchy(const std::vector<V3> &V, const std::vector<V3> &N, const std::vector<std::array<int, 2>> &E, int min_size = 300) {
    std::vector<Level> lv(1);
    lv[0].V = V; lv[0].N = N; lv[0].E = E;
    while (int(lv.back().V.size()) > min_size) {
        Level &f = lv.back();
        V3 lo = f.V[0], hi = f.V[0];
        for (auto &p : f.V) { lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)}; hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)}; }
        double span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-12});
        double target = std::max<double>(f.V.size() / 4, min_size / 2);
        double cell = span / std::max(std::sqrt(target), 2.0);
        std::map<std::array<long long, 4>, int> id;
        std::vector<int> par(f.V.size());
        for (size_t i = 0; i < f.V.size(); ++i) {
            const V3 &n = f.N[i];
            int ax = std::fabs(n.x) > std::fabs(n.y) ? (std::fabs(n.x) > std::fabs(n.z) ? 0 : 2) : (std::fabs(n.y) > std::fabs(n.z) ? 1 : 2);
            long long nb = ax * 2 + (n[ax] > 0);
            std::array<long long, 4> k = {(long long)std::floor(f.V[i].x / cell), (long long)std::floor(f.V[i].y / cell), (long long)std::floor(f.V[i].z / cell), nb};
            auto it = id.find(k);
            if (it == id.end()) { int c = int(id.size()); id[k] = c; par[i] = c; } else par[i] = it->second;
        }
        int nC = int(id.size());
        if (nC >= int(f.V.size() * 0.9)) break;
        Level c;
        c.V.assign(nC, V3()); c.N.assign(nC, V3());
        std::vector<int> cnt(nC, 0);
        for (size_t i = 0; i < f.V.size(); ++i) { c.V[par[i]] += f.V[i]; c.N[par[i]] += f.N[i]; cnt[par[i]]++; }
        for (int i = 0; i < nC; ++i) { c.V[i] = c.V[i] * (1.0 / cnt[i]); c.N[i] = normalized(c.N[i]); }
        std::unordered_set<uint64_t> es;
        for (auto &e : f.E) { int a = par[e[0]], b = par[e[1]]; if (a != b && es.insert(ekey(a, b)).second) c.E.push_back({std::min(a, b), std::max(a, b)}); }
        f.parent = par;
        lv.push_back(std::move(c));
    }
    return lv;
}

static void smooth_orient(std::vector<V3> &q, const std::vector<V3> &N, const std::vector<std::array<int, 2>> &E,
                          const std::vector<V3> &ld, const std::vector<char> &lk, int iters) {
    int n = int(q.size());
    std::vector<V3> acc(n);
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) acc[i] = q[i];
        for (auto &e : E) {
            int a = e[0], b = e[1];
            acc[a] += rosy_match(q[a], q[b], N[b]);
            acc[b] += rosy_match(q[b], q[a], N[a]);
        }
        for (int i = 0; i < n; ++i) q[i] = lk[i] ? ld[i] : normalized(tangent(acc[i], N[i]));
    }
}

static std::vector<V3> orientation_hier(std::vector<Level> &lv, const std::vector<V3> &ld0, const std::vector<char> &lk0) {
    std::vector<std::vector<V3>> LD = {ld0};
    std::vector<std::vector<char>> LK = {lk0};
    for (size_t l = 1; l < lv.size(); ++l) {
        auto &par = lv[l - 1].parent;
        std::vector<V3> cd(lv[l].V.size()); std::vector<char> ck(lv[l].V.size(), 0);
        for (size_t i = 0; i < par.size(); ++i) if (LK.back()[i]) { cd[par[i]] = LD.back()[i]; ck[par[i]] = 1; }
        for (size_t i = 0; i < cd.size(); ++i) cd[i] = normalized(tangent(cd[i], lv[l].N[i]));
        LD.push_back(cd); LK.push_back(ck);
    }
    Level &top = lv.back();
    std::mt19937 rng(1);
    std::normal_distribution<double> nd;
    std::vector<V3> q(top.V.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = normalized(tangent(V3(nd(rng), nd(rng), nd(rng)), top.N[i]));
    smooth_orient(q, top.N, top.E, LD.back(), LK.back(), 400);
    for (int l = int(lv.size()) - 2; l >= 0; --l) {
        auto &par = lv[l].parent;
        std::vector<V3> qf(lv[l].V.size());
        for (size_t i = 0; i < qf.size(); ++i) qf[i] = normalized(tangent(q[par[i]], lv[l].N[i]));
        q.swap(qf);
        smooth_orient(q, lv[l].N, lv[l].E, LD[l], LK[l], 80);
    }
    return q;
}

static void position_field(const Level &L, const std::vector<V3> &Q, double h, const std::vector<V3> &fdir,
                           const std::vector<char> &feat, const std::vector<char> &corner, std::vector<V3> &O, int iters) {
    int n = int(L.V.size());
    std::vector<V3> T(n), perp(n), acc(n);
    std::vector<int> cnt(n, 0);
    for (int i = 0; i < n; ++i) {
        T[i] = cross(L.N[i], Q[i]);
        perp[i] = std::fabs(dot(Q[i], fdir[i])) >= std::fabs(dot(T[i], fdir[i])) ? T[i] : Q[i];
    }
    for (auto &e : L.E) { cnt[e[0]]++; cnt[e[1]]++; }
    for (int it = 0; it < iters; ++it) {
        std::fill(acc.begin(), acc.end(), V3());
        for (auto &e : L.E) {
            int a = e[0], b = e[1];
            V3 oa, ob;
            compat_pos(L.V[a], L.N[a], Q[a], O[a], L.V[b], L.N[b], Q[b], O[b], h, oa, ob);
            acc[a] += ob - oa;
            acc[b] += oa - ob;
        }
        for (int i = 0; i < n; ++i) {
            V3 o = O[i] + acc[i] * (0.8 / std::max(cnt[i], 1));
            o -= L.N[i] * dot(o - L.V[i], L.N[i]);
            o = lattice_round(o, Q[i], T[i], L.V[i], h);
            if (feat[i]) o += perp[i] * dot(L.V[i] - o, perp[i]);
            if (corner[i]) o = L.V[i];
            O[i] = o;
        }
    }
}

static std::vector<V3> position_hier(std::vector<Level> &lv, const std::vector<V3> &Q0, double h, const std::vector<V3> &fdir0,
                                     const std::vector<char> &feat0, const std::vector<char> &corner0) {
    std::vector<std::vector<V3>> QS = {Q0}, FD = {fdir0};
    std::vector<std::vector<char>> FE = {feat0}, CO = {corner0};
    for (size_t l = 1; l < lv.size(); ++l) {
        auto &par = lv[l - 1].parent;
        size_t n = lv[l].V.size();
        std::vector<V3> qc(n), fd(n); std::vector<char> fe(n, 0), co(n, 0);
        for (size_t i = 0; i < par.size(); ++i) {
            qc[par[i]] = QS.back()[i];
            if (FE.back()[i]) { fd[par[i]] = FD.back()[i]; fe[par[i]] = 1; }
            if (CO.back()[i]) co[par[i]] = 1;
        }
        for (size_t i = 0; i < n; ++i) { qc[i] = normalized(tangent(qc[i], lv[l].N[i])); fd[i] = normalized(fd[i]); if (co[i]) fe[i] = 0; }
        QS.push_back(qc); FD.push_back(fd); FE.push_back(fe); CO.push_back(co);
    }
    std::vector<V3> O;
    for (int l = int(lv.size()) - 1; l >= 0; --l) {
        int its;
        if (O.empty()) { O = lv[l].V; its = POS_COARSE; }
        else {
            auto &par = lv[l].parent;
            std::vector<V3> o(lv[l].V.size());
            for (size_t i = 0; i < o.size(); ++i) { V3 c = O[par[i]]; o[i] = c - lv[l].N[i] * dot(c - lv[l].V[i], lv[l].N[i]); }
            O.swap(o); its = POS_FINE;
        }
        position_field(lv[l], QS[l], h, FD[l], FE[l], CO[l], O, its);
    }
    return O;
}

// ------------------------------------------------------------------ extraction

static void extract(const std::vector<V3> &V, const std::vector<V3> &N, const std::vector<V3> &Q, const std::vector<V3> &O,
                    const std::vector<std::array<int, 2>> &E, double h, double merge_tol,
                    std::vector<V3> &P, std::vector<Poly> &faces) {
    int nV = int(V.size()), nE = int(E.size());
    std::vector<int> man(nE);
    std::vector<double> gap(nE);
    for (int k = 0; k < nE; ++k) {
        int a = E[k][0], b = E[k][1];
        V3 oa, ob;
        compat_pos(V[a], N[a], Q[a], O[a], V[b], N[b], Q[b], O[b], h, oa, ob);
        V3 ta = cross(N[a], Q[a]);
        double ka0 = std::round(dot(Q[a], oa - O[a]) / h), ka1 = std::round(dot(ta, oa - O[a]) / h);
        V3 qb = rosy_match(Q[a], Q[b], N[b]), tb = cross(N[b], qb);
        double kb0 = std::round(dot(qb, ob - O[b]) / h), kb1 = std::round(dot(tb, ob - O[b]) / h);
        man[k] = int(std::fabs(ka0 - kb0) + std::fabs(ka1 - kb1));
        gap[k] = norm(O[a] - O[b]);
    }
    // 1. merge vertices on the same lattice node, most confident first
    std::vector<int> same;
    for (int k = 0; k < nE; ++k) if (man[k] == 0) same.push_back(k);
    std::sort(same.begin(), same.end(), [&](int x, int y) { return gap[x] < gap[y]; });
    std::vector<int> parent(nV);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<V3> csum = O;
    std::vector<double> ccnt(nV, 1.0);
    std::function<int(int)> find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    double lim2 = merge_tol * h * merge_tol * h;
    for (int k : same) {
        int ri = find(E[k][0]), rj = find(E[k][1]);
        if (ri == rj) continue;
        V3 d = csum[ri] * (1.0 / ccnt[ri]) - csum[rj] * (1.0 / ccnt[rj]);
        if (dot(d, d) > lim2) continue;
        if (ccnt[ri] < ccnt[rj]) std::swap(ri, rj);
        parent[rj] = ri; csum[ri] += csum[rj]; ccnt[ri] += ccnt[rj];
    }
    std::vector<int> cl(nV, -1);
    std::unordered_map<int, int> rootid;
    for (int i = 0; i < nV; ++i) {
        int r = find(i);
        auto it = rootid.find(r);
        if (it == rootid.end()) { int c = int(rootid.size()); rootid[r] = c; cl[i] = c; } else cl[i] = it->second;
    }
    int nC = int(rootid.size());
    P.assign(nC, V3());
    std::vector<V3> NC(nC);
    for (auto &kv : rootid) P[kv.second] = csum[kv.first] * (1.0 / ccnt[kv.first]);
    for (int i = 0; i < nV; ++i) NC[cl[i]] += N[i];
    for (auto &n : NC) n = normalized(n);
    // 2. unit steps -> graph edges
    std::vector<std::vector<int>> nb(nC);
    std::unordered_set<uint64_t> es;
    for (int k = 0; k < nE; ++k) {
        if (man[k] != 1) continue;
        int a = cl[E[k][0]], b = cl[E[k][1]];
        if (a == b || !es.insert(ekey(a, b)).second) continue;
        nb[a].push_back(b); nb[b].push_back(a);
    }
    // 3. ccw order around nodes, trace faces
    std::vector<std::unordered_map<int, int>> pos(nC);
    for (int i = 0; i < nC; ++i) {
        auto &L = nb[i];
        if (L.empty()) continue;
        V3 n = NC[i];
        V3 ref = normalized(tangent(P[L[0]] - P[i], n));
        V3 side = cross(n, ref);
        std::vector<std::pair<double, int>> a;
        for (int j : L) { V3 d = P[j] - P[i]; a.push_back({std::atan2(dot(d, side), dot(d, ref)), j}); }
        std::sort(a.begin(), a.end());
        for (size_t k = 0; k < a.size(); ++k) L[k] = a[k].second;
        for (size_t k = 0; k < L.size(); ++k) pos[i][L[k]] = int(k);
    }
    std::unordered_set<uint64_t> used;
    auto dkey = [](int u, int v) { return (uint64_t(uint32_t(u)) << 32) | uint32_t(v); };
    for (int i = 0; i < nC; ++i)
        for (int j : nb[i]) {
            if (used.count(dkey(i, j))) continue;
            Poly loop = {i};
            int u = i, v = j;
            bool ok = false;
            for (int step = 0; step < 8; ++step) {
                used.insert(dkey(u, v));
                loop.push_back(v);
                auto &ov = nb[v];
                int w = ov[(pos[v][u] - 1 + int(ov.size())) % int(ov.size())];
                u = v; v = w;
                if (v == i) { used.insert(dkey(u, v)); ok = true; break; }
                if (std::find(loop.begin(), loop.end(), v) != loop.end()) break;
            }
            if (ok && loop.size() >= 3 && loop.size() <= 6) faces.push_back(loop);
        }
}

// ------------------------------------------------------------------ polygon mesh post-processing

struct PolyMesh {
    std::vector<V3> V;
    std::vector<Poly> F;
};

static V3 poly_normal(const PolyMesh &m, const Poly &f) {
    V3 n;
    for (size_t i = 0; i < f.size(); ++i) n += cross(m.V[f[i]], m.V[f[(i + 1) % f.size()]]);
    return normalized(n);
}

// directed half-edge map; boundary loops follow the reverse of their face's direction
static std::vector<std::vector<int>> boundary_loops(const PolyMesh &m) {
    std::unordered_map<uint64_t, int> count;
    for (auto &f : m.F) for (size_t i = 0; i < f.size(); ++i) count[ekey(f[i], f[(i + 1) % f.size()])]++;
    std::unordered_multimap<int, int> next;
    for (auto &f : m.F)
        for (size_t i = 0; i < f.size(); ++i) {
            int a = f[i], b = f[(i + 1) % f.size()];
            if (count[ekey(a, b)] == 1) next.insert({b, a});
        }
    std::vector<std::vector<int>> loops;
    std::unordered_set<uint64_t> done;
    for (auto &kv : next) {
        if (done.count(ekey(kv.first, kv.second) ^ (uint64_t(1) << 63))) continue;
        std::vector<int> loop = {kv.first};
        int cur = kv.second;
        done.insert(ekey(kv.first, kv.second) ^ (uint64_t(1) << 63));
        int guard = 0;
        while (cur != loop[0] && guard++ < 100000) {
            loop.push_back(cur);
            auto range = next.equal_range(cur);
            int nx = -1;
            for (auto it = range.first; it != range.second; ++it)
                if (!done.count(ekey(cur, it->second) ^ (uint64_t(1) << 63))) { nx = it->second; break; }
            if (nx < 0) break;
            done.insert(ekey(cur, nx) ^ (uint64_t(1) << 63));
            cur = nx;
        }
        if (cur == loop[0] && loop.size() >= 3) loops.push_back(loop);
    }
    return loops;
}

// ear clipping of a 3D loop projected on its best plane; falls back to a fan
static void fill_loop(PolyMesh &m, const std::vector<int> &loop) {
    size_t n = loop.size();
    V3 c, nrm;
    for (int v : loop) c += m.V[v];
    c = c * (1.0 / n);
    for (size_t i = 0; i < n; ++i) nrm += cross(m.V[loop[i]] - c, m.V[loop[(i + 1) % n]] - c);
    nrm = normalized(nrm);
    if (norm(nrm) < 0.5) {  // degenerate: fan around the centre
        int ci = int(m.V.size()); m.V.push_back(c);
        for (size_t i = 0; i < n; ++i) m.F.push_back({loop[i], loop[(i + 1) % n], ci});
        return;
    }
    V3 ax = normalized(tangent(m.V[loop[0]] - c, nrm)), ay = cross(nrm, ax);
    std::vector<std::pair<double, double>> p(n);
    for (size_t i = 0; i < n; ++i) { V3 d = m.V[loop[i]] - c; p[i] = {dot(d, ax), dot(d, ay)}; }
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    auto area2 = [&](int a, int b, int cc) {
        return (p[b].first - p[a].first) * (p[cc].second - p[a].second) - (p[b].second - p[a].second) * (p[cc].first - p[a].first);
    };
    auto inside = [&](int a, int b, int cc, int q) {
        return area2(a, b, q) >= 0 && area2(b, cc, q) >= 0 && area2(cc, a, q) >= 0;
    };
    std::vector<Tri> tris;
    int guard = 0;
    while (idx.size() > 3 && guard++ < int(n * n + 10)) {
        bool cut = false;
        for (size_t i = 0; i < idx.size(); ++i) {
            int a = idx[(i + idx.size() - 1) % idx.size()], b = idx[i], cc = idx[(i + 1) % idx.size()];
            if (area2(a, b, cc) <= 0) continue;
            bool ear = true;
            for (int q : idx) if (q != a && q != b && q != cc && inside(a, b, cc, q)) { ear = false; break; }
            if (!ear) continue;
            tris.push_back({a, b, cc});
            idx.erase(idx.begin() + i);
            cut = true;
            break;
        }
        if (!cut) break;
    }
    if (idx.size() == 3) tris.push_back({idx[0], idx[1], idx[2]});
    if (idx.size() > 3) {
        int ci = int(m.V.size()); m.V.push_back(c);
        for (size_t i = 0; i < n; ++i) m.F.push_back({loop[i], loop[(i + 1) % n], ci});
        return;
    }
    for (auto &t : tris) m.F.push_back({loop[t[0]], loop[t[1]], loop[t[2]]});
}

// pair neighbouring triangles into quads (best-shaped first)
static void join_triangles(PolyMesh &m, double max_dev_deg = 60.0, double max_fold_deg = 50.0) {
    std::unordered_map<uint64_t, std::vector<int>> ef;
    for (int f = 0; f < int(m.F.size()); ++f) {
        if (m.F[f].size() != 3) continue;
        for (int k = 0; k < 3; ++k) ef[ekey(m.F[f][k], m.F[f][(k + 1) % 3])].push_back(f);
    }
    struct Cand { double score; int f0, f1; Poly quad; };
    std::vector<Cand> cands;
    double cf = std::cos(max_fold_deg * M_PI / 180.0);
    for (auto &kv : ef) {
        if (kv.second.size() != 2) continue;
        int f0 = kv.second[0], f1 = kv.second[1];
        int a = int(kv.first >> 32), b = int(kv.first & 0xffffffffu);
        auto other = [&](int f) { for (int v : m.F[f]) if (v != a && v != b) return v; return -1; };
        int c = other(f0), d = other(f1);
        if (c < 0 || d < 0 || c == d) continue;
        if (dot(poly_normal(m, m.F[f0]), poly_normal(m, m.F[f1])) < cf) continue;
        // quad in f0's winding: f0 = (.., a, b, ..) -> c, a, d, b or c, b, d, a
        const Poly &t = m.F[f0];
        int ia = int(std::find(t.begin(), t.end(), a) - t.begin());
        Poly q = t[(ia + 1) % 3] == b ? Poly{c, a, d, b} : Poly{c, b, d, a};
        double dev = 0;
        for (int i = 0; i < 4; ++i) {
            V3 x = m.V[q[(i + 3) % 4]] - m.V[q[i]], y = m.V[q[(i + 1) % 4]] - m.V[q[i]];
            double nx = norm(x), ny = norm(y);
            if (nx < 1e-20 || ny < 1e-20) { dev = 180; break; }
            double ang = std::acos(std::max(-1.0, std::min(1.0, dot(x, y) / (nx * ny)))) * 180.0 / M_PI;
            dev = std::max(dev, std::fabs(ang - 90.0));
        }
        if (dev <= max_dev_deg) cands.push_back({dev, f0, f1, q});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand &x, const Cand &y) { return x.score < y.score; });
    std::vector<char> used(m.F.size(), 0), drop(m.F.size(), 0);
    std::vector<Poly> add;
    for (auto &c : cands) {
        if (used[c.f0] || used[c.f1]) continue;
        used[c.f0] = used[c.f1] = 1; drop[c.f0] = drop[c.f1] = 1;
        add.push_back(c.quad);
    }
    std::vector<Poly> nf;
    for (size_t f = 0; f < m.F.size(); ++f) if (!drop[f]) nf.push_back(m.F[f]);
    for (auto &q : add) nf.push_back(q);
    m.F.swap(nf);
}

// every n-gon -> n quads (corner, edge midpoint, centre, edge midpoint)
static PolyMesh split_to_quads(const PolyMesh &m) {
    PolyMesh o;
    o.V = m.V;
    std::unordered_map<uint64_t, int> mid;
    auto midpoint = [&](int a, int b) {
        uint64_t k = ekey(a, b);
        auto it = mid.find(k);
        if (it != mid.end()) return it->second;
        int id = int(o.V.size()); o.V.push_back((m.V[a] + m.V[b]) * 0.5); mid[k] = id;
        return id;
    };
    for (auto &f : m.F) {
        V3 c;
        for (int v : f) c += m.V[v];
        int ci = int(o.V.size()); o.V.push_back(c * (1.0 / f.size()));
        size_t n = f.size();
        for (size_t i = 0; i < n; ++i) {
            int prev = f[(i + n - 1) % n], cur = f[i], nxt = f[(i + 1) % n];
            o.F.push_back({cur, midpoint(cur, nxt), ci, midpoint(prev, cur)});
        }
    }
    return o;
}

static void relax_poly(PolyMesh &m, const BVH &bvh, int iters, double strength) {
    int nV = int(m.V.size());
    std::unordered_map<uint64_t, int> ecount;
    for (auto &f : m.F) for (size_t i = 0; i < f.size(); ++i) ecount[ekey(f[i], f[(i + 1) % f.size()])]++;
    std::vector<std::vector<int>> nbr(nV), bnbr(nV);
    for (auto &kv : ecount) {
        int a = int(kv.first >> 32), b = int(kv.first & 0xffffffffu);
        nbr[a].push_back(b); nbr[b].push_back(a);
        if (kv.second == 1) { bnbr[a].push_back(b); bnbr[b].push_back(a); }
    }
    for (int it = 0; it < iters; ++it) {
        std::vector<V3> vn(nV);
        for (auto &f : m.F) { V3 n = poly_normal(m, f); for (int v : f) vn[v] += n; }
        std::vector<V3> nv = m.V;
        for (int v = 0; v < nV; ++v) {
            auto &L = bnbr[v].empty() ? nbr[v] : bnbr[v];
            if (L.empty() || (!bnbr[v].empty() && bnbr[v].size() != 2)) continue;
            V3 a;
            for (int u : L) a += m.V[u];
            V3 d = a * (1.0 / L.size()) - m.V[v];
            if (bnbr[v].empty()) d = tangent(d, normalized(vn[v]));
            nv[v] = bvh.nearest(m.V[v] + d * strength);
        }
        m.V.swap(nv);
    }
}

// ------------------------------------------------------------------ main pipeline

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: plegma_core job.txt\n"); return 2; }
    Job job;
    if (!read_job(argv[1], job)) { std::fprintf(stderr, "bad job file\n"); return 2; }
    VERBOSE = job.verbose;
    TriMesh ref, m;
    if (!read_mesh(job.ref, ref) || !read_mesh(job.dense, m)) { std::fprintf(stderr, "cannot read meshes\n"); return 3; }
    // rough memory estimate: ~1 KB per dense vertex + 100 B per reference triangle
    double est_mb = (m.V.size() * 4.0 * 1024 + ref.T.size() * 100.0) / (1024.0 * 1024.0);
    if (est_mb > job.mem_limit_mb) { std::fprintf(stderr, "memory estimate %.0f MB above limit\n", est_mb); return 4; }

    stamp("read");
    weld(ref); weld(m);
    if (m.T.empty() || ref.T.empty()) { std::fprintf(stderr, "empty mesh\n"); return 5; }
    stamp("weld");
    BVH bvh; bvh.build(ref);

    double area = 0;
    for (int i = 0; i < int(ref.T.size()); ++i) area += tri_area(ref, i);
    double q0 = std::max(job.target / 4.0 * 1.2, 8.0);
    double h = std::sqrt(area / q0);

    // real openings of the reference (to keep them open later)
    EdgeInfo rei = build_edges(ref);
    std::vector<V3> ref_border;
    for (size_t e = 0; e < rei.E.size(); ++e)
        if (rei.EF[e].size() == 1) { ref_border.push_back(ref.V[rei.E[e][0]]); ref_border.push_back(ref.V[rei.E[e][1]]); }
    bool src_closed = ref_border.empty() || job.fill_all;
    rei = EdgeInfo();

    stamp("bvh+area");
    // dense base: split long edges, relax along features, project on the reference
    split_long_edges(m, 0.45 * h);
    EdgeInfo ei = build_edges(m);
    {
        Features base = find_features(m, ei, face_normals(m), 180.0, h);
        for (int k = 0; k < 2; ++k) relax_project(m, ei, base, bvh, 0.5);
    }
    stamp("dense base");
    std::vector<V3> fn = face_normals(m);
    Features F = find_features(m, ei, fn, job.sharp_angle, h);

    std::vector<V3> N = vertex_normals(m);
    for (int k = 0; k < job.normal_smooth; ++k) {
        std::vector<V3> acc = N;
        for (auto &e : ei.E) { acc[e[0]] += N[e[1]]; acc[e[1]] += N[e[0]]; }
        for (size_t i = 0; i < N.size(); ++i) N[i] = normalized(acc[i]);
    }
    for (size_t i = 0; i < N.size(); ++i) F.fdir[i] = normalized(tangent(F.fdir[i], N[i]));

    stamp("features+normals");
    std::vector<Level> lv = build_hierarchy(m.V, N, ei.E);
    stamp("hierarchy");
    std::vector<V3> Q = orientation_hier(lv, F.fdir, F.feat);
    stamp("orientation");
    std::vector<V3> O = position_hier(lv, Q, h, F.fdir, F.feat, F.corner);

    stamp("position");
    PolyMesh pm;
    extract(m.V, N, Q, O, ei.E, h, job.merge_tol, pm.V, pm.F);
    if (job.verbose) {
        int nq = 0; for (auto &f : pm.F) nq += f.size() == 4;
        std::printf("nodes %zu faces %zu quads %d\n", pm.V.size(), pm.F.size(), nq);
    }
    stamp("extract");
    // orient faces along the surface normal
    {
        BVH dense_bvh; dense_bvh.build(m);
        std::vector<V3> mfn = face_normals(m);
        for (auto &f : pm.F) {
            V3 c; for (int v : f) c += pm.V[v];
            c = c * (1.0 / f.size());
            int face = -1; dense_bvh.nearest(c, &face);
            if (face >= 0 && dot(poly_normal(pm, f), mfn[face]) < 0) std::reverse(f.begin(), f.end());
        }
    }
    // drop unused nodes
    {
        std::vector<int> map(pm.V.size(), -1); std::vector<V3> nv;
        for (auto &f : pm.F) for (int &v : f) { if (map[v] < 0) { map[v] = int(nv.size()); nv.push_back(pm.V[v]); } v = map[v]; }
        pm.V.swap(nv);
    }
    // close extraction gaps; keep real openings and the symmetry seam open
    auto near_border = [&](const V3 &p) {
        double r2 = 0.75 * h * 0.75 * h;
        for (auto &b : ref_border) if (dot(b - p, b - p) < r2) return true;
        return false;
    };
    for (auto &loop : boundary_loops(pm)) {
        bool seam = false;
        for (int v : loop) for (int ax : job.seam_axes) if (std::fabs(pm.V[v][ax]) < job.seam_eps) seam = true;
        if (seam) continue;
        if (!src_closed) {
            int real = 0;
            for (int v : loop) real += near_border(pm.V[v]);
            if (real * 2 > int(loop.size())) continue;
        }
        fill_loop(pm, loop);
    }
    for (auto &v : pm.V) v = bvh.nearest(v);
    stamp("fill+project");
    join_triangles(pm);
    PolyMesh q = split_to_quads(pm);
    relax_poly(q, bvh, 2, 0.4);
    stamp("join+split+relax");
    if (!write_polys(job.out, q.V, q.F)) { std::fprintf(stderr, "cannot write result\n"); return 6; }
    if (job.verbose) std::printf("quads %zu verts %zu\n", q.F.size(), q.V.size());
    return 0;
}
