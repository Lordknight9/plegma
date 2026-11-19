# SPDX-License-Identifier: GPL-3.0-or-later
"""Field-aligned quad extraction (after the Instant Meshes method, Jakob et al. 2015).

On a dense triangle mesh:
  1. orientation field: one tangent direction per vertex, 4-way symmetric, smoothed;
     locked along feature lines (segment borders, open boundaries).
  2. position field: per vertex, the nearest node of a square lattice of spacing h,
     made consistent between neighbours; feature vertices put a lattice line on the feature.
  3. extraction: vertices that land on the same lattice node merge into one quad vertex,
     unit lattice steps become quad edges, closed loops of the graph become faces.
"""

import math
import numpy as np
import bmesh

from . import guided


def _normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-20)


def _vertex_data(bm):
    bm.verts.index_update()
    bm.normal_update()
    V = np.array([v.co for v in bm.verts], np.float64)
    N = np.array([v.normal for v in bm.verts], np.float64)
    E = np.array([(e.verts[0].index, e.verts[1].index) for e in bm.edges], np.int64).reshape(-1, 2)
    return V, N, E


def _rosy_match(qi, qj, nj):
    """Rotate qj by k*90 deg about nj so it best matches qi (vectorised)."""
    tj = np.cross(nj, qj)
    cands = np.stack([qj, tj, -qj, -tj], 1)                       # (m,4,3)
    dots = np.einsum("mk,mck->mc", qi, cands)
    return cands[np.arange(len(qi)), np.argmax(dots, 1)]


def orientation_field(V, N, E, lock_dir, lock, iterations=300):
    """lock_dir: (n,3) directions for locked vertices; lock: bool mask."""
    rng = np.random.default_rng(1)
    q = np.cross(N, rng.normal(size=V.shape))
    q = _normalize(q - N * np.einsum("ij,ij->i", q, N)[:, None])
    q[lock] = lock_dir[lock]
    a, b = E[:, 0], E[:, 1]
    for _ in range(iterations):
        acc = np.zeros_like(q)
        np.add.at(acc, a, _rosy_match(q[a], q[b], N[b]))
        np.add.at(acc, b, _rosy_match(q[b], q[a], N[a]))
        acc += q  # inertia
        acc -= N * np.einsum("ij,ij->i", acc, N)[:, None]
        q = _normalize(acc)
        q[lock] = lock_dir[lock]
    return q


def _middle(p0, n0, p1, n1):
    n0p0 = np.einsum("ij,ij->i", n0, p0); n0p1 = np.einsum("ij,ij->i", n0, p1)
    n1p0 = np.einsum("ij,ij->i", n1, p0); n1p1 = np.einsum("ij,ij->i", n1, p1)
    n0n1 = np.einsum("ij,ij->i", n0, n1)
    denom = 1.0 / (1.0 - n0n1 * n0n1 + 1e-4)
    l0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1)) * denom
    l1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0)) * denom
    return 0.5 * (p0 + p1) - 0.25 * (n0 * l0[:, None] + n1 * l1[:, None])


def _lattice(o, q, t, p, h, fn):
    d = p - o
    return o + q * (fn(np.einsum("ij,ij->i", q, d) / h) * h)[:, None] + t * (fn(np.einsum("ij,ij->i", t, d) / h) * h)[:, None]


def _compat_pos(v0, n0, q0, o0, v1, n1, q1, o1, h):
    t0 = np.cross(n0, q0); t1 = np.cross(n1, q1)
    mid = _middle(v0, n0, v1, n1)
    a = _lattice(o0, q0, t0, mid, h, np.floor)
    b = _lattice(o1, q1, t1, mid, h, np.floor)
    best = None
    for i in range(4):
        ai = a + (q0 * (i & 1) + t0 * ((i & 2) >> 1)) * h
        for j in range(4):
            bj = b + (q1 * (j & 1) + t1 * ((j & 2) >> 1)) * h
            c = np.einsum("ij,ij->i", ai - bj, ai - bj)
            if best is None:
                best = [c, ai.copy(), bj.copy()]
            else:
                m = c < best[0]
                best[0][m] = c[m]; best[1][m] = ai[m]; best[2][m] = bj[m]
    return best[1], best[2]


def position_field(V, N, Q, E, h, feat_dir, feat, corner, iterations=150, init=None, relax=0.8):
    a, b = E[:, 0], E[:, 1]
    T = np.cross(N, Q)
    O = V.copy() if init is None else init.copy()
    cnt = np.zeros(len(V)); np.add.at(cnt, a, 1); np.add.at(cnt, b, 1)
    # for feature vertices, the lattice axis perpendicular to the feature line
    perp = np.where((np.abs(np.einsum("ij,ij->i", Q, feat_dir)) >= np.abs(np.einsum("ij,ij->i", T, feat_dir)))[:, None], T, Q)
    for _ in range(iterations):
        oa, ob = _compat_pos(V[a], N[a], Q[a], O[a], V[b], N[b], Q[b], O[b], h)
        acc = np.zeros_like(O)
        np.add.at(acc, a, ob - oa)          # shift that puts a's lattice on b's
        np.add.at(acc, b, oa - ob)
        newO = O + relax * acc / np.maximum(cnt, 1)[:, None]
        newO -= N * np.einsum("ij,ij->i", newO - V, N)[:, None]
        newO = _lattice(newO, Q, T, V, h, lambda x: np.floor(x + 0.5))
        # a lattice line runs exactly along every feature line
        newO[feat] += perp[feat] * np.einsum("ij,ij->i", V[feat] - newO[feat], perp[feat])[:, None]
        newO[corner] = V[corner]
        O = newO
    return O


def _int_offset(o_from, q, t, p, h):
    d = p - o_from
    return np.stack([np.round(np.einsum("ij,ij->i", q, d) / h), np.round(np.einsum("ij,ij->i", t, d) / h)], 1)


def extract(V, N, Q, O, E, h, merge_tol=0.5):
    """Graph of lattice nodes -> faces (vertex positions, faces as index tuples)."""
    a, b = E[:, 0], E[:, 1]
    T = np.cross(N, Q)
    oa, ob = _compat_pos(V[a], N[a], Q[a], O[a], V[b], N[b], Q[b], O[b], h)
    ka = _int_offset(O[a], Q[a], T[a], oa, h)                 # a's node -> meeting point, in a's axes
    qb_in_a = _rosy_match(Q[a], Q[b], N[b])
    tb_in_a = np.cross(N[b], qb_in_a)
    kb = np.stack([np.round(np.einsum("ij,ij->i", qb_in_a, ob - O[b]) / h),
                   np.round(np.einsum("ij,ij->i", tb_in_a, ob - O[b]) / h)], 1)
    shift = ka - kb                                            # lattice step from a's node to b's node
    man = np.abs(shift).sum(1)

    # 1. collapse vertices that share a lattice node (most confident first, clusters stay < h/2 wide)
    same = np.flatnonzero(man == 0)
    gap = np.linalg.norm(O[a[same]] - O[b[same]], axis=1)
    same = same[np.argsort(gap)]
    parent = np.arange(len(V))
    csum = O.copy()
    ccnt = np.ones(len(V))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    lim2 = (merge_tol * h) ** 2
    for k in same:
        ri, rj = find(a[k]), find(b[k])
        if ri == rj:
            continue
        d = csum[ri] / ccnt[ri] - csum[rj] / ccnt[rj]
        if d.dot(d) > lim2:
            continue
        if ccnt[ri] < ccnt[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        csum[ri] += csum[rj]; ccnt[ri] += ccnt[rj]
    roots = np.array([find(i) for i in range(len(V))])
    uniq, cl = np.unique(roots, return_inverse=True)
    nC = len(uniq)
    P = csum[uniq] / ccnt[uniq][:, None]
    NC = np.zeros((nC, 3)); np.add.at(NC, cl, N); NC = _normalize(NC)

    # 2. unit steps -> graph edges between nodes
    unit = man == 1
    ea, eb = cl[a[unit]], cl[b[unit]]
    keep = ea != eb
    pairs = np.unique(np.sort(np.stack([ea[keep], eb[keep]], 1), 1), axis=0)
    nbrs = [[] for _ in range(nC)]
    for i, j in pairs:
        nbrs[i].append(j); nbrs[j].append(i)

    # 3. order neighbours counter-clockwise around each node and trace faces
    order = []
    for i in range(nC):
        nb = nbrs[i]
        if not nb:
            order.append([]); continue
        n = NC[i]
        ref = P[nb[0]] - P[i]; ref -= n * ref.dot(n)
        ref /= (np.linalg.norm(ref) + 1e-20)
        side = np.cross(n, ref)
        ang = [math.atan2((P[j] - P[i]).dot(side), (P[j] - P[i]).dot(ref)) for j in nb]
        order.append([nb[k] for k in np.argsort(ang)])
    pos = [{j: k for k, j in enumerate(order[i])} for i in range(nC)]
    used = set()
    faces = []
    for i in range(nC):
        for j in order[i]:
            if (i, j) in used:
                continue
            loop = [i]
            u, v = i, j
            ok = False
            for _ in range(8):
                used.add((u, v))
                loop.append(v)
                ov = order[v]
                # next edge: the neighbour of v just before u (clockwise) => face on the left
                w = ov[(pos[v][u] - 1) % len(ov)]
                u, v = v, w
                if v == i and u == loop[-1]:
                    used.add((u, v))
                    ok = True
                    break
                if v in loop:
                    break
            if ok and 3 <= len(loop) <= 6:
                faces.append(tuple(loop))
    return P, faces


# ---------------------------------------------------------------- multi-resolution (like Instant Meshes)

def build_hierarchy(V, N, E, min_size=300):
    """Levels of coarser vertex sets: [(V, N, E, parent_of_fine_vertex), ...] from fine to coarse."""
    levels = [(V, N, E, None)]
    while len(levels[-1][0]) > min_size:
        Vf, Nf, Ef, _ = levels[-1]
        span = np.ptp(Vf, 0).max() or 1.0
        target = max(len(Vf) // 4, min_size // 2)
        cell = span / max(target ** (1 / 2.0), 2)      # surface ~ 2D: cells ~ sqrt(target) per side
        key = np.floor(Vf / cell).astype(np.int64)
        nb = np.argmax(np.abs(Nf), 1) * 2 + (Nf[np.arange(len(Nf)), np.argmax(np.abs(Nf), 1)] > 0)
        key = np.concatenate([key, nb[:, None]], 1)
        _, par = np.unique(key, axis=0, return_inverse=True)
        par = par.ravel()
        nC = par.max() + 1
        if nC >= len(Vf) * 0.9:
            break
        Vc = np.zeros((nC, 3)); np.add.at(Vc, par, Vf); Vc /= np.bincount(par, minlength=nC)[:, None]
        Nc = np.zeros((nC, 3)); np.add.at(Nc, par, Nf); Nc = _normalize(Nc)
        ce = np.sort(np.stack([par[Ef[:, 0]], par[Ef[:, 1]]], 1), 1)
        ce = ce[ce[:, 0] != ce[:, 1]]
        Ec = np.unique(ce, axis=0)
        levels.append((Vc, Nc, Ec, par))
    return levels


def _smooth_orient(q, N, E, lock_dir, lock, iterations):
    a, b = E[:, 0], E[:, 1]
    for _ in range(iterations):
        acc = np.zeros_like(q)
        np.add.at(acc, a, _rosy_match(q[a], q[b], N[b]))
        np.add.at(acc, b, _rosy_match(q[b], q[a], N[a]))
        acc += q
        acc -= N * np.einsum("ij,ij->i", acc, N)[:, None]
        q = _normalize(acc)
        q[lock] = lock_dir[lock]
    return q


def orientation_hier(levels, lock_dir, lock, iters_fine=80, iters_coarse=400):
    # push constraints down the hierarchy
    locks = [(lock_dir, lock)]
    for Vc, Nc, Ec, par in levels[1:]:
        ld, lk = locks[-1]
        nC = len(Vc)
        cd = np.zeros((nC, 3)); ck = np.zeros(nC, bool)
        idx = np.flatnonzero(lk)
        cd[par[idx]] = ld[idx]; ck[par[idx]] = True
        cd -= Nc * np.einsum("ij,ij->i", cd, Nc)[:, None]
        locks.append((_normalize(cd), ck))
    Vc, Nc, Ec, _ = levels[-1]
    rng = np.random.default_rng(1)
    q = np.cross(Nc, rng.normal(size=Vc.shape)); q = _normalize(q)
    q = _smooth_orient(q, Nc, Ec, *locks[-1], iters_coarse)
    for lvl in range(len(levels) - 2, -1, -1):
        Vf, Nf, Ef, _ = levels[lvl]
        par = levels[lvl + 1][3]
        qf = q[par] - Nf * np.einsum("ij,ij->i", q[par], Nf)[:, None]
        q = _smooth_orient(_normalize(qf), Nf, Ef, *locks[lvl], iters_fine)
    return q


def position_hier(levels, Q, h, fdir, feat, corner, iters_fine=100, iters_coarse=200):
    # orientation per level: take the finest representative of each coarse vertex
    Qs = [Q]
    for Vc, Nc, Ec, par in levels[1:]:
        qc = np.zeros((len(Vc), 3)); qc[par] = Qs[-1]
        qc -= Nc * np.einsum("ij,ij->i", qc, Nc)[:, None]
        Qs.append(_normalize(qc))
    feats = [(fdir, feat, corner)]
    for Vc, Nc, Ec, par in levels[1:]:
        fd, fe, co = feats[-1]
        n = len(Vc)
        cfd = np.zeros((n, 3)); cfe = np.zeros(n, bool); cco = np.zeros(n, bool)
        i = np.flatnonzero(fe); cfd[par[i]] = fd[i]; cfe[par[i]] = True
        j = np.flatnonzero(co); cco[par[j]] = True; cfe[cco] = False
        feats.append((_normalize(cfd), cfe, cco))
    O = None
    for lvl in range(len(levels) - 1, -1, -1):
        Vl, Nl, El, _ = levels[lvl]
        if O is None:
            O0 = Vl.copy()
            its = iters_coarse
        else:
            par = levels[lvl + 1][3]
            O0 = O[par] - Nl * np.einsum("ij,ij->i", O[par] - Vl, Nl)[:, None]
            its = iters_fine
        O = position_field(Vl, Nl, Qs[lvl], El, h, *feats[lvl], iterations=its, init=O0)
    return O


def dense_triangles(part_bm, tri_target, max_edge=None):
    """Fast dense triangle base for the fields: C-speed decimation, long edges split, two relax passes."""
    from mathutils.bvhtree import BVHTree
    ref = part_bm.copy()
    bmesh.ops.triangulate(ref, faces=ref.faces)
    bvh = BVHTree.FromBMesh(ref)
    bm = guided._decimate(part_bm, tri_target)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    area = sum(f.calc_area() for f in bm.faces) or 1.0
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=math.sqrt(area / max(len(bm.faces), 1)) * 1e-3)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    if max_edge:
        # decimation leaves huge triangles on flat areas: the fields need several vertices per cell
        for _ in range(8):
            long_e = [e for e in bm.edges if e.calc_length() > max_edge]
            if not long_e:
                break
            bmesh.ops.subdivide_edges(bm, edges=long_e, cuts=1, use_grid_fill=False)
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    for _ in range(2):
        fe, fc = guided._features(bm, bm.faces.layers.int.get(guided.SEG))
        guided._relax_project(bm, bvh, fe, fc, strength=0.5)
    return bm, bvh


def _boundary_loops(bm):
    seen = set()
    loops = []
    for e in bm.edges:
        if not e.is_boundary or e in seen:
            continue
        stack = [e]; comp = []
        seen.add(e)
        while stack:
            x = stack.pop(); comp.append(x)
            for v in x.verts:
                for y in v.link_edges:
                    if y.is_boundary and y not in seen:
                        seen.add(y); stack.append(y)
        loops.append(comp)
    return loops


def _fill_gaps(bm, fill_all, keep_open=None, is_real_opening=None):
    """Close extraction gaps; keep real openings (and the symmetry seam) open."""
    for loop in _boundary_loops(bm):
        if keep_open is not None and any(keep_open(v) for e in loop for v in e.verts):
            continue
        if not fill_all:
            verts = {v for e in loop for v in e.verts}
            if is_real_opening is None:
                if len(loop) > 12:
                    continue
            elif sum(1 for v in verts if is_real_opening(v)) > 0.5 * len(verts):
                continue
        loop = [e for e in loop if e.is_valid]
        if not loop:
            continue
        # triangles (later paired into quads) instead of one n-gon, which would split into a star
        before = len(bm.faces)
        try:
            bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=loop)
        except Exception:
            pass
        if len(bm.faces) == before:
            bmesh.ops.holes_fill(bm, edges=[e for e in loop if e.is_valid], sides=0)


def _prune_features(feat_e, min_len):
    """Keep only feature lines that are long enough: AI / scan noise makes countless short 'creases'."""
    seen = set()
    keep = set()
    for e in feat_e:
        if e in seen:
            continue
        stack = [e]; comp = []
        seen.add(e)
        while stack:
            x = stack.pop(); comp.append(x)
            for v in x.verts:
                for y in v.link_edges:
                    if y in feat_e and y not in seen:
                        seen.add(y); stack.append(y)
        if sum(x.calc_length() for x in comp) >= min_len or any(len(x.link_faces) != 2 for x in comp):
            keep.update(comp)
    return keep


def remesh_quads(part_bm, target, iterations=6, log=print, sharp_angle=40.0, fill_all=None, keep_open=None,
                 normal_smooth=4, merge_tol=0.75):
    """Field-aligned all-quad remesh that follows segment borders. Consumes part_bm."""
    area = sum(f.calc_area() for f in part_bm.faces)
    src_closed = (not any(e.is_boundary for e in part_bm.edges)) if fill_all is None else fill_all
    real_opening = None
    bverts = [v.co.copy() for v in part_bm.verts if v.is_boundary]
    if bverts:
        from mathutils.kdtree import KDTree
        kd = KDTree(len(bverts))
        for i, c in enumerate(bverts):
            kd.insert(c, i)
        kd.balance()
    q0 = max(target / 4.0 * 1.2, 8)           # quads before the final 1->4 split (merging loses ~1/6)
    h = math.sqrt(area / q0)
    if bverts:
        real_opening = lambda v, _kd=kd, _r=0.75 * h: _kd.find(v.co)[2] < _r
    # dense isotropic triangles (~4 vertices per lattice cell), features preserved
    tris, bvh = dense_triangles(part_bm, int(q0 * 12), max_edge=0.45 * h)
    lay = tris.faces.layers.int.get(guided.SEG)
    feat_e, _ = guided._features(tris, lay)
    if sharp_angle and sharp_angle < 180:
        lim = math.radians(sharp_angle)
        feat_e |= {e for e in tris.edges if len(e.link_faces) == 2 and e.calc_face_angle(0) > lim}
    feat_e = _prune_features(feat_e, 2.0 * h)
    count = {}
    for e in feat_e:
        for v in e.verts:
            count[v] = count.get(v, 0) + 1
    V, N, E = _vertex_data(tris)
    nV = len(V)
    feat = np.zeros(nV, bool); corner = np.zeros(nV, bool)
    fdir = np.zeros((nV, 3))
    for v, c in count.items():
        fe = [e for e in v.link_edges if e in feat_e]
        if c == 1:      # end of a line: follows its edge, free to slide
            feat[v.index] = True
            fdir[v.index] = (fe[0].other_vert(v).co - v.co).normalized()
        elif c == 2:
            feat[v.index] = True
            d = (fe[0].other_vert(v).co - v.co).normalized() - (fe[1].other_vert(v).co - v.co).normalized()
            fdir[v.index] = d
        else:           # real junction of lines
            corner[v.index] = True
    fdir -= N * np.einsum("ij,ij->i", fdir, N)[:, None]
    fdir = _normalize(fdir)
    # fields use smoothed normals: relief and AI noise must not twist the flow
    for _ in range(normal_smooth):
        acc = N.copy()
        np.add.at(acc, E[:, 0], N[E[:, 1]]); np.add.at(acc, E[:, 1], N[E[:, 0]])
        N = _normalize(acc)
    fdir -= N * np.einsum("ij,ij->i", fdir, N)[:, None]
    fdir = _normalize(fdir)
    levels = build_hierarchy(V, N, E)
    Q = orientation_hier(levels, fdir, feat)
    O = position_hier(levels, Q, h, fdir, feat, corner)
    P, faces = extract(V, N, Q, O, E, h, merge_tol)
    tris.free()
    nq = sum(1 for f in faces if len(f) == 4)
    log(f"  πεδίο θέσεων: {len(P)} κόμβοι, {len(faces)} faces ({nq} quads πριν το τελικό σπάσιμο)")

    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in P]
    for f in faces:
        try:
            bm.faces.new([vs[i] for i in f])
        except ValueError:
            pass
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # small holes left by irregular spots -> n-gons (become quads in the split below)
    # the source was closed -> every hole is an extraction gap; otherwise only small ones
    _fill_gaps(bm, src_closed, keep_open, real_opening)
    for v in bm.verts:
        hit = bvh.find_nearest(v.co)
        if hit[0] is not None:
            v.co = hit[0]
    # neighbouring triangles of the lattice graph usually make one proper quad
    tri_f = [f for f in bm.faces if len(f.verts) == 3]
    if tri_f:
        bmesh.ops.join_triangles(bm, faces=tri_f, cmp_seam=False, cmp_sharp=False, cmp_uvs=False,
                                 cmp_vcols=False, cmp_materials=False,
                                 angle_face_threshold=math.radians(50), angle_shape_threshold=math.radians(60))
    quads = guided._split_to_quads(bm)
    bm.free()
    for _ in range(2):
        fe, fc = guided._features(quads, None)
        guided._relax_project(quads, bvh, fe, fc, strength=0.4)
    return quads
