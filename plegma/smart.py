# SPDX-License-Identifier: GPL-3.0-or-later
"""Shape-aware part segmentation (no Blender dependency, numpy only).

Parts of a model usually meet along concave creases (hair/forehead, eyelid/eye,
collar/neck). Each face-adjacency gets a cost from how concave the crease is,
optionally plus a colour difference. Faces are grouped into small "super faces"
along cheap edges, then regions are merged greedily across their weakest borders
until every remaining border is a real crease (or the part limit is reached).
"""

import heapq
import numpy as np


def _components(n, a, b):
    comp = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return comp
    while True:
        ca, cb = comp[a], comp[b]
        m = np.minimum(ca, cb)
        old = comp.copy()
        np.minimum.at(comp, ca, m)
        np.minimum.at(comp, cb, m)
        for _ in range(8):
            comp = comp[comp]
        if np.array_equal(comp, old):
            break
    _, comp = np.unique(comp, return_inverse=True)
    return comp


def edge_costs(normals, centers, a, b, lab=None, color_weight=0.0, convex_factor=0.15):
    cosang = np.clip(np.einsum("ij,ij->i", normals[a], normals[b]), -1, 1)
    ang = np.arccos(cosang)
    # concave if the neighbour's centre lies in front of this face's plane
    d = np.einsum("ij,ij->i", centers[b] - centers[a], normals[a])
    concave = d > 0
    cost = np.where(concave, ang, convex_factor * ang)
    if lab is not None and color_weight > 0:
        de = np.linalg.norm(lab[a] - lab[b], axis=1)
        cost = cost + color_weight * np.minimum(de / 25.0, 2.0) * 0.35
    return cost


def smooth_edge_costs(cost, a, b, n_faces, iters=2):
    """Average each edge cost with costs of edges around its two faces: kills single-edge noise."""
    if iters <= 0:
        return cost
    for _ in range(iters):
        fs = np.zeros(n_faces); fc = np.zeros(n_faces)
        np.add.at(fs, a, cost); np.add.at(fs, b, cost)
        np.add.at(fc, a, 1); np.add.at(fc, b, 1)
        fm = fs / np.maximum(fc, 1)
        # keep strong creases strong: blend with own value, and take max along crease direction
        cost = 0.5 * cost + 0.25 * (fm[a] + fm[b])
    return cost


def segment(areas, a, b, cost, elen, stop, max_parts=0, min_frac=0.002, log=print):
    n = len(areas)
    # 1. super faces: join across clearly cheap edges
    low = min(np.quantile(cost, 0.5), stop * 0.35)
    reg = _components(n, a[cost < low], b[cost < low])
    nreg = int(reg.max()) + 1
    log(f"  smart: {nreg} initial regions")

    # 2. region adjacency with boundary length and cost
    ra, rb = reg[a], reg[b]
    m = ra != rb
    ra, rb, c, l = ra[m], rb[m], cost[m], elen[m]
    lo = np.minimum(ra, rb); hi = np.maximum(ra, rb)
    key = lo * nreg + hi
    uk, inv = np.unique(key, return_inverse=True)
    blen = np.bincount(inv, l)
    bsum = np.bincount(inv, c * l)
    pa = (uk // nreg).astype(np.int64); pb = (uk % nreg).astype(np.int64)

    rarea = np.bincount(reg, areas, nreg).astype(float)
    total = float(rarea.sum())
    small = min_frac * total

    adj = [dict() for _ in range(nreg)]
    for x, y, L, S in zip(pa.tolist(), pb.tolist(), blen.tolist(), bsum.tolist()):
        adj[x][y] = [L, S]
        adj[y][x] = [L, S]
    parent = list(range(nreg))
    alive = nreg
    ver = [0] * nreg

    def prio(x, y):
        L, S = adj[x][y]
        mean = S / max(L, 1e-12)
        if rarea[x] < small or rarea[y] < small:
            return -1.0 + mean * 0.01  # tiny islands are always absorbed first
        # long shared borders relative to the smaller region's size are cheaper to merge
        return mean

    heap = []
    for x in range(nreg):
        for y in adj[x]:
            if x < y:
                heap.append((prio(x, y), x, y, 0, 0))
    heapq.heapify(heap)

    while heap:
        p, x, y, vx, vy = heapq.heappop(heap)
        if parent[x] != x or parent[y] != y or ver[x] != vx or ver[y] != vy or y not in adj[x]:
            continue
        if p > stop:
            if max_parts <= 0 or alive <= max_parts:
                break
        # merge smaller into larger
        if len(adj[x]) < len(adj[y]):
            x, y = y, x
        parent[y] = x
        rarea[x] += rarea[y]
        del adj[x][y]
        del adj[y][x]
        for z, (L, S) in adj[y].items():
            del adj[z][y]
            if z in adj[x]:
                adj[x][z][0] += L; adj[x][z][1] += S
            else:
                adj[x][z] = [L, S]
            adj[z][x] = adj[x][z]
        adj[y] = {}
        alive -= 1
        ver[x] += 1
        for z in adj[x]:
            u, v = (x, z) if x < z else (z, x)
            heapq.heappush(heap, (prio(u, v), u, v, ver[u], ver[v]))

    root = np.array(parent)
    for _ in range(64):
        nr = root[root]
        if np.array_equal(nr, root):
            break
        root = nr
    lab = root[reg]
    _, lab = np.unique(lab, return_inverse=True)
    return lab + 1


# ====================================================================== automatic parts (v0.9)
# colour classes + concave creases, then a second, finer pass inside the large parts only,
# small crease fragments absorbed, small islands of one colour that sit close together
# (honeycomb cells, the letters of a word) grouped into one part.

def kmeans(x, k, iters=30, seed=1):
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):                       # k-means++
        d = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
        if d.sum() <= 0:
            break
        c.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), 1)
        nc = np.array([x[lab == i].mean(0) if np.any(lab == i) else c[i] for i in range(len(c))])
        if np.allclose(nc, c, atol=1e-3):
            break
        c = nc
    return lab, c


def color_classes(lab_cols, kmax=6, min_de=20.0):
    """Material classes from colour: the largest k whose centres are all at least min_de apart
    (lightness at half weight, so shading baked into AI textures does not split a material)."""
    x = lab_cols * np.array([0.5, 1.0, 1.0])
    best = np.zeros(len(x), np.int64)
    for k in range(2, kmax + 1):
        lab, c = kmeans(x, k)
        if len(c) < k:
            break
        d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(k) * 1e9
        if d.min() < min_de or np.bincount(lab, minlength=k).min() < 0.002 * len(x):
            break
        best = lab
    return best


def _relabel(lab):
    return np.unique(lab, return_inverse=True)[1].ravel()


def _merge_small(lab, area, a, b, elen, frac, cls=None):
    """Regions below frac of the area join the neighbour with the longest shared border
    (of the same colour class when cls is given and such a neighbour exists)."""
    tot = area.sum()
    for _ in range(30):
        n = lab.max() + 1
        ra = np.bincount(lab, area, n)
        m = lab[a] != lab[b]
        s = np.concatenate([lab[a][m], lab[b][m]]); t = np.concatenate([lab[b][m], lab[a][m]])
        w = np.concatenate([elen[m]] * 2)
        if cls is not None:
            rc = np.zeros(n, np.int64); rc[lab] = cls
            w = w * np.where(rc[s] == rc[t], 1.0, 1e-3)
        keep = ra[s] < frac * tot
        if not keep.any():
            break
        s, t, w = s[keep], t[keep], w[keep]
        key = s * n + t
        uk, inv = np.unique(key, return_inverse=True)
        bw = np.bincount(inv, w)
        ss, tt = uk // n, uk % n
        o = np.lexsort((-bw, ss))
        first = np.r_[True, ss[o][1:] != ss[o][:-1]]
        tgt = np.arange(n)
        for x, y in zip(ss[o][first], tt[o][first]):
            if ra[y] >= ra[x]:
                tgt[x] = y
        new = tgt[lab]
        if np.array_equal(new, lab):
            break
        lab = _relabel(new)
    return lab


def _gap_merge(lab, cls, area, a, b, ev, verts, elen, pairs_within, gapf=0.6, small=0.01):
    """Group small islands of one colour (mostly bordered by another colour) that lie close together."""
    n = lab.max() + 1
    ra = np.bincount(lab, area, n)
    tot = area.sum()
    size = np.sqrt(ra)
    rc = np.zeros(n, np.int64); rc[lab] = cls
    bnd = lab[a] != lab[b]
    diff = bnd & (cls[a] != cls[b])
    bl = np.bincount(np.concatenate([lab[a][bnd], lab[b][bnd]]), np.concatenate([elen[bnd]] * 2), n)
    bd = np.bincount(np.concatenate([lab[a][diff], lab[b][diff]]), np.concatenate([elen[diff]] * 2), n)
    island = (bd > 0.7 * np.maximum(bl, 1e-12)) & (ra < small * tot)
    sel = bnd & (island[lab[a]] | island[lab[b]])
    if not sel.any():
        return lab
    e = ev[sel]
    vr = np.concatenate([np.stack([e[:, 0], lab[a][sel]], 1), np.stack([e[:, 1], lab[a][sel]], 1),
                         np.stack([e[:, 0], lab[b][sel]], 1), np.stack([e[:, 1], lab[b][sel]], 1)])
    vr = np.unique(vr, axis=0)
    vr = vr[island[vr[:, 1]]]
    if len(vr) < 2:
        return lab
    r = gapf * float(size[island].max())
    pi, pj = pairs_within(verts[vr[:, 0]], r)
    ri, rj = vr[pi, 1], vr[pj, 1]
    dist = np.linalg.norm(verts[vr[pi, 0]] - verts[vr[pj, 0]], axis=1)
    ok = (ri != rj) & (rc[ri] == rc[rj]) & (dist < gapf * np.minimum(size[ri], size[rj]))
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in set(zip(ri[ok].tolist(), rj[ok].tolist())):
        fx, fy = find(x), find(y)
        if fx != fy:
            parent[fy] = fx
    root = np.array([find(i) for i in range(n)])
    return _relabel(root[lab])


def segment_parts(area, a, b, ev, verts, elen, cost, cls, pairs_within, detail=0.5, min_frac=0.0005,
                  log=print):
    """Automatic parts. cost: crease cost per adjacency, cls: colour class per face (zeros = shape only)."""
    nf = len(area)
    tot = area.sum()
    stop1 = 0.30 - 0.10 * detail
    stop2 = 0.15 - 0.10 * detail
    big = 0.08
    frac = 0.03 - 0.02 * detail
    # 1. colour regions (connected faces of one colour class)
    same = cls[a] == cls[b]
    col = _components(nf, a[same], b[same])
    col = _merge_small(col, area, a, b, elen, min_frac)
    c = cost.copy()
    c[col[a] != col[b]] = 10.0
    # 2. coarse crease split
    lab = segment(area, a, b, c, elen, stop1, 0, min_frac * 0.6, log=lambda m: None) - 1
    lab = _merge_small(lab, area, a, b, elen, min_frac, cls)
    log(f"  auto parts: {len(np.unique(cls))} colours, {lab.max() + 1} coarse parts")
    # 3. finer split inside the large parts only, fragments absorbed inside them
    ra = np.bincount(lab, area) / tot
    nxt = lab.max() + 1
    lab = lab.copy()
    for r in np.flatnonzero(ra > big):
        m = lab == r
        idx = np.flatnonzero(m)
        loc = -np.ones(nf, np.int64); loc[idx] = np.arange(len(idx))
        e = m[a] & m[b]
        sub = segment(area[idx], loc[a[e]], loc[b[e]], c[e], elen[e], stop2, 0,
                      min_frac=frac * tot / area[idx].sum(), log=lambda s: None) - 1
        lab[idx] = sub + nxt
        nxt += sub.max() + 1
    lab = _relabel(lab)
    # 4. small islands of one colour close together -> one part
    lab = _gap_merge(lab, cls, area, a, b, ev, verts, elen, pairs_within)
    log(f"  auto parts: {lab.max() + 1} parts")
    return lab + 1
