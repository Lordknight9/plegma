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
    log(f"  smart: {nreg} αρχικές περιοχές")

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
