# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic face count per part.

Every part gets a target edge length L from its own shape, then faces = area / L^2:
  * base       : the Detail level (Low / Medium / High) as a fraction of the model size
  * curvature  : the chord error of an edge of length L on a surface with curvature k is k L^2 / 8,
                 so L = sqrt(8 tol / k), with k a high percentile of the (smoothed) curvature of the part
  * width      : at least ~5 quads across the narrow side of the part (PCA extents)
  * thickness  : at least ~1 quad across the thickness of thin parts (side walls)
L is clamped to [base / 6, base] so noise or a paper-thin part cannot explode the count.
"""

import numpy as np

# (tolerance as fraction of the model diagonal, base edge as fraction of the diagonal)
LEVELS = {
    'LOW': (1.2e-3, 1 / 50),
    'MEDIUM': (5e-4, 1 / 90),
    'HIGH': (2e-4, 1 / 160),
}


def vertex_curvature(co, edges, normals, smooth_iters=8):
    """Curvature per vertex: change of the smoothed normal per unit length along the edges."""
    a, b = edges[:, 0], edges[:, 1]
    n = normals.astype(np.float64)
    nv = len(co)
    deg = np.bincount(a, minlength=nv) + np.bincount(b, minlength=nv)
    def scatter(vals):           # sum of vals over the neighbours of every vertex (bincount is much faster than add.at)
        return np.stack([np.bincount(a, weights=vals[b, i], minlength=nv) +
                         np.bincount(b, weights=vals[a, i], minlength=nv) for i in range(vals.shape[1])], 1)

    for _ in range(smooth_iters):       # Laplacian smoothing of the normals removes the noise of AI meshes
        acc = n + scatter(n)
        n = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    dn = np.linalg.norm(n[a] - n[b], axis=1)
    dp = np.linalg.norm(co[a] - co[b], axis=1)
    k_e = dn / np.maximum(dp, 1e-12)
    k = np.bincount(a, weights=k_e, minlength=nv) + np.bincount(b, weights=k_e, minlength=nv)
    return k / np.maximum(deg, 1)


def weighted_percentile(x, w, q):
    if len(x) == 0:
        return 0.0
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def pca_extents(pts):
    if len(pts) < 3:
        return np.zeros(3)
    c = pts - pts.mean(0)
    _, vec = np.linalg.eigh(c.T @ c)
    proj = c @ vec
    return np.sort(proj.max(0) - proj.min(0))[::-1]      # e1 >= e2 >= e3


def part_targets(src, level, ids, factor=None, max_faces=0, log=print):
    """src: jobs.Source. Returns {part id: quad count}."""
    tol_f, base_f = LEVELS[level]
    diag = float(np.linalg.norm(src.co.max(0) - src.co.min(0))) or 1.0
    tol, base = tol_f * diag, base_f * diag
    k_face = src.face_curvature()
    out = {}
    for fid in ids:
        m = src.labels == fid
        area = float(src.areas[m].sum())
        if area <= 0:
            continue
        kp = weighted_percentile(k_face[m], src.areas[m], 0.8)
        L_curv = np.sqrt(8 * tol / kp) if kp > 1e-12 else base
        e = pca_extents(src.co[src.part_verts(m)])
        L_width = e[1] / 5 if e[1] > 0 else base
        L_thick = e[2] / 1.0 if e[2] > 0 else base
        L = min(base, L_curv, L_width, L_thick)
        L = max(L, base / 6)
        f = factor(fid) if factor else 1.0
        n = int(area / (L * L) * f)
        out[fid] = max(n, 24)
        log(f"  auto density part {fid}: area {area:.4g}, L {L / diag * 1000:.1f}‰ "
            f"(curv {L_curv / diag * 1000:.1f}, width {L_width / diag * 1000:.1f}, "
            f"thick {L_thick / diag * 1000:.1f}) -> {out[fid]} faces")
    total = sum(out.values())
    if max_faces and total > max_faces:
        sc = max_faces / total
        out = {k: max(int(v * sc), 24) for k, v in out.items()}
    return out
