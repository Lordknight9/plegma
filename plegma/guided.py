# SPDX-License-Identifier: GPL-3.0-or-later
"""Πλέγμα's own engine: remeshing that follows segment borders.

1. Feature-aware isotropic triangle remeshing: segment borders, open boundaries and
   non-manifold edges are feature lines. Feature vertices only slide along their line,
   corners (where lines meet) never move.
2. For quads: triangles are paired into quads without crossing feature lines, then every
   face is split into quads (corner, edge midpoints, centre) -> 100% quads whose edge
   loops run along the segment borders.
"""

import math
import numpy as np
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SEG = "plegma_segment"


def _features(bm, lay):
    feat = set()
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) != 2 or (lay is not None and lf[0][lay] != lf[1][lay]):
            feat.add(e)
    count = {}
    for e in feat:
        for v in e.verts:
            count[v] = count.get(v, 0) + 1
    return feat, count


def _kind(v, count):
    c = count.get(v, 0)
    return 'i' if c == 0 else ('f' if c == 2 else 'c')


def _relax_project(bm, bvh, feat, count, strength=0.6):
    bm.verts.index_update()
    bm.normal_update()
    verts = bm.verts[:]
    co = np.array([v.co for v in verts], np.float64)
    nrm = np.array([v.normal for v in verts], np.float64)
    new = co.copy()
    for i, v in enumerate(verts):
        k = _kind(v, count)
        if k == 'c':
            continue
        if k == 'f':
            nb = [e.other_vert(v) for e in v.link_edges if e in feat]
        else:
            nb = [e.other_vert(v) for e in v.link_edges]
        if not nb:
            continue
        avg = np.mean([co[o.index] for o in nb], axis=0)
        d = avg - co[i]
        if k == 'i':
            d -= nrm[i] * np.dot(d, nrm[i])
        new[i] = co[i] + strength * d
    for v, p in zip(verts, new):
        hit = bvh.find_nearest(Vector(p))
        v.co = hit[0] if hit[0] is not None else Vector(p)


def _decimate(bm, tri_target):
    """Fast pre-reduction with Blender's decimate modifier (keeps face attributes)."""
    import bpy
    n = len(bm.faces)
    if n <= tri_target:
        return bm
    me = bpy.data.meshes.new("plegma_dec")
    bm.to_mesh(me)
    ob = bpy.data.objects.new("plegma_dec", me)
    bpy.context.scene.collection.objects.link(ob)
    try:
        mod = ob.modifiers.new("dec", 'DECIMATE')
        mod.ratio = max(tri_target / n, 0.0005)
        mod.use_collapse_triangulate = True
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        out = bmesh.new()
        out.from_mesh(ev.to_mesh())
        ev.to_mesh_clear()
    finally:
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
    bm.free()
    return out


def remesh_triangles(part_bm, target, iterations=8, log=print):
    """Isotropic triangles that respect feature lines. Consumes part_bm, returns a new bmesh."""
    ref = part_bm.copy()
    bmesh.ops.triangulate(ref, faces=ref.faces)
    bvh = BVHTree.FromBMesh(ref)
    area = sum(f.calc_area() for f in ref.faces)
    L = math.sqrt(4.0 * area / (math.sqrt(3.0) * max(target, 4)))
    lo, hi = 0.8 * L, 4.0 / 3.0 * L
    bm = _decimate(part_bm, int(target * 3))
    bmesh.ops.triangulate(bm, faces=bm.faces)

    for it in range(iterations):
        lay = bm.faces.layers.int.get(SEG)
        # 1. split long edges
        long_e = [e for e in bm.edges if e.calc_length() > hi]
        if long_e:
            bmesh.ops.subdivide_edges(bm, edges=long_e, cuts=1, use_grid_fill=False)
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        # 2. collapse short edges (independent set, never across or off a feature line)
        feat, count = _features(bm, lay)
        used = set()
        merges = []
        for e in sorted(bm.edges, key=lambda e: e.calc_length()):
            if e.calc_length() >= lo:
                break
            a, b = e.verts
            if a in used or b in used:
                continue
            ka, kb = _kind(a, count), _kind(b, count)
            if ka == 'i' and kb == 'i':
                co = (a.co + b.co) * 0.5
            elif ka == 'i':
                co = b.co.copy()
            elif kb == 'i':
                co = a.co.copy()
            else:
                if e not in feat or (ka == 'c' and kb == 'c'):
                    continue
                co = a.co.copy() if ka == 'c' else (b.co.copy() if kb == 'c' else (a.co + b.co) * 0.5)
            nb = {le.other_vert(v) for v in (a, b) for le in v.link_edges} - {a, b}
            if any((o.co - co).length > hi for o in nb):
                continue
            used.update((a, b))
            used.update(nb)
            merges.append((a, b, co))
        for a, b, co in merges:
            if a.is_valid and b.is_valid:
                bmesh.ops.pointmerge(bm, verts=[a, b], merge_co=co)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=L * 1e-4)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        # 3. flips toward valence 6 (4 on lines), never on feature edges
        lay = bm.faces.layers.int.get(SEG)
        feat, count = _features(bm, lay)

        def tv(v):
            return 4 if v in count else 6
        for e in list(bm.edges):
            if not e.is_valid or e in feat or len(e.link_faces) != 2:
                continue
            f0, f1 = e.link_faces
            if len(f0.verts) != 3 or len(f1.verts) != 3:
                continue
            a, b = e.verts
            c = next(v for v in f0.verts if v not in (a, b))
            d = next(v for v in f1.verts if v not in (a, b))
            if c == d or any(le.other_vert(d) == c for le in d.link_edges):
                continue
            va, vb, vc, vd = len(a.link_edges), len(b.link_edges), len(c.link_edges), len(d.link_edges)
            before = abs(va - tv(a)) + abs(vb - tv(b)) + abs(vc - tv(c)) + abs(vd - tv(d))
            after = abs(va - 1 - tv(a)) + abs(vb - 1 - tv(b)) + abs(vc + 1 - tv(c)) + abs(vd + 1 - tv(d))
            if after < before:
                bmesh.utils.edge_rotate(e, False)
        # 4. relax (features slide along their line) + project back on the surface
        feat, count = _features(bm, bm.faces.layers.int.get(SEG))
        _relax_project(bm, bvh, feat, count)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=L * 1e-4)
    bm.normal_update()
    return bm, bvh


def _split_to_quads(bm):
    """Every n-gon -> n quads (corner, edge midpoint, centre, edge midpoint)."""
    lay = bm.faces.layers.int.get(SEG)
    out = bmesh.new()
    olay = out.faces.layers.int.new(SEG)
    vmap = {v: out.verts.new(v.co) for v in bm.verts}
    emid = {e: out.verts.new((e.verts[0].co + e.verts[1].co) * 0.5) for e in bm.edges}
    for f in bm.faces:
        c = out.verts.new(f.calc_center_median())
        loops = f.loops[:]
        for i, l in enumerate(loops):
            prev = loops[i - 1]
            try:
                nf = out.faces.new((vmap[l.vert], emid[l.edge], c, emid[prev.edge]))
                nf[olay] = f[lay] if lay is not None else 0
            except ValueError:
                pass
    out.normal_update()
    return out


def remesh_quads(part_bm, target, iterations=8, log=print, align=True):
    """All-quad remesh whose edge loops follow the segment borders and the cross field."""
    # each triangle ends up as ~4.2 quads (paired tris -> 4 quads per pair, single tris -> 3)
    tris, bvh = remesh_triangles(part_bm, max(int(target / 4.2), 8), iterations, log)
    lay = tris.faces.layers.int.get(SEG)
    feat, _ = _features(tris, lay)
    if align:
        u, n = cross_field(tris, feat)
        align_flips(tris, feat, u, n)
        feat, _ = _features(tris, lay)
        u, n = cross_field(tris, feat)
        paired = aligned_pairs(tris, feat, u, n)
        log(f"  πεδίο: {paired} ζεύγη τριγώνων στοιχισμένα")
        feat, _ = _features(tris, lay)
    for e in tris.edges:
        e.smooth = e not in feat
    bmesh.ops.join_triangles(tris, faces=[f for f in tris.faces if len(f.verts) == 3], cmp_seam=False,
                             cmp_sharp=True, cmp_uvs=False, cmp_vcols=False, cmp_materials=False,
                             angle_face_threshold=math.radians(40), angle_shape_threshold=math.radians(40))
    quads = _split_to_quads(tris)
    tris.free()
    for _ in range(3):
        feat, count = _features(quads, quads.faces.layers.int.get(SEG))
        _relax_project(quads, bvh, feat, count, strength=0.5)
    for e in quads.edges:
        e.smooth = True
    return quads


# ---------------------------------------------------------------- cross field (field alignment)

def _face_frames(bm):
    faces = bm.faces[:]
    n = np.array([f.normal for f in faces], np.float64)
    e1 = np.array([f.loops[1].vert.co - f.loops[0].vert.co for f in faces], np.float64)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1), 1e-20)[:, None]
    e2 = np.cross(n, e1)
    return faces, n, e1, e2


def _angle_in(frame_e1, frame_e2, d):
    return np.arctan2(np.einsum("ij,ij->i", d, frame_e2), np.einsum("ij,ij->i", d, frame_e1))


def cross_field(bm, feat, iterations=250, curvature_weight=0.4):
    """Smooth 4-direction field per face: locked to feature lines, soft-aligned to curvature.

    Returns a unit tangent vector per face (bm.faces order). Each face's field is the
    4 directions u, -u, n x u, -(n x u).
    """
    bm.faces.index_update()
    bm.verts.index_update()
    bm.normal_update()
    faces, n, e1, e2 = _face_frames(bm)
    F = len(faces)
    # adjacency through manifold edges
    fa, fb, dirs, isfeat = [], [], [], []
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) == 2:
            fa.append(lf[0].index); fb.append(lf[1].index)
            dirs.append(e.verts[1].co - e.verts[0].co)
            isfeat.append(e in feat)
    fa = np.array(fa, np.int64); fb = np.array(fb, np.int64)
    dirs = np.array(dirs, np.float64).reshape(-1, 3)
    rot = np.exp(4j * (_angle_in(e1[fa], e2[fa], dirs) - _angle_in(e1[fb], e2[fb], dirs)))

    # hard constraints: faces touching a feature line follow it
    cz = np.zeros(F, np.complex128)
    cw = np.zeros(F)
    for e in feat:
        d = np.array(e.verts[1].co - e.verts[0].co, np.float64)[None]
        for f in e.link_faces:
            i = f.index
            cz[i] += np.exp(4j * _angle_in(e1[i:i + 1], e2[i:i + 1], d)[0])
            cw[i] += 1.0
    # soft constraint: principal curvature direction from vertex-normal variation
    if curvature_weight > 0:
        vn = np.array([v.normal for v in bm.verts], np.float64)
        vco = np.array([v.co for v in bm.verts], np.float64)
        tri = np.array([[l.vert.index for l in f.loops[:3]] for f in faces], np.int64)
        S = np.zeros((F, 2, 2))
        for k in range(3):
            a, b = tri[:, k], tri[:, (k + 1) % 3]
            de = vco[b] - vco[a]; dn = vn[b] - vn[a]
            de2 = np.stack([np.einsum("ij,ij->i", de, e1), np.einsum("ij,ij->i", de, e2)], 1)
            dn2 = np.stack([np.einsum("ij,ij->i", dn, e1), np.einsum("ij,ij->i", dn, e2)], 1)
            S += np.einsum("fi,fj->fij", dn2, de2) / np.maximum(np.einsum("ij,ij->i", de2, de2), 1e-20)[:, None, None]
        S = 0.5 * (S + np.transpose(S, (0, 2, 1)))
        a, b, c = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
        ang = 0.5 * np.arctan2(2 * b, a - c)          # principal direction angle
        aniso = np.sqrt((a - c) ** 2 + 4 * b ** 2)    # |k1 - k2|
        if aniso.max() > 0:
            w = np.clip(aniso / (np.percentile(aniso, 90) + 1e-20), 0, 1) * curvature_weight
            free = cw == 0
            cz[free] += w[free] * np.exp(4j * ang[free])
    hard = cw > 0
    z = np.where(np.abs(cz) > 0, cz / np.maximum(np.abs(cz), 1e-20), 1.0 + 0j)
    for _ in range(iterations):
        acc = np.zeros(F, np.complex128)
        np.add.at(acc, fa, z[fb] * rot)
        np.add.at(acc, fb, z[fa] * np.conj(rot))
        acc += cz * np.where(hard, 50.0, 1.0)
        z = acc / np.maximum(np.abs(acc), 1e-20)
    theta = np.angle(z) / 4.0
    u = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    return u, n


def _edge_align(d, u, n):
    """cos(4*phi): +1 when the edge runs along the field, -1 when it is diagonal to it."""
    v = np.cross(n, u)
    phi = np.arctan2(np.dot(d, v), np.dot(d, u))
    return math.cos(4.0 * phi)


def _tri_min_angle(a, b, c):
    def ang(p, q, r):
        x, y = q - p, r - p
        lx, ly = x.length, y.length
        if lx < 1e-20 or ly < 1e-20:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, x.dot(y) / (lx * ly)))))
    return min(ang(a, b, c), ang(b, c, a), ang(c, a, b))


def align_flips(bm, feat, u, n, rounds=2):
    """Flip edges so every edge is either along the field or exactly diagonal to it."""
    for _ in range(rounds):
        bm.faces.index_update()
        flips = 0
        for e in list(bm.edges):
            if not e.is_valid or e in feat or len(e.link_faces) != 2:
                continue
            f0, f1 = e.link_faces
            if len(f0.verts) != 3 or len(f1.verts) != 3 or f0.index >= len(u) or f1.index >= len(u):
                continue
            a, b = e.verts
            c = next(v for v in f0.verts if v not in (a, b))
            d = next(v for v in f1.verts if v not in (a, b))
            if any(le.other_vert(d) == c for le in d.link_edges):
                continue
            uu = u[f0.index]; nn = n[f0.index]
            now = abs(_edge_align(np.array(b.co - a.co), uu, nn))
            new = abs(_edge_align(np.array(d.co - c.co), uu, nn))
            if new <= now + 0.15:
                continue
            if min(_tri_min_angle(c.co, d.co, a.co), _tri_min_angle(d.co, c.co, b.co)) < 20.0:
                continue
            if bmesh.utils.edge_rotate(e, False) is not None:
                flips += 1
        if flips == 0:
            break


def aligned_pairs(bm, feat, u, n, min_score=1.4):
    """Merge triangle pairs whose shared edge is diagonal to the field and whose sides follow it."""
    bm.faces.index_update()
    cand = []
    for e in bm.edges:
        if e in feat or len(e.link_faces) != 2:
            continue
        f0, f1 = e.link_faces
        if len(f0.verts) != 3 or len(f1.verts) != 3:
            continue
        uu = (u[f0.index] + u[f1.index])
        if np.linalg.norm(uu) < 1e-6:
            uu = u[f0.index]
        uu = uu / np.linalg.norm(uu)
        nn = n[f0.index]
        diag = -_edge_align(np.array(e.verts[1].co - e.verts[0].co), uu, nn)
        sides = [le for f in (f0, f1) for le in f.edges if le != e]
        side = sum(_edge_align(np.array(le.verts[1].co - le.verts[0].co), uu, nn) for le in sides) / 4.0
        # quad corner angles close to 90 degrees
        a, b = e.verts
        c = next(v for v in f0.verts if v not in (a, b))
        d = next(v for v in f1.verts if v not in (a, b))
        quad = [c.co, a.co, d.co, b.co]
        dev = 0.0
        for i in range(4):
            p, q, r = quad[i - 1], quad[i], quad[(i + 1) % 4]
            x, y = p - q, r - q
            if x.length < 1e-20 or y.length < 1e-20:
                dev = 90.0
                break
            dev = max(dev, abs(math.degrees(x.angle(y)) - 90.0))
        score = diag + side + (1.0 - dev / 90.0)
        if score >= min_score:
            cand.append((score, e))
    cand.sort(key=lambda t: -t[0])
    used = set()
    sel = []
    for score, e in cand:
        f0, f1 = e.link_faces
        if f0 in used or f1 in used:
            continue
        used.add(f0); used.add(f1)
        sel.append(e)
    if sel:
        bmesh.ops.dissolve_edges(bm, edges=sel, use_verts=False, use_face_split=False)
    return len(sel)
