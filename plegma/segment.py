# SPDX-License-Identifier: GPL-3.0-or-later
"""Splitting a mesh into parts. Every method returns one int label per face."""

import numpy as np
import bpy

FS = ".sculpt_face_set"


# ---------------------------------------------------------------- mesh arrays

def face_arrays(me):
    nf = len(me.polygons)
    ls = np.empty(nf, np.int32); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(nf, np.int32); me.polygons.foreach_get("loop_total", lt)
    area = np.empty(nf, np.float32); me.polygons.foreach_get("area", area)
    return ls, lt, area


def weld_ids(co):
    """Same id for vertices at the same position (glTF/FBX split vertices along UV seams)."""
    co = np.asarray(co, np.float64).reshape(-1, 3)
    if len(co) == 0:
        return np.zeros(0, np.int64)
    span = float(np.ptp(co, 0).max()) or 1.0
    q = np.round(co / (span * 1e-6)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.ravel()


def mesh_weld_ids(me):
    co = np.empty(len(me.vertices) * 3, np.float64); me.vertices.foreach_get("co", co)
    return weld_ids(co)


def face_adjacency(me):
    """Pairs (a, b) of faces sharing a manifold edge (by position, so UV seams don't cut)."""
    nl = len(me.loops)
    lv = np.empty(nl, np.int32); me.loops.foreach_get("vertex_index", lv)
    wid = mesh_weld_ids(me)[lv]
    ls, lt, _ = face_arrays(me)
    lf = np.repeat(np.arange(len(lt), dtype=np.int32), lt)
    start = np.repeat(ls, lt)
    nxt = start + (np.arange(nl) - start + 1) % np.repeat(lt, lt)
    lo = np.minimum(wid, wid[nxt]).astype(np.int64)
    hi = np.maximum(wid, wid[nxt]).astype(np.int64)
    le = lo * (int(wid.max()) + 1) + hi
    order = np.argsort(le, kind="stable")
    e_sorted, f_sorted = le[order], lf[order]
    same = e_sorted[1:] == e_sorted[:-1]
    # keep only edges with exactly two faces
    starts = np.flatnonzero(np.r_[True, ~same])
    counts = np.diff(np.r_[starts, len(e_sorted)])
    two = starts[counts == 2]
    a, b = f_sorted[two], f_sorted[two + 1]
    keep = a != b
    return a[keep], b[keep]


def components(n, a, b, mask=None):
    """Connected components of n nodes with edges (a,b); mask filters edges."""
    if mask is not None:
        a, b = a[mask], b[mask]
    comp = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return comp
    while True:
        ca, cb = comp[a], comp[b]
        m = np.minimum(ca, cb)
        old = comp.copy()
        np.minimum.at(comp, ca, m)
        np.minimum.at(comp, cb, m)
        # pointer jumping
        for _ in range(8):
            comp = comp[comp]
        comp = np.minimum(comp, comp[comp])
        if np.array_equal(comp, old):
            break
    _, comp = np.unique(comp, return_inverse=True)
    return comp


# ---------------------------------------------------------------- methods

def by_loose(me):
    wid = mesh_weld_ids(me)
    ev = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", ev)
    vc = components(int(wid.max()) + 1, wid[ev[0::2]], wid[ev[1::2]])[wid]
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    ls, _, _ = face_arrays(me)
    return vc[lv[ls]]


def by_material(me):
    mi = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", mi)
    return mi


def by_face_sets(me):
    at = me.attributes.get(FS)
    if at is None:
        return np.zeros(len(me.polygons), np.int64)
    v = np.empty(len(me.polygons), np.int32); at.data.foreach_get("value", v)
    return v


def by_angle(me, angle_deg):
    """Regions bounded by edges sharper than angle (like Select Linked > Delimit Normal)."""
    a, b = face_adjacency(me)
    n = np.empty(len(me.polygons) * 3, np.float32); me.polygons.foreach_get("normal", n)
    n = n.reshape(-1, 3)
    cos = np.einsum("ij,ij->i", n[a], n[b])
    return components(len(me.polygons), a, b, cos > np.cos(np.radians(angle_deg)))


# ---------------------------------------------------------------- colour

def _find_base_image(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    start = bsdf.inputs.get("Base Color") if bsdf else None
    if start is None and out is not None:
        start = out.inputs.get("Surface")
    seen = set()

    def walk(sock):
        if sock is None or not sock.is_linked:
            return None
        node = sock.links[0].from_node
        if node.name in seen:
            return None
        seen.add(node.name)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node.image
        for inp in node.inputs:
            img = walk(inp)
            if img is not None:
                return img
        return None

    img = walk(start)
    if img is None:  # any image node as last resort
        img = next((n.image for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image), None)
    return img


def _image_array(img, max_size=1024):
    w, h = img.size
    if w == 0 or h == 0:
        return None
    src = img
    tmp = None
    if max(w, h) > max_size:
        tmp = img.copy()
        s = max_size / max(w, h)
        tmp.scale(max(1, int(w * s)), max(1, int(h * s)))
        src = tmp
    w, h = src.size
    px = np.empty(w * h * 4, np.float32)
    src.pixels.foreach_get(px)
    if tmp is not None:
        bpy.data.images.remove(tmp)
    return px.reshape(h, w, 4)[:, :, :3]


def face_colors(obj):
    """Average colour per face from the base colour texture or a colour attribute."""
    me = obj.data
    ls, lt, _ = face_arrays(me)
    nl = len(me.loops)
    lf = np.repeat(np.arange(len(lt)), lt)

    uvl = me.uv_layers.active
    mats = [s.material for s in obj.material_slots]
    images = {i: _find_base_image(m) for i, m in enumerate(mats)}
    if uvl is not None and any(images.values()):
        uv = np.empty(nl * 2, np.float32); uvl.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        mi = by_material(me)
        col = np.zeros((len(lt), 3), np.float32)
        cache = {}
        for idx in np.unique(mi):
            img = images.get(int(idx)) or next((im for im in images.values() if im), None)
            if img is None:
                continue
            if img.name not in cache:
                cache[img.name] = _image_array(img)
            arr = cache[img.name]
            if arr is None:
                continue
            h, w, _ = arr.shape
            fmask = mi == idx
            lmask = fmask[lf]
            u = np.mod(uv[lmask, 0], 1.0); v = np.mod(uv[lmask, 1], 1.0)
            c_l = arr[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]
            # face centre sample in UV space
            sums = np.zeros((len(lt), 2), np.float32)
            np.add.at(sums, lf[lmask], uv[lmask])
            cuv = np.mod(sums[fmask] / lt[fmask, None], 1.0)
            c_c = arr[np.clip((cuv[:, 1] * h).astype(int), 0, h - 1), np.clip((cuv[:, 0] * w).astype(int), 0, w - 1)]
            acc = np.zeros((len(lt), 3), np.float32)
            np.add.at(acc, lf[lmask], c_l)
            col[fmask] = 0.5 * (acc[fmask] / lt[fmask, None]) + 0.5 * c_c
        return col, "texture"

    ca = me.color_attributes.active_color or (me.color_attributes[0] if len(me.color_attributes) else None)
    if ca is not None:
        n = len(ca.data)
        c = np.empty(n * 4, np.float32); ca.data.foreach_get("color", c)
        c = c.reshape(-1, 4)[:, :3]
        if ca.domain == 'POINT':
            lv = np.empty(nl, np.int32); me.loops.foreach_get("vertex_index", lv)
            c = c[lv]
        if ca.domain in ('POINT', 'CORNER'):
            acc = np.zeros((len(lt), 3), np.float32)
            np.add.at(acc, lf, c)
            return acc / lt[:, None], "color attribute"
        if ca.domain == 'FACE':
            return c, "color attribute"
    return None, None


def _to_lab(rgb):
    rgb = np.clip(rgb, 0, 1)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def kmeans(x, w, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    centers = [x[rng.choice(len(x), p=p)]]
    for _ in range(1, k):  # k-means++
        d = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), 1)
        q = d * w
        if q.sum() <= 0:
            break
        centers.append(x[rng.choice(len(x), p=q / q.sum())])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), 1)
        newc = np.array([np.average(x[lab == i], axis=0, weights=w[lab == i]) if np.any(lab == i) else c[i]
                         for i in range(len(c))])
        if np.allclose(newc, c, atol=1e-3):
            break
        c = newc
    return lab, c


def majority_filter(labels, a, b, iters):
    """Each face takes the label most of its neighbours have (memory-light, any number of labels)."""
    labels = np.asarray(labels, np.int64)
    src = np.concatenate([a, b]).astype(np.int64)
    dst = np.concatenate([b, a]).astype(np.int64)
    n = len(labels)
    for _ in range(iters):
        K = int(labels.max()) + 1
        key = src * K + labels[dst]
        uk, cnt = np.unique(key, return_counts=True)
        f, l = uk // K, uk % K
        own_n = np.zeros(n)
        same = l == labels[f]
        own_n[f[same]] = cnt[same]
        order = np.lexsort((-cnt, f))
        f, l, cnt = f[order], l[order], cnt[order]
        first = np.r_[True, f[1:] != f[:-1]]
        bf, bl, bc = f[first], l[first], cnt[first]
        change = (bl != labels[bf]) & (bc > own_n[bf] + 1.5)
        if not change.any():
            break
        labels = labels.copy()
        labels[bf[change]] = bl[change]
    return labels


def merge_small(labels, area, a, b, min_frac, rounds=6):
    """Absorb connected regions smaller than min_frac of total area into their main neighbour."""
    total = area.sum()
    for _ in range(rounds):
        reg = components(len(labels), a, b, labels[a] == labels[b])
        nreg = reg.max() + 1
        rarea = np.bincount(reg, area, nreg)
        small = rarea < min_frac * total
        if not small.any():
            break
        ra, rb = reg[a], reg[b]
        diff = ra != rb
        ra, rb = ra[diff], rb[diff]
        pairs = np.concatenate([np.stack([ra, rb], 1), np.stack([rb, ra], 1)])
        pairs = pairs[small[pairs[:, 0]]]
        if len(pairs) == 0:
            break
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        # neighbour with longest shared border (prefer larger neighbours on ties)
        order = np.lexsort((-rarea[uniq[:, 1]], -cnt, uniq[:, 0]))
        uniq = uniq[order]
        first = np.r_[True, uniq[1:, 0] != uniq[:-1, 0]]
        best = uniq[first]
        reg_label = np.zeros(nreg, labels.dtype)
        reg_label[reg] = labels
        changed = False
        for r, nb in best:
            if reg_label[r] != reg_label[nb]:
                reg_label[r] = reg_label[nb]
                changed = True
        labels = reg_label[reg]
        if not changed:
            break
    return labels


def by_color(obj, k, smooth_iters, min_frac):
    me = obj.data
    col, src = face_colors(obj)
    if col is None:
        return None, None, None
    _, _, area = face_arrays(me)
    lab = _to_lab(col)
    labels, centers = kmeans(lab, area.astype(np.float64) + 1e-12, k)
    a, b = face_adjacency(me)
    labels = majority_filter(labels, a, b, smooth_iters)
    labels = merge_small(labels, area, a, b, min_frac)
    # mean colour per final label (for the swatches)
    mean = {int(l): col[labels == l].mean(0) for l in np.unique(labels)}
    return labels, mean, src


# ---------------------------------------------------------------- write back

def relabel(labels):
    """Map arbitrary labels to 1..n ordered by first appearance."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inv] + 1


def write_int_attr(me, name, labels):
    at = me.attributes.get(name)
    if at is None:
        at = me.attributes.new(name, 'INT', 'FACE')
    at.data.foreach_set("value", np.asarray(labels).astype(np.int32))
    me.update()


def read_int_attr(me, name):
    at = me.attributes.get(name)
    if at is None:
        return None
    v = np.empty(len(me.polygons), np.int32); at.data.foreach_get("value", v)
    return v


def write_face_sets(me, labels):
    at = me.attributes.get(FS)
    if at is None:
        at = me.attributes.new(FS, 'INT', 'FACE')
    at.data.foreach_set("value", labels.astype(np.int32))
    me.update()


# ---------------------------------------------------------------- smart (shape + optional colour)

def _tri_adjacency(T):
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    f = np.tile(np.arange(len(T)), 3)
    e.sort(1)
    key = e[:, 0].astype(np.int64) * (int(T.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    k, fo, eo = key[order], f[order], e[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.diff(np.r_[starts, len(k)])
    two = starts[counts == 2]
    return fo[two], fo[two + 1], eo[two]


def by_smart(obj, detail=0.5, color_weight=0.3, max_parts=0, min_frac=0.003, proxy_faces=40000, log=print,
             convex_factor=0.15):
    """Shape-aware split: parts meet at concave creases. Works on a decimated proxy for speed."""
    from mathutils.bvhtree import BVHTree
    from . import smart

    me = obj.data
    scene = bpy.context.scene
    tmp = bpy.data.objects.new("plegma_proxy", me)
    scene.collection.objects.link(tmp)
    try:
        nf = len(me.polygons)
        if nf > proxy_faces:
            mod = tmp.modifiers.new("dec", 'DECIMATE')
            mod.ratio = proxy_faces / nf
            mod.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        ev = tmp.evaluated_get(dg)
        pm = ev.to_mesh()
        pm.calc_loop_triangles()
        V = np.empty(len(pm.vertices) * 3, np.float64); pm.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        T = np.empty(len(pm.loop_triangles) * 3, np.int64); pm.loop_triangles.foreach_get("vertices", T); T = T.reshape(-1, 3)
        ev.to_mesh_clear()
    finally:
        bpy.data.objects.remove(tmp, do_unlink=True)

    # proxy geometry
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fn = cr / np.maximum(2 * area, 1e-20)[:, None]
    cen = (p0 + p1 + p2) / 3
    # weld split vertices (UV seams) so the surface is analysed as one piece
    wv = weld_ids(V)
    Tw = wv[T]
    Vw = np.zeros((int(wv.max()) + 1, 3)); Vw[wv] = V
    a, b, ev_ = _tri_adjacency(Tw)
    elen = np.linalg.norm(Vw[ev_[:, 0]] - Vw[ev_[:, 1]], axis=1)
    # smoothed normals suppress the surface noise of AI / scanned meshes
    vn = np.zeros_like(Vw)
    for k in range(3):
        np.add.at(vn, Tw[:, k], cr)
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-20)[:, None]
    ed = np.concatenate([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]])
    for _ in range(3):
        acc = vn.copy()
        np.add.at(acc, ed[:, 0], vn[ed[:, 1]]); np.add.at(acc, ed[:, 1], vn[ed[:, 0]])
        vn = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-20)[:, None]
    sn = vn[Tw].sum(1)
    sn /= np.maximum(np.linalg.norm(sn, axis=1), 1e-20)[:, None]

    # map full faces -> nearest proxy triangle
    bvh = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(t) for t in T], all_triangles=True)
    fc = np.empty(len(me.polygons) * 3, np.float32); me.polygons.foreach_get("center", fc); fc = fc.reshape(-1, 3)
    near = np.empty(len(fc), np.int64)
    fnd = bvh.find_nearest
    for i, c in enumerate(fc):
        h = fnd(c)
        near[i] = h[2] if h[2] is not None else 0

    lab_col = None
    if color_weight > 0:
        col, _ = face_colors(obj)
        if col is not None:
            _, _, farea = face_arrays(me)
            acc = np.zeros((len(T), 3)); w = np.zeros(len(T))
            np.add.at(acc, near, col * farea[:, None]); np.add.at(w, near, farea)
            has = w > 0
            pc = np.full((len(T), 3), 0.5)
            pc[has] = acc[has] / w[has, None]
            lab_col = _to_lab(pc)

    cost = smart.edge_costs(sn, cen, a, b, lab_col, color_weight, convex_factor)
    cost = smart.smooth_edge_costs(cost, a, b, len(T), 1)
    stop = 0.45 - 0.4 * detail
    plab = smart.segment(area, a, b, cost, elen, stop, max_parts, min_frac, log)
    labels = plab[near]

    # tidy borders on the full mesh
    fa, fb = face_adjacency(me)
    _, _, farea = face_arrays(me)
    labels = majority_filter(labels - 1, fa, fb, 3)
    labels = merge_small(labels, farea, fa, fb, min_frac)
    return relabel(labels)
