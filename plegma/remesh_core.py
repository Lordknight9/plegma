# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-part remeshing: QuadWild, Quadriflow, isotropic triangles, projection, attribute transfer."""

import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

from .engine import quadwild


# ---------------------------------------------------------------- helpers

def bm_from_part(src_bm, face_mask):
    """Copy the faces of one part into a new, cleaned, triangulated bmesh."""
    bm = src_bm.copy()
    bm.faces.ensure_lookup_table()
    kill = [f for f, keep in zip(bm.faces, face_mask) if not keep]
    bmesh.ops.delete(bm, geom=kill, context='FACES_ONLY')
    loose_e = [e for e in bm.edges if not e.link_faces]
    bmesh.ops.delete(bm, geom=loose_e, context='EDGES')
    loose_v = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose_v, context='VERTS')
    size = max((max(v.co[i] for v in bm.verts) - min(v.co[i] for v in bm.verts)) for i in range(3)) if bm.verts else 1
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(size, 1e-6) * 1e-6)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=max(size, 1e-6) * 1e-7)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
    return bm


def bm_area(bm):
    return sum(f.calc_area() for f in bm.faces)


def bisect_positive_x(bm):
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, 0), plane_no=(1, 0, 0), clear_inner=True, dist=1e-6)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])


def mirror_x(bm, eps):
    """Mirror across X=0 and weld verts on the plane."""
    for v in bm.verts:
        if abs(v.co.x) < eps:
            v.co.x = 0.0
    ret = bmesh.ops.duplicate(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:])
    dv = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
    df = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMFace)]
    for v in dv:
        v.co.x = -v.co.x
    bmesh.ops.reverse_faces(bm, faces=df)
    plane = [v for v in bm.verts if abs(v.co.x) < eps * 0.5]
    bmesh.ops.remove_doubles(bm, verts=plane, dist=eps)
    bm.normal_update()


def bisect_positive(bm, axis):
    """Keep the + side of the plane through the origin, normal to `axis` (0=X, 1=Y, 2=Z)."""
    no = [0.0, 0.0, 0.0]
    no[axis] = 1.0
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, 0), plane_no=no, clear_inner=True, dist=1e-6)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])


def mirror_axis(bm, axis, eps):
    """Mirror across the plane through the origin normal to `axis` and weld the seam."""
    # open edges near the plane are the cut: snap them onto it (tolerance = half an edge)
    lens = [e.calc_length() for e in bm.edges]
    seam_tol = max(eps, 0.5 * (sum(lens) / len(lens))) if lens else eps
    for v in bm.verts:
        if abs(v.co[axis]) < eps or (v.is_boundary and abs(v.co[axis]) < seam_tol):
            v.co[axis] = 0.0
    ret = bmesh.ops.duplicate(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:])
    dv = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
    df = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMFace)]
    for v in dv:
        v.co[axis] = -v.co[axis]
    bmesh.ops.reverse_faces(bm, faces=df)
    plane = [v for v in bm.verts if abs(v.co[axis]) < eps * 0.5]
    bmesh.ops.remove_doubles(bm, verts=plane, dist=eps)
    bm.normal_update()


def symmetry_matrix(ctx, ob, s):
    """Matrix from the object's local space to the chosen symmetry space."""
    from mathutils import Matrix
    if s.sym_space == 'LOCAL':
        return Matrix.Identity(4)
    if s.sym_space == 'GLOBAL':
        frame = Matrix.Identity(4)
    elif s.sym_space == 'CURSOR':
        frame = ctx.scene.cursor.matrix.copy()
    else:
        ref = s.sym_object
        if ref is None:
            raise RuntimeError("Pick the object that defines the symmetry axis")
        loc, rot, _ = ref.matrix_world.decompose()
        frame = Matrix.LocRotScale(loc, rot, None)
    return frame.inverted() @ ob.matrix_world


def sharp_edges(bm, angle_deg):
    """QuadWild feature list: (convex, face_index, edge_index_in_face)."""
    bm.faces.index_update()
    bm.edges.index_update()
    out = []
    lim = math.radians(angle_deg)
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue  # boundaries are handled by QuadWild itself
        if e.seam or not e.smooth or e.calc_face_angle(0) > lim:
            f = e.link_faces[0]
            ei = next(i for i, fe in enumerate(f.edges) if fe == e)
            out.append((1 if e.is_convex else 0, f.index, ei))
    return out


def bm_to_lists(bm):
    bm.verts.index_update()
    return [tuple(v.co) for v in bm.verts], [tuple(v.index for v in f.verts) for f in bm.faces]


def lists_to_bm(verts, faces):
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        try:
            bm.faces.new([vs[i] for i in f])
        except ValueError:
            pass
    bm.normal_update()
    return bm


def temp_object(bm, name="plegma_tmp"):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def free_object(ob):
    me = ob.data
    bpy.data.objects.remove(ob, do_unlink=True)
    if me.users == 0:
        bpy.data.meshes.remove(me)


# ---------------------------------------------------------------- engines

def run_quadwild(part_bm, target, sharp_angle, tag, log):
    verts, tris = bm_to_lists(part_bm)
    sharp = sharp_edges(part_bm, sharp_angle)
    v, f = quadwild.remesh(verts, tris, sharp, bpy.app.tempdir, tag, target, sharp_angle, log)
    return lists_to_bm(v, f)


def run_quadriflow(part_bm, target, sharp, log):
    ob = temp_object(part_bm)
    try:
        with bpy.context.temp_override(active_object=ob, object=ob, selected_objects=[ob],
                                       selected_editable_objects=[ob]):
            res = bpy.ops.object.quadriflow_remesh(
                use_mesh_symmetry=False, use_preserve_sharp=sharp, use_preserve_boundary=True,
                preserve_attributes=False, smooth_normals=False, mode='FACES',
                target_faces=max(int(target), 8), seed=0)
        if 'FINISHED' not in res or len(ob.data.polygons) == 0:
            raise RuntimeError("Quadriflow failed")
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        return bm
    finally:
        free_object(ob)


def decimate_to(part_bm, tri_target):
    n = len(part_bm.faces)
    if n <= tri_target:
        return part_bm
    ob = temp_object(part_bm)
    try:
        mod = ob.modifiers.new("dec", 'DECIMATE')
        mod.ratio = max(tri_target / n, 0.001)
        mod.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        bm = bmesh.new()
        bm.from_mesh(ev.to_mesh())
        ev.to_mesh_clear()
        part_bm.free()
        return bm
    finally:
        free_object(ob)


def isotropic_triangles(part_bm, target, iterations=6, log=print):
    """Botsch-Kobbelt style isotropic remeshing toward `target` triangles."""
    ref = part_bm.copy()
    bvh = BVHTree.FromBMesh(ref)
    area = bm_area(part_bm)
    L = math.sqrt(4.0 * area / (math.sqrt(3.0) * max(target, 4)))
    lo, hi = 0.8 * L, 4.0 / 3.0 * L
    bm = decimate_to(part_bm, int(target * 2.5))

    for it in range(iterations):
        # 1. split long edges
        long_e = [e for e in bm.edges if e.calc_length() > hi]
        if long_e:
            bmesh.ops.subdivide_edges(bm, edges=long_e, cuts=1, use_grid_fill=False)
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        # 2. collapse short edges (independent set, no boundary/interior mixing)
        used = set()
        col = []
        for e in sorted(bm.edges, key=lambda e: e.calc_length()):
            if e.calc_length() >= lo:
                break
            a, b = e.verts
            if a in used or b in used:
                continue
            if a.is_boundary != b.is_boundary or (a.is_boundary and not e.is_boundary):
                continue
            mid = (a.co + b.co) * 0.5
            nb = {o for v in (a, b) for le in v.link_edges for o in le.verts} - {a, b}
            if any((o.co - mid).length > hi for o in nb):
                continue
            used.add(a); used.add(b)
            used.update(nb)
            col.append(e)
        if col:
            bmesh.ops.collapse(bm, edges=col, uvs=False)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=L * 1e-4)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        # 3. flip edges toward valence 6 (4 on boundary)
        def tv(v):
            return 4 if v.is_boundary else 6
        for e in list(bm.edges):
            if not e.is_valid or e.is_boundary or len(e.link_faces) != 2:
                continue
            if any(len(f.verts) != 3 for f in e.link_faces):
                continue
            a, b = e.verts
            c = next(v for v in e.link_faces[0].verts if v not in (a, b))
            d = next(v for v in e.link_faces[1].verts if v not in (a, b))
            if c == d or any(c in (le.other_vert(d),) for le in d.link_edges):
                continue
            va, vb, vc, vd = len(a.link_edges), len(b.link_edges), len(c.link_edges), len(d.link_edges)
            before = abs(va - tv(a)) + abs(vb - tv(b)) + abs(vc - tv(c)) + abs(vd - tv(d))
            after = abs(va - 1 - tv(a)) + abs(vb - 1 - tv(b)) + abs(vc + 1 - tv(c)) + abs(vd + 1 - tv(d))
            if after < before:
                bmesh.utils.edge_rotate(e, False)
        # 4. tangential relaxation + 5. projection
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        co = np.array([v.co for v in bm.verts], np.float64)
        nrm = np.array([v.normal for v in bm.verts], np.float64)
        bm.verts.index_update()
        ed = np.array([(e.verts[0].index, e.verts[1].index) for e in bm.edges], np.int64)
        if len(ed):
            acc = np.zeros_like(co); cnt = np.zeros(len(co))
            np.add.at(acc, ed[:, 0], co[ed[:, 1]]); np.add.at(acc, ed[:, 1], co[ed[:, 0]])
            np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
            avg = acc / np.maximum(cnt, 1)[:, None]
            d = avg - co
            d -= nrm * np.einsum("ij,ij->i", d, nrm)[:, None]
            fixed = np.array([v.is_boundary for v in bm.verts])
            d[fixed] = 0
            co += 0.8 * d
        for v, p in zip(bm.verts, co):
            hit = bvh.find_nearest(Vector(p))
            v.co = hit[0] if hit[0] is not None else Vector(p)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=L * 1e-4)
    ref.free()
    bm.normal_update()
    return bm


# ---------------------------------------------------------------- transfer

class Transfer:
    """Carries UVs and colours from the original part to its new mesh by surface proximity."""

    def __init__(self, src_bm):
        self.src = src_bm
        src_bm.faces.ensure_lookup_table()
        self.bvh = BVHTree.FromBMesh(src_bm)
        self.uv_names = [l.name for l in src_bm.loops.layers.uv]
        self.col_names = [l.name for l in src_bm.loops.layers.float_color] + \
                         [l.name for l in src_bm.loops.layers.color]
        self.vcol_names = [l.name for l in src_bm.verts.layers.float_color] + \
                          [l.name for l in src_bm.verts.layers.color]
        # typical UV/3D ratio to detect faces that straddle a UV seam
        self.ratio = {}
        for name in self.uv_names:
            lay = src_bm.loops.layers.uv[name]
            r = []
            step = max(1, len(src_bm.faces) // 2000)
            for i in range(0, len(src_bm.faces), step):
                f = src_bm.faces[i]
                p3 = sum((l.vert.co - l.link_loop_next.vert.co).length for l in f.loops)
                pu = sum((l[lay].uv - l.link_loop_next[lay].uv).length for l in f.loops)
                if p3 > 0:
                    r.append(pu / p3)
            self.ratio[name] = float(np.median(r)) if r else 1.0

    def _interp(self, face, p, getter):
        tri = [l for l in face.loops][:3]
        a, b, c = (l.vert.co for l in tri)
        va, vb, vc = (getter(l) for l in tri)
        return barycentric_transform(p, a, b, c, va, vb, vc)

    def apply(self, dst_bm):
        if not (self.uv_names or self.col_names or self.vcol_names):
            return
        src = self.src
        uv_s = {n: src.loops.layers.uv[n] for n in self.uv_names}
        uv_d = {n: dst_bm.loops.layers.uv.get(n) or dst_bm.loops.layers.uv.new(n) for n in self.uv_names}
        col_s, col_d = {}, {}
        for n in self.col_names:
            lay = src.loops.layers.float_color.get(n) or src.loops.layers.color.get(n)
            col_s[n] = lay
            col_d[n] = dst_bm.loops.layers.float_color.get(n) or dst_bm.loops.layers.float_color.new(n)
        vcol_s, vcol_d = {}, {}
        for n in self.vcol_names:
            lay = src.verts.layers.float_color.get(n) or src.verts.layers.color.get(n)
            vcol_s[n] = lay
            vcol_d[n] = dst_bm.verts.layers.float_color.get(n) or dst_bm.verts.layers.float_color.new(n)

        vert_face = {}
        for v in dst_bm.verts:
            hit = self.bvh.find_nearest(v.co)
            if hit[2] is not None:
                vert_face[v] = src.faces[hit[2]]
                sf = vert_face[v]
                for n, lay in vcol_s.items():
                    c = self._interp(sf, v.co, lambda l: Vector(l.vert[lay][:3]))
                    v[vcol_d[n]] = (*c, 1.0)

        for f in dst_bm.faces:
            ch = self.bvh.find_nearest(f.calc_center_median())
            cf = src.faces[ch[2]] if ch[2] is not None else None
            for n, lay in uv_s.items():
                vals = []
                for l in f.loops:
                    sf = vert_face.get(l.vert, cf)
                    vals.append(self._interp(sf, l.vert.co, lambda sl: Vector((*sl[lay].uv, 0))))
                p3 = sum((f.loops[i].vert.co - f.loops[i - 1].vert.co).length for i in range(len(vals)))
                pu = sum((vals[i] - vals[i - 1]).length for i in range(len(vals)))
                if cf is not None and pu > 4.0 * self.ratio[n] * p3 + 1e-6:
                    vals = [self._interp(cf, l.vert.co, lambda sl: Vector((*sl[lay].uv, 0))) for l in f.loops]
                for l, val in zip(f.loops, vals):
                    l[uv_d[n]].uv = val.xy
            for n, lay in col_s.items():
                for l in f.loops:
                    sf = vert_face.get(l.vert, cf)
                    c = self._interp(sf, l.vert.co, lambda sl: Vector(sl[lay][:3]))
                    l[col_d[n]] = (*[min(max(x, 0.0), 1.0) for x in c], 1.0)
