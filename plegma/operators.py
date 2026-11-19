# SPDX-License-Identifier: GPL-3.0-or-later
import colorsys
import os
import time
import traceback

import numpy as np
import bpy
import bmesh

from . import segment as seg
from . import remesh_core as rc
from . import guided
from . import fieldquad
from . import core_bridge
from .engine import quadwild

FS = seg.FS
PREVIEW = "Plegma_Parts"


def _log(msg):
    print("[Πλέγμα]", msg)


def _cache_dir():
    try:
        return bpy.utils.extension_path_user(__package__, path="cache", create=True)
    except Exception:
        return os.path.join(bpy.app.tempdir, "plegma_cache")


def active_mesh(ctx):
    ob = ctx.active_object
    return ob if ob is not None and ob.type == 'MESH' else None


def _pastel(i):
    h = (i * 0.61803398875) % 1.0
    return colorsys.hsv_to_rgb(h, 0.55, 0.95)


def read_labels(ob):
    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(ob.data)
        lay = bm.faces.layers.int.get(FS)
        if lay is None:
            return np.ones(len(bm.faces), np.int64)
        return np.array([f[lay] for f in bm.faces], np.int64)
    return seg.by_face_sets(ob.data).astype(np.int64)


def refresh_parts(ob, colors=None):
    """Rebuild the part list from the object's face sets, keeping user settings."""
    me = ob.data
    labels = read_labels(ob)
    if ob.mode == 'EDIT':
        area = np.ones(len(labels))
    else:
        _, _, area = seg.face_arrays(me)
    old = {p.fs_id: (p.name, tuple(p.color), p.mode, p.density) for p in ob.plegma_parts}
    ob.plegma_parts.clear()
    total = float(area.sum()) or 1.0
    ids, counts = np.unique(labels, return_counts=True)
    for i, (fid, cnt) in enumerate(zip(ids, counts)):
        p = ob.plegma_parts.add()
        p.fs_id = int(fid)
        if int(fid) in old:
            p.name, col, p.mode, p.density = old[int(fid)]
            p.color = col
        else:
            p.name = f"Μέρος {i + 1}"
            p.color = _pastel(i)
        if colors is not None and int(fid) in colors:
            p.color = colors[int(fid)]
        p.faces = int(cnt)
        p.area_pct = 100.0 * float(area[labels == fid].sum()) / total
    ob.plegma_index = min(ob.plegma_index, max(len(ob.plegma_parts) - 1, 0))


def write_preview(ob):
    me = ob.data
    if ob.mode == 'EDIT':
        return
    labels = seg.by_face_sets(me)
    cmap = {p.fs_id: (*p.color, 1.0) for p in ob.plegma_parts}
    _, lt, _ = seg.face_arrays(me)
    lab_loop = np.repeat(labels, lt)
    keys = np.array(list(cmap.keys()) or [0])
    vals = np.array(list(cmap.values()) or [(0.6, 0.6, 0.6, 1)], np.float32)
    lut_idx = np.searchsorted(np.sort(keys), lab_loop)
    order = np.argsort(keys)
    cols = vals[order][np.clip(lut_idx, 0, len(keys) - 1)]
    ca = me.color_attributes.get(PREVIEW)
    if ca is None:
        ca = me.color_attributes.new(PREVIEW, 'BYTE_COLOR', 'CORNER')
    ca.data.foreach_set("color", cols.ravel())
    me.update()


# Blender 4.x calls the attribute colour mode 'ATTRIBUTE', 5.x calls it 'VERTEX'
_ATTR_COLOR = 'VERTEX' if 'VERTEX' in bpy.types.View3DShading.bl_rna.properties['color_type'].enum_items else 'ATTRIBUTE'


def write_seg_preview(ob):
    me = ob.data
    lab = seg.read_int_attr(me, guided.SEG)
    if lab is None:
        return
    uniq, inv = np.unique(lab, return_inverse=True)
    pal = np.array([(*_pastel(i), 1.0) for i in range(len(uniq))], np.float32)
    _, lt, _ = seg.face_arrays(me)
    cols = pal[np.repeat(inv, lt)]
    ca = me.color_attributes.get(PREVIEW) or me.color_attributes.new(PREVIEW, 'BYTE_COLOR', 'CORNER')
    ca.data.foreach_set("color", cols.ravel())
    me.update()


def show_preview(ctx, ob, on=True, segments=False):
    if on:
        if segments:
            write_seg_preview(ob)
        else:
            write_preview(ob)
        ca = ob.data.color_attributes.get(PREVIEW)
        if ca is not None:
            ob.data.color_attributes.active_color = ca
    for area in ctx.screen.areas if ctx.screen else []:
        if area.type == 'VIEW_3D':
            sh = area.spaces.active.shading
            if on:
                if sh.type not in {'SOLID'}:
                    sh.type = 'SOLID'
                sh.color_type = _ATTR_COLOR
            elif sh.color_type == _ATTR_COLOR:
                sh.color_type = 'MATERIAL'


# =============================================================== split

class PLEGMA_OT_split(bpy.types.Operator):
    """Χώρισε το μοντέλο σε μέρη (γράφονται ως Face Sets)"""
    bl_idname = "plegma.split"
    bl_label = "Χωρισμός σε μέρη"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        s = ctx.scene.plegma
        prev_mode = ob.mode
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        me = ob.data
        t = time.time()
        colors = None
        try:
            if s.split_method == 'SMART':
                lab = seg.by_smart(ob, s.detail, s.color_weight, s.max_parts, s.min_region / 100.0, log=_log)
                if s.also_loose:
                    loose = seg.by_loose(me)
                    if loose.max() > 0:
                        lab = seg.relabel(lab.astype(np.int64) * (int(loose.max()) + 1) + loose)
            elif s.split_method == 'COLOR':
                labels, mean, src = seg.by_color(ob, s.color_count, s.smooth_iters, s.min_region / 100.0)
                if labels is None:
                    self.report({'ERROR'}, "Δεν βρέθηκε texture ή vertex color. Δοκίμασε «Υλικά», «Γωνία» ή «Face Sets».")
                    return {'CANCELLED'}
                base = labels.copy()
                if s.also_loose:
                    loose = seg.by_loose(me)
                    if loose.max() > 0:
                        labels = labels.astype(np.int64) * (int(loose.max()) + 1) + loose
                lab = seg.relabel(labels)
                colors = {}
                for fid in np.unique(lab):
                    c = mean[int(base[np.flatnonzero(lab == fid)[0]])]
                    colors[int(fid)] = tuple(float(x) for x in np.clip(c, 0, 1))
            elif s.split_method == 'LOOSE':
                lab = seg.relabel(seg.by_loose(me))
            elif s.split_method == 'MATERIAL':
                lab = seg.relabel(seg.by_material(me))
            elif s.split_method == 'ANGLE':
                lab = seg.by_angle(me, np.degrees(s.angle))
                _, _, area = seg.face_arrays(me)
                a, b = seg.face_adjacency(me)
                lab = seg.relabel(seg.merge_small(lab, area, a, b, s.min_region / 100.0))
            else:
                lab = seg.by_face_sets(me)
            if s.split_method != 'FACESETS':
                seg.write_face_sets(me, lab)
            ob.plegma_parts.clear()
            refresh_parts(ob, colors)
            if s.split_method == 'COLOR':
                for p in ob.plegma_parts:
                    p.name = f"Χρώμα {p.fs_id}"
            show_preview(ctx, ob, True)
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Ο χωρισμός απέτυχε: {e}")
            return {'CANCELLED'}
        finally:
            if prev_mode != 'OBJECT' and ob.mode != prev_mode:
                bpy.ops.object.mode_set(mode=prev_mode)
        self.report({'INFO'}, f"{len(ob.plegma_parts)} μέρη σε {time.time() - t:.1f}s")
        return {'FINISHED'}


class PLEGMA_OT_refresh(bpy.types.Operator):
    """Ξαναδιάβασε τα μέρη από τα Face Sets (μετά από αλλαγές στο Sculpt Mode)"""
    bl_idname = "plegma.refresh"
    bl_label = "Ανανέωση"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        refresh_parts(ob)
        if ob.mode == 'OBJECT' and ob.data.color_attributes.get(PREVIEW):
            write_preview(ob)
        return {'FINISHED'}


class PLEGMA_OT_merge(bpy.types.Operator):
    """Ένωσε τα σημειωμένα μέρη σε ένα"""
    bl_idname = "plegma.merge"
    bl_label = "Συγχώνευση σημειωμένων"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        ob = active_mesh(ctx)
        return ob is not None and sum(p.mark for p in ob.plegma_parts) >= 2

    def execute(self, ctx):
        ob = active_mesh(ctx)
        prev = ob.mode
        if prev != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        marked = [p.fs_id for p in ob.plegma_parts if p.mark]
        keep = marked[0]
        lab = seg.by_face_sets(ob.data)
        lab[np.isin(lab, marked)] = keep
        seg.write_face_sets(ob.data, lab)
        for p in ob.plegma_parts:
            p.mark = False
        refresh_parts(ob)
        if ob.data.color_attributes.get(PREVIEW):
            write_preview(ob)
        if prev != 'OBJECT':
            bpy.ops.object.mode_set(mode=prev)
        return {'FINISHED'}


class PLEGMA_OT_select_part(bpy.types.Operator):
    """Επίλεξε στο Edit Mode τα faces του μέρους"""
    bl_idname = "plegma.select_part"
    bl_label = "Επιλογή μέρους"
    bl_options = {'REGISTER', 'UNDO'}
    index: bpy.props.IntProperty(default=-1)
    extend: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, ctx):
        ob = active_mesh(ctx)
        return ob is not None and len(ob.plegma_parts) > 0

    def execute(self, ctx):
        ob = active_mesh(ctx)
        idx = self.index if self.index >= 0 else ob.plegma_index
        fid = ob.plegma_parts[idx].fs_id
        if ob.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bm = bmesh.from_edit_mesh(ob.data)
        lay = bm.faces.layers.int.get(FS)
        if lay is None:
            return {'CANCELLED'}
        if not self.extend:
            for f in bm.faces:
                f.select_set(False)
        for f in bm.faces:
            if f[lay] == fid:
                f.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(ob.data, loop_triangles=False, destructive=False)
        return {'FINISHED'}


class PLEGMA_OT_assign(bpy.types.Operator):
    """Κάνε τα επιλεγμένα faces (Edit Mode) νέο μέρος"""
    bl_idname = "plegma.assign"
    bl_label = "Επιλεγμένα → νέο μέρος"
    bl_options = {'REGISTER', 'UNDO'}
    part_name: bpy.props.StringProperty(name="Όνομα", default="Νέο μέρος")

    @classmethod
    def poll(cls, ctx):
        ob = active_mesh(ctx)
        return ob is not None and ob.mode == 'EDIT'

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_props_dialog(self)

    def execute(self, ctx):
        ob = active_mesh(ctx)
        bm = bmesh.from_edit_mesh(ob.data)
        lay = bm.faces.layers.int.get(FS) or bm.faces.layers.int.new(FS)
        sel = [f for f in bm.faces if f.select]
        if not sel:
            self.report({'WARNING'}, "Δεν υπάρχουν επιλεγμένα faces")
            return {'CANCELLED'}
        if not ob.plegma_parts:
            for f in bm.faces:
                if f[lay] == 0:
                    f[lay] = 1
        new_id = max([f[lay] for f in bm.faces] + [0]) + 1
        for f in sel:
            f[lay] = new_id
        bmesh.update_edit_mesh(ob.data, loop_triangles=False, destructive=False)
        refresh_parts(ob)
        for p in ob.plegma_parts:
            if p.fs_id == new_id:
                p.name = self.part_name
        return {'FINISHED'}


class PLEGMA_OT_preview(bpy.types.Operator):
    """Δείξε / κρύψε τα μέρη με χρώματα στο viewport"""
    bl_idname = "plegma.preview"
    bl_label = "Προεπισκόπηση χρωμάτων"
    on: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        if ob.mode == 'EDIT' and self.on:
            bpy.ops.object.mode_set(mode='OBJECT')
        show_preview(ctx, ob, self.on)
        return {'FINISHED'}


class PLEGMA_OT_sculpt(bpy.types.Operator):
    """Διόρθωσε τα μέρη ζωγραφίζοντας Face Sets (Ctrl = σβήσιμο, Shift = εξομάλυνση)"""
    bl_idname = "plegma.sculpt"
    bl_label = "Ζωγράφισε μέρη (Sculpt)"

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        bpy.ops.object.mode_set(mode='SCULPT')
        for a in ctx.screen.areas:
            if a.type == 'VIEW_3D':
                a.spaces.active.overlay.show_sculpt_face_sets = True
        try:
            bpy.ops.wm.tool_set_by_id(name="builtin.draw_face_sets")
        except Exception:
            try:
                bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS',
                                             relative_asset_identifier="brushes/essentials_brushes-mesh_sculpt.blend/Brush/Draw Face Sets")
            except Exception:
                pass
        self.report({'INFO'}, "Ζωγράφισε τα μέρη και μετά πάτα «Ανανέωση» στο Πλέγμα")
        return {'FINISHED'}


class PLEGMA_OT_segments(bpy.types.Operator):
    """Χώρισε κάθε μέρος σε τμήματα με παρόμοια γεωμετρία. Τα όρια τους γίνονται οδηγοί για τις ακμές του remesh"""
    bl_idname = "plegma.segments"
    bl_label = "Βρες τμήματα"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        s = ctx.scene.plegma
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        t = time.time()
        try:
            detail = min(1.0, 0.35 + 0.6 * s.seg_detail)  # segment borders need a stricter scale than parts
            lab = seg.by_smart(ob, detail, 0.0, s.seg_max, s.seg_min / 100.0, log=_log, convex_factor=1.0)
            parts = seg.by_face_sets(ob.data).astype(np.int64)
            if parts.max() > 0:  # a segment never crosses a part border
                lab = seg.relabel(parts * (int(lab.max()) + 1) + lab)
            seg.write_int_attr(ob.data, guided.SEG, lab)
            show_preview(ctx, ob, True, segments=True)
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Τα τμήματα απέτυχαν: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{len(np.unique(lab))} τμήματα σε {time.time() - t:.1f}s")
        return {'FINISHED'}


class PLEGMA_OT_preview_segments(bpy.types.Operator):
    """Δείξε τα τμήματα με χρώματα"""
    bl_idname = "plegma.preview_segments"
    bl_label = "Δείξε τμήματα"

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        show_preview(ctx, ob, True, segments=True)
        return {'FINISHED'}


class PLEGMA_OT_clear_segments(bpy.types.Operator):
    """Σβήσε τα τμήματα (το remesh θα ακολουθεί μόνο τα όρια των μερών)"""
    bl_idname = "plegma.clear_segments"
    bl_label = "Καθάρισε τμήματα"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        at = ob.data.attributes.get(guided.SEG)
        if at is not None:
            ob.data.attributes.remove(at)
        show_preview(ctx, ob, False)
        return {'FINISHED'}


class PLEGMA_OT_install(bpy.types.Operator):
    """Κατέβασε τη μηχανή QuadWild (μία φορά, ~2 MB από το GitHub του QRemeshify)"""
    bl_idname = "plegma.install_engine"
    bl_label = "Λήψη μηχανής QuadWild"

    def execute(self, ctx):
        try:
            tag, n = quadwild.install()
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Αποτυχία λήψης: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"QuadWild {tag} εγκαταστάθηκε ({n} αρχεία)")
        return {'FINISHED'}


# =============================================================== remesh

class PLEGMA_OT_remesh(bpy.types.Operator):
    """Remesh κάθε μέρους ξεχωριστά και ένωση σε ένα object (κάθε μέρος = ξεχωριστό νησί, L το επιλέγει)"""
    bl_idname = "plegma.remesh"
    bl_label = "Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return active_mesh(ctx) is not None

    def execute(self, ctx):
        ob = active_mesh(ctx)
        s = ctx.scene.plegma
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        t0 = time.time()
        wm = ctx.window_manager

        dg = ctx.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        src = bmesh.new()
        src.from_mesh(me)
        ev.to_mesh_clear()
        # drop the preview colours from the source copy
        for lay in list(src.loops.layers.color):
            if lay.name == PREVIEW:
                src.loops.layers.color.remove(lay)
        src.faces.ensure_lookup_table()
        fs = src.faces.layers.int.get(FS)
        labels = np.array([f[fs] for f in src.faces]) if fs else np.ones(len(src.faces), int)
        mats = np.array([f.material_index for f in src.faces])
        areas = np.array([f.calc_area() for f in src.faces])

        parts = {p.fs_id: p for p in ob.plegma_parts}
        ids = [int(i) for i in np.unique(labels)]

        def mode_of(fid):
            p = parts.get(fid)
            m = p.mode if p else 'GLOBAL'
            return s.mode if m == 'GLOBAL' else m

        weights = {fid: areas[labels == fid].sum() * (parts[fid].density if fid in parts else 1.0)
                   for fid in ids if mode_of(fid) in {'QUAD', 'TRI'}}
        wsum = sum(weights.values()) or 1.0

        engine = s.engine
        if engine == 'AUTO':
            engine = 'GUIDED'
        if engine == 'QUADWILD' and not quadwild.is_installed():
            self.report({'ERROR'}, "Η μηχανή QuadWild δεν έχει κατέβει. Πάτα «Λήψη μηχανής QuadWild» ή διάλεξε Quadriflow.")
            src.free()
            return {'CANCELLED'}

        axes = [i for i, on in enumerate((s.sym_x, s.sym_y, s.sym_z)) if on]
        try:
            to_sym = rc.symmetry_matrix(ctx, ob, s) if axes else None
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            src.free()
            return {'CANCELLED'}

        out_meshes = []
        notes = []
        wm.progress_begin(0, len(ids))
        diag = max(ob.dimensions) / max(max(ob.scale), 1e-9) or 1.0
        try:
            for n, fid in enumerate(ids):
                wm.progress_update(n)
                mode = mode_of(fid)
                pname = parts[fid].name if fid in parts else f"Μέρος {fid}"
                if mode == 'DELETE':
                    continue
                mask = labels == fid
                part = rc.bm_from_part(src, mask)
                if not part.faces:
                    part.free()
                    continue
                mat = int(np.bincount(mats[mask]).argmax())
                tsrc = part.copy() if s.transfer else None

                if mode == 'KEEP':
                    res = part
                else:
                    target = max(int(s.target_faces * weights[fid] / wsum), 24)
                    eps = diag * 1e-4
                    work = part.copy()
                    used_axes = []
                    if axes:
                        work.transform(to_sym)
                        for ax in axes:
                            cs = [v.co[ax] for v in work.verts]
                            if min(cs) < -eps and max(cs) > eps:  # the part crosses this plane
                                trial = work.copy()
                                rc.bisect_positive(trial, ax)
                                if trial.faces:
                                    work.free()
                                    work = trial
                                    used_axes.append(ax)
                                    target = max(target // 2, 12)
                                else:
                                    trial.free()
                    sym_txt = f" (συμμετρία {''.join('XYZ'[a] for a in used_axes)})" if used_axes else ""
                    _log(f"{pname}: {mode}, στόχος {target}{sym_txt}")
                    res = None
                    if mode == 'QUAD':
                        chain = {'QUADWILD': ['QUADWILD', 'GUIDED', 'QUADRIFLOW'], 'GUIDED': ['GUIDED', 'QUADRIFLOW'],
                                 'QUADRIFLOW': ['QUADRIFLOW', 'GUIDED']}[engine]
                        if len(work.faces) < 40:
                            chain = [c for c in chain if c != 'QUADWILD']
                        for eng in chain:
                            try:
                                if eng == 'GUIDED':
                                    try:
                                        on_seam = (lambda v, _ax=tuple(used_axes), _e=eps * 5:
                                                   any(abs(v.co[a]) < _e for a in _ax)) if used_axes else None
                                        closed = not any(e.is_boundary for e in part.edges)
                                        sharp = s.sharp_angle if s.keep_sharp else 180.0
                                        if core_bridge.available():   # C++ core: fast, one process per part
                                            res = core_bridge.remesh_quads(
                                                work.copy(), target, _cache_dir(), f"part_{fid}", _log, sharp,
                                                fill_all=closed, seam_axes=used_axes, seam_eps=eps * 5)
                                        else:
                                            res = fieldquad.remesh_quads(work.copy(), target, s.tri_iterations, _log, sharp,
                                                                         fill_all=closed, keep_open=on_seam)
                                    except Exception as fe:
                                        _log(f"  το πεδίο ροής απέτυχε ({fe}), απλή μέθοδος")
                                        res = guided.remesh_quads(work.copy(), target, s.tri_iterations, _log, align=False)
                                elif eng == 'QUADWILD':
                                    res = rc.run_quadwild(work, target, s.sharp_angle if s.keep_sharp else 180,
                                                          f"{ob.name.encode('ascii', 'ignore').decode() or 'obj'}_{fid}", _log)
                                else:
                                    res = rc.run_quadriflow(work, target, s.keep_sharp, _log)
                                if res is not None and len(res.faces) > 0:
                                    break
                            except Exception as e:
                                _log(f"  {eng} απέτυχε στο «{pname}»: {e}")
                                res = None
                        if res is None:
                            notes.append(f"«{pname}» έγινε τρίγωνα (το quad απέτυχε)")
                    if res is None:
                        res, _ = guided.remesh_triangles(work.copy(), target * (2 if mode == 'QUAD' else 1),
                                                         s.tri_iterations, _log)
                    work.free()
                    if axes:
                        for ax in reversed(used_axes):
                            rc.mirror_axis(res, ax, eps)
                        res.transform(to_sym.inverted())
                        res.normal_update()
                    part.free()

                if tsrc is not None:
                    try:
                        rc.Transfer(tsrc).apply(res)
                    except Exception as e:
                        _log(f"  μεταφορά UV απέτυχε στο «{pname}»: {e}")
                    tsrc.free()

                lay = res.faces.layers.int.get(FS) or res.faces.layers.int.new(FS)
                play = res.faces.layers.int.get("plegma_part") or res.faces.layers.int.new("plegma_part")
                for f in res.faces:
                    f[lay] = fid
                    f[play] = fid
                    f.material_index = mat
                    f.smooth = True
                m = bpy.data.meshes.new(f"plegma_{fid}")
                res.to_mesh(m)
                res.free()
                out_meshes.append((fid, pname, m))
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Το remesh σταμάτησε: {e}")
            for _, _, m in out_meshes:
                bpy.data.meshes.remove(m)
            src.free()
            wm.progress_end()
            return {'CANCELLED'}
        wm.progress_end()
        src.free()

        if not out_meshes:
            self.report({'WARNING'}, "Δεν βγήκε κανένα μέρος")
            return {'CANCELLED'}

        # ---------- join into one object, one island per part
        final = bmesh.new()
        for _, _, m in out_meshes:
            final.from_mesh(m)
        new_me = bpy.data.meshes.new(f"{ob.data.name}_Plegma")
        final.to_mesh(new_me)
        final.free()
        for _, _, m in out_meshes:
            bpy.data.meshes.remove(m)
        for slot in ob.material_slots:
            new_me.materials.append(slot.material)
        new_ob = bpy.data.objects.new(f"{ob.name}_Πλέγμα", new_me)
        for c in ob.users_collection:
            c.objects.link(new_ob)
        new_ob.matrix_world = ob.matrix_world.copy()

        # part list + vertex groups on the result
        for fid, pname, _ in out_meshes:
            p = new_ob.plegma_parts.add()
            p.fs_id = fid
            p.name = pname
            if fid in parts:
                p.color = parts[fid].color
                p.density = parts[fid].density
        if s.vertex_groups:
            pa = np.empty(len(new_me.polygons), np.int32)
            new_me.attributes["plegma_part"].data.foreach_get("value", pa)
            _, lt, _ = seg.face_arrays(new_me)
            lv = np.empty(len(new_me.loops), np.int32); new_me.loops.foreach_get("vertex_index", lv)
            lp = np.repeat(pa, lt)
            for fid, pname, _ in out_meshes:
                vg = new_ob.vertex_groups.new(name=pname)
                vg.add(np.unique(lv[lp == fid]).tolist(), 1.0, 'REPLACE')
        refresh_parts(new_ob)

        if s.hide_source:
            ob.hide_set(True)
        for o in ctx.selected_objects:
            o.select_set(False)
        ctx.view_layer.objects.active = new_ob
        new_ob.select_set(True)
        show_preview(ctx, new_ob, False)

        nq = sum(1 for p in new_me.polygons if p.loop_total == 4)
        msg = (f"{len(out_meshes)} μέρη, {len(new_me.polygons)} faces "
               f"({100 * nq / max(len(new_me.polygons), 1):.0f}% quads) σε {time.time() - t0:.0f}s")
        if notes:
            msg += " | " + "; ".join(notes)
        self.report({'INFO'}, msg)
        _log(msg)
        return {'FINISHED'}


classes = (PLEGMA_OT_split, PLEGMA_OT_refresh, PLEGMA_OT_merge, PLEGMA_OT_select_part, PLEGMA_OT_assign,
           PLEGMA_OT_preview, PLEGMA_OT_sculpt, PLEGMA_OT_segments, PLEGMA_OT_preview_segments,
           PLEGMA_OT_clear_segments, PLEGMA_OT_install, PLEGMA_OT_remesh)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
