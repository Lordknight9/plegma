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
    print("[Plegma]", msg)


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
    old = {p.fs_id: (p.name, tuple(p.color), p.mode, p.density, p.status, p.geo_key, p.cache_key, p.res_faces)
           for p in ob.plegma_parts}
    ob.plegma_parts.clear()
    total = float(area.sum()) or 1.0
    ids, counts = np.unique(labels, return_counts=True)
    for i, (fid, cnt) in enumerate(zip(ids, counts)):
        p = ob.plegma_parts.add()
        p.fs_id = int(fid)
        if int(fid) in old:
            p.name, col, p.mode, p.density, p.status, p.geo_key, p.cache_key, p.res_faces = old[int(fid)]
            p.color = col
        else:
            p.name = f"Part {i + 1}"
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
    """Split the model into parts (stored as Face Sets)"""
    bl_idname = "plegma.split"
    bl_label = "Split into parts"
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
                    self.report({'ERROR'}, "No texture or vertex colour found. Try Materials, Edge angle or Face Sets.")
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
                    p.name = f"Colour {p.fs_id}"
            show_preview(ctx, ob, True)
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Split failed: {e}")
            return {'CANCELLED'}
        finally:
            if prev_mode != 'OBJECT' and ob.mode != prev_mode:
                bpy.ops.object.mode_set(mode=prev_mode)
        self.report({'INFO'}, f"{len(ob.plegma_parts)} parts in {time.time() - t:.1f}s")
        return {'FINISHED'}


class PLEGMA_OT_refresh(bpy.types.Operator):
    """Re-read the parts from the Face Sets (after changes in Sculpt Mode)"""
    bl_idname = "plegma.refresh"
    bl_label = "Refresh"
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
    """Merge the marked parts into one"""
    bl_idname = "plegma.merge"
    bl_label = "Merge marked"
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
    """Select the faces of the part in Edit Mode"""
    bl_idname = "plegma.select_part"
    bl_label = "Select part"
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
    """Make the selected faces (Edit Mode) a new part"""
    bl_idname = "plegma.assign"
    bl_label = "Selection → new part"
    bl_options = {'REGISTER', 'UNDO'}
    part_name: bpy.props.StringProperty(name="Name", default="New part")

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
            self.report({'WARNING'}, "No faces selected")
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
    """Show / hide the parts as colours in the viewport"""
    bl_idname = "plegma.preview"
    bl_label = "Colour preview"
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
    """Fix the parts by painting Face Sets (Ctrl = erase, Shift = smooth)"""
    bl_idname = "plegma.sculpt"
    bl_label = "Paint parts (Sculpt)"

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
        self.report({'INFO'}, "Paint the parts, then press Refresh in the Plegma panel")
        return {'FINISHED'}


class PLEGMA_OT_segments(bpy.types.Operator):
    """Split every part into segments of similar geometry. Their borders guide the edge flow of the remesh"""
    bl_idname = "plegma.segments"
    bl_label = "Find segments"
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
            self.report({'ERROR'}, f"Segments failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{len(np.unique(lab))} segments in {time.time() - t:.1f}s")
        return {'FINISHED'}


class PLEGMA_OT_preview_segments(bpy.types.Operator):
    """Show the segments as colours"""
    bl_idname = "plegma.preview_segments"
    bl_label = "Show segments"

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
    """Delete the segments (the remesh then follows only the part borders)"""
    bl_idname = "plegma.clear_segments"
    bl_label = "Clear segments"
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
    """Download the QuadWild engine (once, ~2 MB from the QRemeshify GitHub)"""
    bl_idname = "plegma.install_engine"
    bl_label = "Download QuadWild engine"

    def execute(self, ctx):
        try:
            tag, n = quadwild.install()
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Download failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"QuadWild {tag} installed ({n} files)")
        return {'FINISHED'}


classes = (PLEGMA_OT_split, PLEGMA_OT_refresh, PLEGMA_OT_merge, PLEGMA_OT_select_part, PLEGMA_OT_assign,
           PLEGMA_OT_preview, PLEGMA_OT_sculpt, PLEGMA_OT_segments, PLEGMA_OT_preview_segments,
           PLEGMA_OT_clear_segments, PLEGMA_OT_install)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
