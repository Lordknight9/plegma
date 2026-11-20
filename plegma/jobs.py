# SPDX-License-Identifier: GPL-3.0-or-later
"""Part queue: remesh the parts one at a time, small to large, keep every result in a cache on disk,
and build the final object SM_<name>_Plegma in the collection COL_Plegma from the cached results."""

import hashlib
import json
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
from . import density
from . import operators as ops
from .engine import quadwild

FS = seg.FS
CORE_TAG = "core0.6"
COLLECTION = "COL_Plegma"


def _log(msg):
    print("[Plegma]", msg)


# ------------------------------------------------------------------ cache on disk

def cache_dir():
    d = os.path.join(ops._cache_dir(), "parts")
    os.makedirs(d, exist_ok=True)
    return d


def cache_path(key):
    return os.path.join(cache_dir(), key + ".npz")


def cache_size_mb():
    d = cache_dir()
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1048576.0


def save_result(key, res):
    res.verts.ensure_lookup_table()
    res.verts.index_update()
    co = np.array([v.co[:] for v in res.verts], np.float32).reshape(-1, 3)
    sizes = np.array([len(f.verts) for f in res.faces], np.int32)
    idx = np.array([v.index for f in res.faces for v in f.verts], np.int32)
    np.savez(cache_path(key), co=co, sizes=sizes, idx=idx)


def load_result(key):
    d = np.load(cache_path(key))
    bm = bmesh.new()
    verts = [bm.verts.new(c) for c in d["co"]]
    pos = 0
    for n in d["sizes"]:
        try:
            bm.faces.new([verts[i] for i in d["idx"][pos:pos + n]])
        except ValueError:
            pass
        pos += int(n)
    bm.normal_update()
    return bm


def drop_result(key):
    if key:
        try:
            os.remove(cache_path(key))
        except OSError:
            pass


# ------------------------------------------------------------------ the source model

def source_of(ob):
    """A result object leads back to the object it was made from."""
    name = ob.get("plegma_source")
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    return ob


_VALUE_KEY = {'FLOAT': 'value', 'INT': 'value', 'BOOLEAN': 'value', 'INT8': 'value',
              'FLOAT_VECTOR': 'vector', 'FLOAT2': 'vector', 'FLOAT_COLOR': 'color', 'BYTE_COLOR': 'color',
              'INT32_2D': 'value', 'QUATERNION': 'value'}
_WIDTH = {'FLOAT_VECTOR': 3, 'FLOAT2': 2, 'FLOAT_COLOR': 4, 'BYTE_COLOR': 4, 'INT32_2D': 2, 'QUATERNION': 4}


class Source:
    """The evaluated source mesh as numpy arrays: parts, areas, geometry hashes.
    A part is cut out straight from the arrays, so the whole model is never copied per part."""

    def __init__(self, ctx, ob):
        dg = ctx.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        self.me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
        me = self.me
        for ca in list(me.color_attributes):
            if ca.name == ops.PREVIEW:
                me.color_attributes.remove(ca)
        self.ls, self.lt, area = seg.face_arrays(me)
        self.lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", self.lv)
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co)
        self.co = co.reshape(-1, 3).astype(np.float64)
        fs = seg.by_face_sets(me).astype(np.int64)
        self.mats = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", self.mats)
        self.labels = fs
        self.areas = area.astype(np.float64)
        self.ids = [int(i) for i in np.unique(fs)]
        self.has_seg = guided.SEG in me.attributes
        self._kface = None
        # geometry hash per part (positions of the part's corners, so it ignores index order changes)
        loop_label = fs[np.repeat(np.arange(len(self.lt)), self.lt)]
        self.geo = {}
        for fid in self.ids:
            m = loop_label == fid
            h = hashlib.md5()
            h.update(np.ascontiguousarray(self.co[self.lv[m]].astype(np.float32)).tobytes())
            h.update(np.ascontiguousarray(self.lt[fs == fid]).tobytes())
            self.geo[fid] = h.hexdigest()[:16]

    def free(self):
        if self.me is not None:
            bpy.data.meshes.remove(self.me)
            self.me = None

    # -------------------------------------------------- one part
    def _loops_of(self, faces):
        tot = self.lt[faces]
        offs = np.cumsum(tot) - tot
        loops = np.arange(int(tot.sum())) - np.repeat(offs, tot) + np.repeat(self.ls[faces], tot)
        return loops, offs

    def part_verts(self, mask):
        loops, _ = self._loops_of(np.flatnonzero(mask))
        return np.unique(self.lv[loops])

    def part_bm(self, mask, attributes=True):
        """Cleaned, triangulated bmesh of the faces in mask (with UVs, colours and the segment layer)."""
        faces = np.flatnonzero(mask)
        loops, offs = self._loops_of(faces)
        verts, inv = np.unique(self.lv[loops], return_inverse=True)
        m = bpy.data.meshes.new("plegma_part")
        try:
            m.vertices.add(len(verts))
            m.vertices.foreach_set("co", self.co[verts].astype(np.float32).ravel())
            m.loops.add(len(loops))
            m.loops.foreach_set("vertex_index", inv.astype(np.int32))
            m.polygons.add(len(faces))
            m.polygons.foreach_set("loop_start", offs.astype(np.int32))
            m.update(calc_edges=True)
            if attributes:
                pick = {'POINT': verts, 'CORNER': loops, 'FACE': faces}
                for at in self.me.attributes:
                    name = at.name
                    if at.domain not in pick or name == "position" or at.data_type not in _VALUE_KEY:
                        continue
                    if name.startswith(".") and name != seg.FS:
                        continue
                    key = _VALUE_KEY[at.data_type]
                    w = _WIDTH.get(at.data_type, 1)
                    n = len(at.data)
                    dt = np.float32 if at.data_type in {'FLOAT', 'FLOAT_VECTOR', 'FLOAT2', 'FLOAT_COLOR', 'BYTE_COLOR', 'QUATERNION'} \
                        else (bool if at.data_type == 'BOOLEAN' else np.int32)
                    buf = np.empty(n * w, dt)
                    try:
                        at.data.foreach_get(key, buf)
                    except Exception:
                        continue
                    sel = buf.reshape(n, w)[pick[at.domain]].ravel()
                    dst = m.attributes.get(name) or m.attributes.new(name, at.data_type, at.domain)
                    try:
                        dst.data.foreach_set(key, sel)
                    except Exception:
                        pass
            bm = bmesh.new()
            bm.from_mesh(m)
        finally:
            bpy.data.meshes.remove(m)
        c = self.co[verts]
        size = float((c.max(0) - c.min(0)).max()) if len(c) else 1.0
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(size, 1e-6) * 1e-6)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=max(size, 1e-6) * 1e-7)
        bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
        return bm

    # -------------------------------------------------- curvature (for the automatic density)
    def face_curvature(self):
        if self._kface is None:
            from . import density
            me = self.me
            ed = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", ed)
            vn = np.empty(len(me.vertices) * 3, np.float32)
            try:
                me.vertex_normals.foreach_get("vector", vn)
            except AttributeError:
                me.vertices.foreach_get("normal", vn)
            kv = density.vertex_curvature(self.co, ed.reshape(-1, 2), vn.reshape(-1, 3))
            fl = np.repeat(np.arange(len(self.lt)), self.lt)
            self._kface = np.bincount(fl, weights=kv[self.lv], minlength=len(self.lt)) / np.maximum(self.lt, 1)
        return self._kface


# ------------------------------------------------------------------ the job

class Job:
    """which: 'NEXT' (one part), 'ALL' (every part that needs it) or 'ONE' (the given part)."""

    def __init__(self, ctx, ob, which='ALL', index=-1, need_targets=True):
        self.ctx = ctx
        self.ob = source_of(ob)
        self.s = ctx.scene.plegma
        s = self.s
        ob = self.ob
        if ob.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if not ob.plegma_parts:
            ops.refresh_parts(ob)
        self.t0 = time.time()
        self.notes = []
        self.done = 0
        self.source = Source(ctx, ob)
        S = self.source
        self.parts = {p.fs_id: p for p in ob.plegma_parts}
        self.diag = (max(ob.dimensions) / max(max(ob.scale), 1e-9)) or 1.0

        engine = s.engine
        self.engine = 'GUIDED' if engine == 'AUTO' else engine
        if self.engine == 'QUADWILD' and not quadwild.is_installed():
            self.engine = 'GUIDED'
        self.axes = [i for i, on in enumerate((s.sym_x, s.sym_y, s.sym_z)) if on]
        self.to_sym = rc.symmetry_matrix(ctx, ob, s) if self.axes else None

        live = [fid for fid in S.ids if self.mode_of(fid) in {'QUAD', 'TRI'}]
        dens = lambda fid: self.parts[fid].density if fid in self.parts else 1.0
        if not need_targets:
            self.target = {}
        elif s.density_mode == 'AUTO':
            self.target = density.part_targets(S, s.detail_level, live, factor=dens,
                                               max_faces=s.max_faces, log=_log)
        else:
            weights = {fid: S.areas[S.labels == fid].sum() * dens(fid) for fid in live}
            wsum = sum(weights.values()) or 1.0
            self.target = {fid: max(int(s.target_faces * w / wsum), 24) for fid, w in weights.items()}

        # small to large
        self.order = sorted(S.ids, key=lambda f: S.areas[S.labels == f].sum())
        self.keys = {}
        for fid in S.ids:
            p = self.parts.get(fid)
            if p is None:
                continue
            self.keys[fid] = self.make_key(fid)
            if p.geo_key != S.geo[fid]:          # the geometry of the part changed: old result is void
                p.status = 'PENDING'
                p.geo_key = S.geo[fid]
                p.cache_key = ""

        self.todo = []
        for fid in self.order:
            p = self.parts.get(fid)
            if p is None or self.mode_of(fid) == 'DELETE':
                continue
            if which == 'ONE':
                if p is not None and index >= 0 and ob.plegma_parts[index].fs_id == fid:
                    self.todo.append(fid)
                continue
            if self.needs_work(fid):
                self.todo.append(fid)
        if which == 'NEXT':
            self.todo = self.todo[:1]
        self.total = len(self.todo)

    # -------------------------------------------------- helpers
    def mode_of(self, fid):
        p = self.parts.get(fid)
        m = p.mode if p else 'GLOBAL'
        return self.s.mode if m == 'GLOBAL' else m

    def make_key(self, fid):
        s = self.s
        d = dict(v=CORE_TAG, mode=self.mode_of(fid), engine=self.engine, target=self.target.get(fid, 0),
                 sharp=s.sharp_angle if s.keep_sharp else 180.0, axes=self.axes, tri=s.tri_iterations,
                 sym=None if self.to_sym is None else [round(x, 5) for r in self.to_sym for x in r],
                 seg=self.source.has_seg)
        return hashlib.md5((self.source.geo[fid] + json.dumps(d, sort_keys=True)).encode()).hexdigest()[:20]

    def needs_work(self, fid):
        p = self.parts[fid]
        have = p.cache_key and os.path.isfile(cache_path(p.cache_key))
        if p.status == 'APPROVED' and have:
            return False                          # locked
        if p.status in {'PENDING', 'REDO'} or not have:
            return True
        if p.cache_key != self.keys[fid]:         # settings changed since the result was made
            return True
        return False

    # -------------------------------------------------- one part
    def step(self):
        """Remesh the next part of the queue. Returns the part name, or None when the queue is empty."""
        if not self.todo:
            return None
        fid = self.todo.pop(0)
        p = self.parts[fid]
        t = time.time()
        try:
            res = self.remesh_part(fid, p.name)
            if res is None:
                return p.name
            drop_result(p.cache_key)
            key = self.keys[fid]
            save_result(key, res)
            p.res_faces = len(res.faces)
            res.free()
            p.cache_key = key
            p.geo_key = self.source.geo[fid]
            p.status = 'DONE'
            self.done += 1
            _log(f"{p.name}: {p.res_faces} faces in {time.time() - t:.1f}s")
        except Exception as e:
            traceback.print_exc()
            self.notes.append(f"«{p.name}» failed: {e}")
            p.status = 'PENDING'
        return p.name

    def remesh_part(self, fid, pname):
        s = self.s
        S = self.source
        mode = self.mode_of(fid)
        mask = S.labels == fid
        part = S.part_bm(mask, attributes=S.has_seg)
        if not part.faces:
            part.free()
            return None
        if mode == 'KEEP':
            return part

        target = self.target[fid] * (2 if mode == 'TRI' else 1)
        eps = self.diag * 1e-4
        work = part.copy()
        used_axes = []
        if self.axes:
            work.transform(self.to_sym)
            for ax in self.axes:
                cs = [v.co[ax] for v in work.verts]
                if min(cs) < -eps and max(cs) > eps:      # the part crosses this plane
                    trial = work.copy()
                    rc.bisect_positive(trial, ax)
                    if trial.faces:
                        work.free()
                        work = trial
                        used_axes.append(ax)
                        target = max(target // 2, 12)
                    else:
                        trial.free()
        sym_txt = f" (symmetry {''.join('XYZ'[a] for a in used_axes)})" if used_axes else ""
        _log(f"{pname}: {mode}, target {target}{sym_txt}")

        res = None
        closed = not any(e.is_boundary for e in part.edges)
        sharp = s.sharp_angle if s.keep_sharp else 180.0
        if mode == 'QUAD':
            chain = {'GUIDED': ['GUIDED', 'QUADRIFLOW'], 'QUADRIFLOW': ['QUADRIFLOW', 'GUIDED'],
                     'QUADWILD': ['QUADWILD', 'GUIDED', 'QUADRIFLOW']}[self.engine]
            if len(work.faces) < 40:
                chain = [c for c in chain if c != 'QUADWILD']
            for eng in chain:
                try:
                    if eng == 'GUIDED':
                        try:
                            if core_bridge.available():       # C++ core: fast, one process per part
                                res = core_bridge.remesh_quads(
                                    work.copy(), target, ops._cache_dir(), f"part_{fid}", _log, sharp,
                                    fill_all=closed, seam_axes=used_axes, seam_eps=eps * 5,
                                    mem_limit_mb=s.mem_limit_mb)
                            else:
                                on_seam = (lambda v, _ax=tuple(used_axes), _e=eps * 5:
                                           any(abs(v.co[a]) < _e for a in _ax)) if used_axes else None
                                res = fieldquad.remesh_quads(work.copy(), target, s.tri_iterations, _log, sharp,
                                                             fill_all=closed, keep_open=on_seam)
                        except Exception as fe:
                            _log(f"  flow field failed ({fe}), simple method")
                            res = guided.remesh_quads(work.copy(), target, s.tri_iterations, _log, align=False)
                    elif eng == 'QUADWILD':
                        res = rc.run_quadwild(work, target, sharp, f"part_{fid}", _log)
                    else:
                        res = rc.run_quadriflow(work, target, s.keep_sharp, _log)
                    if res is not None and len(res.faces) > 0:
                        break
                except Exception as e:
                    _log(f"  {eng} failed on «{pname}»: {e}")
                    res = None
            if res is None:
                self.notes.append(f"«{pname}» became triangles (quad failed)")
        if res is None:
            res, _ = guided.remesh_triangles(work.copy(), target * (2 if mode == 'QUAD' else 1),
                                             s.tri_iterations, _log)
        work.free()
        if self.axes:
            for ax in reversed(used_axes):
                rc.mirror_axis(res, ax, eps)
            res.transform(self.to_sym.inverted())
            res.normal_update()
        part.free()
        return res

    # -------------------------------------------------- the final object
    def build(self, select=False):
        """SM_<name>_Plegma in COL_Plegma, from all cached results."""
        ctx, s, ob, S = self.ctx, self.s, self.ob, self.source
        pieces = []
        for fid in S.ids:
            p = self.parts.get(fid)
            if p is None or self.mode_of(fid) == 'DELETE':
                continue
            if p.status not in {'DONE', 'APPROVED'} or not p.cache_key or not os.path.isfile(cache_path(p.cache_key)):
                continue
            res = load_result(p.cache_key)
            if s.transfer:
                try:
                    tsrc = S.part_bm(S.labels == fid)
                    rc.Transfer(tsrc).apply(res)
                    tsrc.free()
                except Exception as e:
                    _log(f"  UV/colour transfer failed on «{p.name}»: {e}")
            lay = res.faces.layers.int.get(FS) or res.faces.layers.int.new(FS)
            play = res.faces.layers.int.get("plegma_part") or res.faces.layers.int.new("plegma_part")
            mat = int(np.bincount(S.mats[S.labels == fid]).argmax())
            for f in res.faces:
                f[lay] = fid
                f[play] = fid
                f.material_index = mat
                f.smooth = True
            m = bpy.data.meshes.new(f"plegma_{fid}")
            res.to_mesh(m)
            res.free()
            pieces.append((fid, p, m))
        if not pieces:
            return None

        final = bmesh.new()
        for _, _, m in pieces:
            final.from_mesh(m)
        name = f"SM_{ob.name}_Plegma"
        new_me = bpy.data.meshes.new(name)
        final.to_mesh(new_me)
        final.free()
        for _, _, m in pieces:
            bpy.data.meshes.remove(m)
        for slot in ob.material_slots:
            new_me.materials.append(slot.material)

        col = bpy.data.collections.get(COLLECTION)
        if col is None:
            col = bpy.data.collections.new(COLLECTION)
            ctx.scene.collection.children.link(col)
        elif col.name not in ctx.scene.collection.children_recursive and col not in ctx.scene.collection.children_recursive:
            try:
                ctx.scene.collection.children.link(col)
            except RuntimeError:
                pass
        new_ob = bpy.data.objects.get(name)
        if new_ob is not None and new_ob.type == 'MESH':
            old_me = new_ob.data
            new_ob.data = new_me
            if old_me.users == 0:
                bpy.data.meshes.remove(old_me)
            for vg in list(new_ob.vertex_groups):
                new_ob.vertex_groups.remove(vg)
            new_ob.plegma_parts.clear()
        else:
            new_ob = bpy.data.objects.new(name, new_me)
        if new_ob.name not in col.objects:
            col.objects.link(new_ob)
        new_ob.matrix_world = ob.matrix_world.copy()
        new_ob["plegma_source"] = ob.name
        ob["plegma_result"] = new_ob.name

        for fid, p, _ in pieces:
            q = new_ob.plegma_parts.add()
            q.fs_id = fid
            q.name = p.name
            q.color = p.color
            q.density = p.density
            q.status = p.status
        if s.vertex_groups:
            pa = np.empty(len(new_me.polygons), np.int32)
            new_me.attributes["plegma_part"].data.foreach_get("value", pa)
            _, lt, _ = seg.face_arrays(new_me)
            lv = np.empty(len(new_me.loops), np.int32); new_me.loops.foreach_get("vertex_index", lv)
            lp = np.repeat(pa, lt)
            for fid, p, _ in pieces:
                vg = new_ob.vertex_groups.new(name=p.name)
                vg.add(np.unique(lv[lp == fid]).tolist(), 1.0, 'REPLACE')
        ops.refresh_parts(new_ob)
        for q in new_ob.plegma_parts:               # refresh_parts rebuilds the list: keep the queue status
            src = self.parts.get(q.fs_id)
            if src is not None:
                q.status = src.status
        if select:
            if s.hide_source:
                ob.hide_set(True)
            for o in ctx.selected_objects:
                o.select_set(False)
            ctx.view_layer.objects.active = new_ob
            new_ob.select_set(True)
            ops.show_preview(ctx, new_ob, False)
        return new_ob

    def finish(self, select=True):
        new_ob = None
        try:
            new_ob = self.build(select=select)
        except Exception as e:
            traceback.print_exc()
            self.notes.append(f"build failed: {e}")
        self.source.free()
        return new_ob

    def summary(self, new_ob):
        n = len(new_ob.data.polygons) if new_ob else 0
        nq = sum(1 for p in new_ob.data.polygons if p.loop_total == 4) if new_ob else 0
        msg = (f"{self.done}/{self.total} parts remeshed, {n} faces "
               f"({100 * nq / max(n, 1):.0f}% quads) in {time.time() - self.t0:.0f}s")
        if self.notes:
            msg += " | " + "; ".join(self.notes)
        return msg


# ------------------------------------------------------------------ operators

def counts(ob):
    c = {'PENDING': 0, 'DONE': 0, 'APPROVED': 0, 'REDO': 0}
    for p in ob.plegma_parts:
        if p.mode != 'DELETE':
            c[p.status] += 1
    return c


class PLEGMA_OT_process(bpy.types.Operator):
    """Remesh the parts of the queue one by one, small to large. Esc stops after the current part"""
    bl_idname = "plegma.process"
    bl_label = "Process"
    bl_options = {'REGISTER'}

    which: bpy.props.EnumProperty(items=[
        ('NEXT', "Next part", "Remesh only the next waiting part"),
        ('ALL', "All pending", "Remesh every part that is waiting or changed"),
        ('ONE', "Selected part", "Remesh the part selected in the list"),
    ], default='ALL')
    index: bpy.props.IntProperty(default=-1)

    _timer = None
    _job = None

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return ob is not None and ob.type == 'MESH' and not ctx.scene.plegma.queue_running

    def _start(self, ctx):
        ob = ops.active_mesh(ctx)
        idx = self.index
        if self.which == 'ONE' and idx < 0:
            idx = source_of(ob).plegma_index
        try:
            self._job = Job(ctx, ob, self.which, idx)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return False
        return True

    def execute(self, ctx):          # synchronous (scripts, tests)
        if not self._start(ctx):
            return {'CANCELLED'}
        job = self._job
        while job.step() is not None:
            pass
        new_ob = job.finish(select=True)
        self.report({'INFO'}, job.summary(new_ob))
        _log(job.summary(new_ob))
        return {'FINISHED'}

    def invoke(self, ctx, event):    # interactive: one part per timer tick, Esc cancels, UI stays alive
        if not self._start(ctx):
            return {'CANCELLED'}
        if not self._job.todo:
            new_ob = self._job.finish(select=False)
            self.report({'INFO'}, "Nothing to do: every part is already remeshed")
            return {'FINISHED'}
        ctx.scene.plegma.queue_running = True
        wm = ctx.window_manager
        self._timer = wm.event_timer_add(0.05, window=ctx.window)
        wm.modal_handler_add(self)
        wm.progress_begin(0, max(self._job.total, 1))
        return {'RUNNING_MODAL'}

    def _end(self, ctx, cancelled):
        wm = ctx.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        ctx.scene.plegma.queue_running = False
        job = self._job
        new_ob = job.finish(select=True)
        msg = job.summary(new_ob) + (" (stopped)" if cancelled else "")
        self.report({'INFO'}, msg)
        _log(msg)
        for a in ctx.screen.areas:
            a.tag_redraw()
        return {'CANCELLED' if cancelled else 'FINISHED'}

    def modal(self, ctx, event):
        job = self._job
        if event.type == 'ESC':
            return self._end(ctx, True)
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        name = job.step()
        ctx.window_manager.progress_update(job.total - len(job.todo))
        if ctx.workspace:
            ctx.workspace.status_text_set(f"Plegma: {job.total - len(job.todo)}/{job.total} parts (Esc = stop)")
        for a in ctx.screen.areas:
            a.tag_redraw()
        if name is None or not job.todo:
            if ctx.workspace:
                ctx.workspace.status_text_set(None)
            return self._end(ctx, False)
        return {'RUNNING_MODAL'}


class PLEGMA_OT_build(bpy.types.Operator):
    """Rebuild SM_<name>_Plegma in COL_Plegma from the finished parts"""
    bl_idname = "plegma.build"
    bl_label = "Build final object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return ob is not None and ob.type == 'MESH'

    def execute(self, ctx):
        job = Job(ctx, ops.active_mesh(ctx), 'ONE', -1, need_targets=False)   # nothing queued: just load the source
        new_ob = job.finish(select=True)
        if new_ob is None:
            self.report({'WARNING'}, "No finished parts yet. Press Process first.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{new_ob.name}: {len(new_ob.data.polygons)} faces")
        return {'FINISHED'}


class PLEGMA_OT_part_action(bpy.types.Operator):
    """Approve (lock) a finished part, or mark it to be computed again"""
    bl_idname = "plegma.part_action"
    bl_label = "Part action"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(items=[
        ('APPROVE', "Approve", "Lock the result of this part"),
        ('UNLOCK', "Unlock", "Allow the part to change again"),
        ('REDO', "Redo", "Throw the result away and compute the part again"),
        ('APPROVE_ALL', "Approve all done", "Lock every finished part"),
    ], default='APPROVE')
    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return ob is not None and ob.type == 'MESH' and len(source_of(ob).plegma_parts) > 0

    def execute(self, ctx):
        ob = source_of(ops.active_mesh(ctx))
        if self.action == 'APPROVE_ALL':
            for p in ob.plegma_parts:
                if p.status == 'DONE':
                    p.status = 'APPROVED'
            return {'FINISHED'}
        idx = self.index if self.index >= 0 else ob.plegma_index
        if not 0 <= idx < len(ob.plegma_parts):
            return {'CANCELLED'}
        p = ob.plegma_parts[idx]
        if self.action == 'APPROVE' and p.status == 'DONE':
            p.status = 'APPROVED'
        elif self.action == 'UNLOCK' and p.status == 'APPROVED':
            p.status = 'DONE'
        elif self.action == 'REDO':
            drop_result(p.cache_key)
            p.cache_key = ""
            p.status = 'REDO'
        res = ob.get("plegma_result")
        if res and res in bpy.data.objects:
            for q in bpy.data.objects[res].plegma_parts:
                if q.fs_id == p.fs_id:
                    q.status = p.status
        return {'FINISHED'}


class PLEGMA_OT_clear_cache(bpy.types.Operator):
    """Delete all remesh results stored on disk. Every part goes back to Pending"""
    bl_idname = "plegma.clear_cache"
    bl_label = "Clear cache"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        d = cache_dir()
        n = 0
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
                n += 1
            except OSError:
                pass
        for o in bpy.data.objects:
            for p in o.plegma_parts:
                p.status = 'PENDING'
                p.cache_key = ""
        self.report({'INFO'}, f"Cache cleared ({n} files)")
        return {'FINISHED'}


classes = (PLEGMA_OT_process, PLEGMA_OT_build, PLEGMA_OT_part_action, PLEGMA_OT_clear_cache)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
