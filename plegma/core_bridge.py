# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge to plegma_core (C++): one part per process, files in the cache folder.

The process exits after each part, so the operating system takes back all of its memory.
"""

import math
import os
import platform
import subprocess

import numpy as np
import bmesh

from . import guided

BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


def exe_path():
    name = "plegma_core.exe" if platform.system() == "Windows" else "plegma_core"
    p = os.path.join(BIN_DIR, name)
    return p if os.path.isfile(p) else None


def available():
    return exe_path() is not None


def write_mesh(path, bm, label_layer=None):
    """PLM1: nV, nT, float32 V, int32 T, int32 labels (bmesh must be triangulated)."""
    import bpy
    me = bpy.data.meshes.new("plegma_io")
    bm.to_mesh(me)
    try:
        V = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", V)
        T = np.empty(len(me.polygons) * 3, np.int32); me.polygons.foreach_get("vertices", T)
        L = np.zeros(len(me.polygons), np.int32)
        if label_layer is not None and guided.SEG in me.attributes:
            me.attributes[guided.SEG].data.foreach_get("value", L)
    finally:
        bpy.data.meshes.remove(me)
    with open(path, "wb") as f:
        f.write(b"PLM1")
        f.write(np.array([len(V) // 3, len(T) // 3], np.int32).tobytes())
        f.write(V.tobytes()); f.write(T.tobytes()); f.write(L.tobytes())


def read_polys(path):
    with open(path, "rb") as f:
        if f.read(4) != b"PLQ1":
            raise RuntimeError("plegma_core: bad result file")
        nv, nf = np.frombuffer(f.read(8), np.int32)
        V = np.frombuffer(f.read(int(nv) * 12), np.float32).reshape(-1, 3)
        sizes = np.frombuffer(f.read(int(nf) * 4), np.int32)
        idx = np.frombuffer(f.read(int(sizes.sum()) * 4), np.int32)
    return V, sizes, idx


def polys_to_bm(V, sizes, idx):
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in V.tolist()]
    pos = 0
    for n in sizes.tolist():
        try:
            bm.faces.new([vs[i] for i in idx[pos:pos + n].tolist()])
        except ValueError:
            pass
        pos += n
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.normal_update()
    return bm


def remesh_quads(part_bm, target, workdir, tag, log=print, sharp_angle=40.0, fill_all=True,
                 seam_axes=(), seam_eps=0.0, mem_limit_mb=2048, timeout=600):
    """Consumes part_bm. Returns the quad bmesh, or raises RuntimeError."""
    exe = exe_path()
    if exe is None:
        raise RuntimeError("plegma_core is missing")
    os.makedirs(workdir, exist_ok=True)
    base = os.path.join(workdir, tag)
    ref_path, dense_path, out_path, job_path = base + "_ref.plm", base + "_dense.plm", base + "_out.plq", base + "_job.txt"

    bmesh.ops.triangulate(part_bm, faces=[f for f in part_bm.faces if len(f.verts) > 3])
    lay = part_bm.faces.layers.int.get(guided.SEG)
    write_mesh(ref_path, part_bm, lay)
    q0 = max(target / 4.0 * 1.2, 8)
    dense = guided._decimate(part_bm, int(q0 * 12))      # C-speed decimation inside Blender
    bmesh.ops.triangulate(dense, faces=[f for f in dense.faces if len(f.verts) > 3])
    write_mesh(dense_path, dense, dense.faces.layers.int.get(guided.SEG))
    dense.free()

    with open(job_path, "w") as f:
        f.write(f"ref {ref_path}\ndense {dense_path}\nout {out_path}\n")
        f.write(f"target {float(target)}\nsharp_angle {float(sharp_angle)}\nfill_all {int(bool(fill_all))}\n")
        f.write(f"seam_axes {' '.join(str(a) for a in seam_axes)}\nseam_eps {float(seam_eps)}\n")
        f.write(f"mem_limit_mb {float(mem_limit_mb)}\nverbose {int(os.environ.get('PLEGMA_VERBOSE', '1'))}\n")
    if os.path.exists(out_path):
        os.remove(out_path)
    flags = 0x08000000 if platform.system() == "Windows" else 0
    import time
    t0 = time.time()
    p = subprocess.run([exe, job_path], capture_output=True, text=True, timeout=timeout, creationflags=flags)
    log(f"  core: {time.time() - t0:.1f}s")
    for line in (p.stdout or "").strip().splitlines():
        log("  core: " + line)
    for path in (ref_path, dense_path, job_path):
        try:
            os.remove(path)
        except OSError:
            pass
    if p.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"plegma_core failed ({p.returncode}): {(p.stderr or '').strip()[-200:]}")
    V, sizes, idx = read_polys(out_path)
    os.remove(out_path)
    return polys_to_bm(V, sizes, idx)
