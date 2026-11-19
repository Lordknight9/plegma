# SPDX-License-Identifier: GPL-3.0-or-later
# QuadWild (Bi-MDF) bridge: download on demand, run in a separate process (qw_native.py).
# Native libraries come from the QRemeshify release by ksami (GPL-3.0) - https://github.com/ksami/QRemeshify

import os
import io
import sys
import json
import zipfile
import platform
import subprocess
import urllib.request

import bpy

RELEASE_API = "https://api.github.com/repos/ksami/QRemeshify/releases/latest"
NATIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qw_native.py")


def _lib_names():
    s = platform.system()
    if s == "Windows":
        return "lib_quadwild.dll", "lib_quadpatches.dll"
    if s == "Darwin":
        return "liblib_quadwild.dylib", "liblib_quadpatches.dylib"
    return "liblib_quadwild.so", "liblib_quadpatches.so"


def engine_dir(create=False):
    try:
        return bpy.utils.extension_path_user(__package__.rsplit(".", 1)[0], path="quadwild", create=create)
    except Exception:
        d = os.path.join(bpy.utils.user_resource('SCRIPTS'), "plegma_quadwild")
        if create:
            os.makedirs(d, exist_ok=True)
        return d


def is_installed():
    d = engine_dir()
    a, b = _lib_names()
    return os.path.isfile(os.path.join(d, a)) and os.path.isfile(os.path.join(d, b))


def install():
    """Download the QRemeshify release for this OS and keep only its native libraries."""
    if not bpy.app.online_access:
        raise RuntimeError("Online access is off (Preferences > System > Network > Allow Online Access)")
    osname = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    req = urllib.request.Request(RELEASE_API, headers={"User-Agent": "plegma-addon"})
    rel = json.loads(urllib.request.urlopen(req, timeout=30).read())
    asset = next((a for a in rel["assets"] if osname in a["name"].lower() and a["name"].endswith(".zip")), None)
    if asset is None:
        raise RuntimeError("No QuadWild build found for this operating system")
    data = urllib.request.urlopen(urllib.request.Request(asset["browser_download_url"],
                                  headers={"User-Agent": "plegma-addon"}), timeout=120).read()
    dest = engine_dir(create=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base.lower().endswith((".dll", ".so", ".dylib")) or ".so." in base:
                with open(os.path.join(dest, base), "wb") as f:
                    f.write(z.read(name))
                n += 1
    if not is_installed():
        raise RuntimeError("The download finished but the QuadWild libraries are missing")
    return rel.get("tag_name", "?"), n


def _write_obj(path, verts, tris):
    with open(path, "w") as f:
        f.write("# plegma\n")
        f.write("".join(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in verts))
        f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in tris))


def _write_sharp(path, sharp):
    with open(path, "w") as f:
        f.write(f"{len(sharp)}\n")
        f.write("".join(f"{cv},{fi},{ei}\n" for cv, fi, ei in sharp))


def _run(job, timeout):
    path = job["base"] + f"_{job['stage']}.json"
    with open(path, "w") as f:
        json.dump(job, f)
    flags = 0x08000000 if platform.system() == "Windows" else 0  # CREATE_NO_WINDOW
    p = subprocess.run([sys.executable, "-I", NATIVE, path], capture_output=True, text=True,
                       timeout=timeout, creationflags=flags)
    return p.returncode, (p.stdout or "")[-2000:] + (p.stderr or "")[-2000:]


def _read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append(tuple(float(x.replace(",", ".")) for x in t[1:4]))
            elif t[0] == "f":
                faces.append(tuple(int(s.split("/")[0]) - 1 for s in t[1:]))
    return verts, faces


def remesh(verts, tris, sharp, workdir, tag, target_faces, sharp_angle, log=print, timeout=600):
    """Run QuadWild on a triangle soup. Returns (verts, faces) of the quad mesh.

    The density is steered toward target_faces by re-running the cheap last stage.
    """
    base = os.path.join(workdir, f"plegma_{tag}")
    out_s = base + "_rem_p0_0_quadrangulation_smooth.obj"
    out_r = base + "_rem_p0_0_quadrangulation.obj"
    _write_obj(base + ".obj", verts, tris)
    _write_sharp(base + "_rem.sharp", sharp)
    a, b = _lib_names()
    job = {"lib_dir": engine_dir(), "lib_qw": a, "lib_qp": b, "base": base, "sharp_angle": float(sharp_angle)}
    code, out = _run(dict(job, stage="prepare"), timeout)
    if code != 0 or not os.path.exists(base + "_rem_p0.obj"):
        raise RuntimeError(f"QuadWild (prepare) failed, code {code}")

    scale = 1.0
    best = None
    for attempt in range(3):
        for p in (out_s, out_r):
            if os.path.exists(p):
                os.remove(p)
        code, out = _run(dict(job, stage="quads", scale=scale), timeout)
        path = out_s if os.path.exists(out_s) else out_r
        if not os.path.exists(path):
            if best is not None:
                break
            raise RuntimeError(f"QuadWild (quads) failed, code {code}")
        v, f = _read_obj(path)
        n = max(len(f), 1)
        err = abs(n - target_faces) / max(target_faces, 1)
        if best is None or err < best[0]:
            best = (err, v, f)
        log(f"  QuadWild {tag}: scale {scale:.3f} -> {n} quads (target {target_faces})")
        if err < 0.2:
            break
        scale *= (n / max(target_faces, 1)) ** 0.5
        scale = min(max(scale, 0.05), 50.0)
    return best[1], best[2]
