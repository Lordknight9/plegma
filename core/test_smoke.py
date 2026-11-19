"""Smoke test for plegma_core: remesh a subdivided icosahedron (sphere), check the result.

Standard library only, so it runs on the CI machine and on any PC.
Usage: python test_smoke.py path/to/plegma_core(.exe)
"""
import math
import os
import struct
import subprocess
import sys
import tempfile


def icosphere(levels=4):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [tuple(c / math.sqrt(sum(x * x for x in v)) for c in v) for v in V]
    for _ in range(levels):
        mid = {}

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = [(V[a][i] + V[b][i]) / 2 for i in range(3)]
                n = math.sqrt(sum(x * x for x in p))
                V.append(tuple(x / n for x in p))
                mid[k] = len(V) - 1
            return mid[k]
        F = [f for a, b, c in F for f in ((a, m(a, b), m(c, a)), (b, m(b, c), m(a, b)), (c, m(c, a), m(b, c)),
                                          (m(a, b), m(b, c), m(c, a)))]
    return V, F


def write_plm(path, V, F):
    with open(path, "wb") as f:
        f.write(b"PLM1" + struct.pack("<ii", len(V), len(F)))
        f.write(struct.pack(f"<{3 * len(V)}f", *[c for v in V for c in v]))
        f.write(struct.pack(f"<{3 * len(F)}i", *[i for t in F for i in t]))
        f.write(struct.pack(f"<{len(F)}i", *([0] * len(F))))


def main(exe):
    V, F = icosphere(5)
    d = tempfile.mkdtemp()
    ref = os.path.join(d, "ref.plm"); out = os.path.join(d, "out.plq"); job = os.path.join(d, "job.txt")
    write_plm(ref, V, F)
    with open(job, "w") as f:
        f.write(f"ref {ref}\ndense {ref}\nout {out}\ntarget 1500\nsharp_angle 40\nfill_all 1\n")
    p = subprocess.run([exe, job], capture_output=True, text=True, timeout=300)
    print(p.stdout, p.stderr)
    if p.returncode != 0:
        sys.exit(f"core exited with {p.returncode}")
    with open(out, "rb") as f:
        assert f.read(4) == b"PLQ1"
        nv, nf = struct.unpack("<ii", f.read(8))
        f.read(12 * nv)
        sizes = struct.unpack(f"<{nf}i", f.read(4 * nf))
    quads = sum(1 for s in sizes if s == 4)
    print(f"faces {nf} quads {quads}")
    if nf < 500 or quads != nf:
        sys.exit("unexpected result")
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
