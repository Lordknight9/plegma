# SPDX-License-Identifier: GPL-3.0-or-later
# Runs QuadWild in its own process: a crash in the native library can never take Blender down.
# ctypes structures and call sequence adapted from QRemeshify by ksami (GPL-3.0).
# Usage: python qw_native.py job.json

import json
import locale
import os
import sys
from ctypes import (Structure, POINTER, addressof, byref, cdll, c_bool, c_char_p, c_double,
                    c_float, c_int, c_void_p, create_string_buffer)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


class Parameters(Structure):
    _fields_ = [
        ('remesh', c_bool),
        ('sharpAngle', c_float),
        ('alpha', c_float),
        ('scaleFact', c_float),
        ('hasFeature', c_bool),
        ('hasField', c_bool),
    ]


class FloatVec(Structure):
    """Memory layout of std::vector<float> (MSVC / libstdc++ / libc++: begin, end, capacity)."""
    _fields_ = [("begin", c_void_p), ("end", c_void_p), ("cap", c_void_p)]


def float_vec(values, keep):
    arr = (c_float * len(values))(*values)
    keep.append(arr)
    a = addressof(arr)
    return FloatVec(a, a + 4 * len(values), a + 4 * len(values))


class QRParameters(Structure):
    _fields_ = [
        ("useFlowSolver", c_bool),
        ("flow_config_filename", c_char_p),
        ("satsuma_config_filename", c_char_p),
        ("initialRemeshing", c_bool),
        ("initialRemeshingEdgeFactor", c_double),
        ("reproject", c_bool),
        ("splitConcaves", c_bool),
        ("finalSmoothing", c_bool),
        ("ilpMethod", c_int),
        ("alpha", c_double),
        ("isometry", c_bool),
        ("regularityQuadrilaterals", c_bool),
        ("regularityNonQuadrilaterals", c_bool),
        ("regularityNonQuadrilateralsWeight", c_double),
        ("alignSingularities", c_bool),
        ("alignSingularitiesWeight", c_double),
        ("repeatLosingConstraintsIterations", c_bool),
        ("repeatLosingConstraintsQuads", c_bool),
        ("repeatLosingConstraintsNonQuads", c_bool),
        ("repeatLosingConstraintsAlign", c_bool),
        ("feasibilityFix", c_bool),
        ("hardParityConstraint", c_bool),
        ("timeLimit", c_double),
        ("gapLimit", c_double),
        ("minimumGap", c_double),
        ("callbackTimeLimit", FloatVec),  # std::vector<float> on the C++ side
        ("callbackGapLimit", FloatVec),
        ("chartSmoothingIterations", c_int),
        ("quadrangulationFixedSmoothingIterations", c_int),
        ("quadrangulationNonFixedSmoothingIterations", c_int),
        ("doubletRemoval", c_bool),
        ("resultSmoothingIterations", c_int),
        ("resultSmoothingNRing", c_double),
        ("resultSmoothingLaplacianIterations", c_int),
        ("resultSmoothingLaplacianNRing", c_double),
    ]


_KEEP = []


def _qr_params():
    tl = [3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0]
    gl = [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
    p = QRParameters()
    p.initialRemeshing = True
    p.initialRemeshingEdgeFactor = 1
    p.reproject = True
    p.splitConcaves = False
    p.finalSmoothing = True
    p.doubletRemoval = True
    p.resultSmoothingIterations = 5
    p.resultSmoothingNRing = 3
    p.resultSmoothingLaplacianIterations = 2
    p.resultSmoothingLaplacianNRing = 3
    p.alpha = 0.005
    p.ilpMethod = 1
    p.timeLimit = 200
    p.gapLimit = 0.0
    p.callbackTimeLimit = float_vec(tl, _KEEP)
    p.callbackGapLimit = float_vec(gl, _KEEP)
    p.minimumGap = 0.4
    p.isometry = True
    p.regularityQuadrilaterals = True
    p.regularityNonQuadrilaterals = True
    p.regularityNonQuadrilateralsWeight = 0.9
    p.alignSingularities = True
    p.alignSingularitiesWeight = 0.1
    p.repeatLosingConstraintsIterations = True
    p.repeatLosingConstraintsQuads = False
    p.repeatLosingConstraintsNonQuads = False
    p.repeatLosingConstraintsAlign = True
    p.hardParityConstraint = True
    p.useFlowSolver = True
    p.flow_config_filename = os.path.join(CONFIG_DIR, "main_config", "flow_virtual_simple.json").encode()
    p.satsuma_config_filename = os.path.join(CONFIG_DIR, "satsuma", "default.json").encode()
    p.chartSmoothingIterations = 0
    p.quadrangulationFixedSmoothingIterations = 0
    p.quadrangulationNonFixedSmoothingIterations = 0
    p.feasibilityFix = False
    return p


def main(job_path):
    job = json.load(open(job_path))
    locale.setlocale(locale.LC_ALL, "C")
    d = job["lib_dir"]
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)
    qw = cdll.LoadLibrary(os.path.join(d, job["lib_qw"]))
    qp = cdll.LoadLibrary(os.path.join(d, job["lib_qp"]))
    qw.remeshAndField2.argtypes = [POINTER(Parameters), c_char_p, c_char_p, c_char_p]
    qw.remeshAndField2.restype = None
    qw.trace2.argtypes = [c_char_p]
    qw.trace2.restype = c_bool
    qp.quadPatches.argtypes = [c_char_p, POINTER(QRParameters), c_float, c_int, c_bool]
    qp.quadPatches.restype = c_int

    base = job["base"]
    if job["stage"] == "prepare":
        prm = Parameters(remesh=True, sharpAngle=job["sharp_angle"], alpha=0.01, scaleFact=1,
                         hasFeature=True, hasField=False)
        qw.remeshAndField2(byref(prm), create_string_buffer((base + ".obj").encode()),
                           create_string_buffer((base + "_rem.sharp").encode()),
                           create_string_buffer((base + "_rem.rosy").encode()))
        ok = qw.trace2(create_string_buffer((base + "_rem").encode()))
        sys.exit(0 if ok else 3)
    else:
        params = _qr_params()
        qp.quadPatches((base + "_rem_p0.obj").encode(), byref(params), c_float(job["scale"]), c_int(0), c_bool(True))
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
