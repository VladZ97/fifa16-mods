"""ctypes wrapper for raster.c; compiles it on first use (needs the system C compiler)."""
import ctypes
import os
import platform
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    src = os.path.join(_HERE, "raster.c")
    out = os.path.join(_HERE, f"_raster_{platform.machine()}.dylib")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O3", "-shared", "-fPIC", "-arch", platform.machine(), "-o", out, src])
    _lib = ctypes.CDLL(out)
    _lib.raster.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                            ctypes.c_int]
    _lib.raster.restype = None
    return _lib


def raster(verts, tris, width, height, persp=False, near=1.0, cull=0, ids=None, want_ids=True):
    """-> (key float32 (H,W), id int32 (H,W) or None). Empty pixels: key = -inf, id = -1."""
    lib = _load()
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.uint32)
    key = np.full((height, width), -np.inf, dtype=np.float32)
    idbuf = np.full((height, width), -1, dtype=np.int32) if want_ids else None
    ids_arr = np.ascontiguousarray(ids, dtype=np.int32) if ids is not None else None
    lib.raster(verts.ctypes.data, tris.ctypes.data, ids_arr.ctypes.data if ids_arr is not None else None,
               len(tris), width, height, key.ctypes.data, idbuf.ctypes.data if want_ids else None,
               1 if persp else 0, near, cull)
    return key, idbuf
