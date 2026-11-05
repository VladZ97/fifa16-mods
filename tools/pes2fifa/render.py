"""Offline preview renderer for FIFA 16 stadium RX3 files (model + texture file).

Draws what the converter wrote, not what it meant to write: positions, t0/t1 UVs, material
texture bindings and texel data are all read back from the output files.

  render.py <stadium.rx3> <textures.rx3> <out.png> [--view tv|top|goal|corner|high] [--size 1600x900]
            [--groups A,B] [--skip A,B] [--no-light]
"""
import sys

import numpy as np
from PIL import Image

import ftex
import raster
import rx3

VIEWS = {
    # name: (eye, target, vertical fov degrees) in FIFA centimetres; main camera sits on -Z
    "tv": ((0, 2600, -7200), (0, 0, 600), 50),
    "high": ((0, 9000, -14000), (0, 0, 0), 55),
    "aerial": ((-30000, 20000, -42000), (0, 2500, 0), 38),
    "aerial2": ((34000, 14000, 30000), (0, 2500, 0), 38),
    "street": ((-26000, 300, 9000), (0, 2500, 0), 55),
    "goal": ((-9000, 1800, 0), (0, 300, 0), 60),
    "goal2": ((9000, 1800, 0), (0, 300, 0), 60),
    "corner": ((-7500, 2500, -6000), (1000, 500, 1000), 60),
    "far": ((0, 2600, 7200), (0, 0, -600), 50),
    "pitch": ((0, 180, 0), (0, 900, 6000), 75),
    "pitch2": ((0, 180, 0), (0, 900, -6000), 75),
    "pitch3": ((0, 180, 0), (9000, 900, 0), 75),
    "pitch4": ((0, 180, 0), (-9000, 900, 0), 75),
}


def decode_attr(mesh, tag):
    e = mesh.element(tag)
    if e is None:
        return None
    off, typ = e
    a = np.frombuffer(mesh.vdata, dtype=np.uint8).reshape(-1, mesh.stride)
    if typ == "3f32":
        return a[:, off:off + 12].copy().view("<f4")
    if typ == "4f16":
        return a[:, off:off + 8].copy().view("<f2").astype(np.float32)[:, :3]
    if typ == "2f16":
        return a[:, off:off + 4].copy().view("<f2").astype(np.float32)
    if typ == "2f32":
        return a[:, off:off + 8].copy().view("<f4")
    if typ == "3s10n":
        v = a[:, off:off + 4].copy().view("<u4")[:, 0]
        out = np.empty((len(v), 3), np.float32)
        for i in range(3):
            c = ((v >> (10 * i)) & 0x3ff).astype(np.int32)
            out[:, i] = np.where(c >= 512, c - 1024, c) / 511.0
        return out
    if typ == "4u8n":
        return a[:, off:off + 4].astype(np.float32) / 255.0
    return None


class TexCache:
    def __init__(self, texfile, max_dim=512):
        self.tex = {t.name: t for t in texfile.textures}
        self.cache = {}
        self.max_dim = max_dim

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        t = self.tex.get(name)
        img = None
        if t is not None and t.fmt in (rx3.TEX_DXT1, rx3.TEX_DXT5):
            lvl = 0
            while lvl + 1 < len(t.mips) and max(t.width >> lvl, t.height >> lvl) > self.max_dim:
                lvl += 1
            w, h = max(1, t.width >> lvl), max(1, t.height >> lvl)
            try:
                img = ftex.decode_bc(t.mips[lvl][2], w, h, b"DXT1" if t.fmt == rx3.TEX_DXT1 else b"DXT5")
            except Exception:
                img = None
        self.cache[name] = img
        return img


def look_at(eye, target, fov_deg, width, height):
    eye = np.asarray(eye, np.float64)
    f = np.asarray(target, np.float64) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fx = (height / 2) / np.tan(np.radians(fov_deg) / 2)
    return eye, r, u, f, fx


def gather(stadium, groups=None, skip=()):
    """-> verts (N,3), uv0 (N,2), uv1 (N,2), normals (N,3), tris (M,3), tri material index, materials list"""
    P, U0, U1, NR, T, TM = [], [], [], [], [], []
    base = 0
    for g in stadium.groups:
        if groups and g.name not in groups:
            continue
        if g.name in skip:
            continue
        for mi, ma, _a, _b in g.entries:
            m = stadium.meshes[mi]
            if m.prim != 4 or len(m.indices) < 3:
                continue
            p = m.positions()
            n = len(p)
            uv0 = decode_attr(m, "t0")
            uv1 = decode_attr(m, "t1")
            nr = decode_attr(m, "n0")
            P.append(p)
            U0.append(uv0 if uv0 is not None else np.zeros((n, 2), np.float32))
            U1.append(uv1 if uv1 is not None else np.full((n, 2), -1.0, np.float32))
            NR.append(nr if nr is not None else np.zeros((n, 3), np.float32))
            t = m.indices[:len(m.indices) // 3 * 3].reshape(-1, 3).astype(np.uint32) + base
            T.append(t)
            TM.append(np.full(len(t), ma, np.int32))
            base += n
    cat = np.concatenate
    return cat(P), cat(U0), cat(U1), cat(NR), cat(T), cat(TM)


def sample(img, uv):
    h, w = img.shape[:2]
    x = np.floor((uv[:, 0] % 1.0) * w).astype(np.int64) % w
    y = np.floor((uv[:, 1] % 1.0) * h).astype(np.int64) % h
    return img[y, x]


def seat_geometry(recs, size=38.0, back=45.0):
    """Crowd records -> (verts, tris, per-triangle RGB): a seat pan and a backrest per seat."""
    n = len(recs)
    p = recs["pos"].astype(np.float32)
    a = np.radians(recs["angle"])
    f = np.stack([np.cos(a), np.zeros(n), -np.sin(a)], 1).astype(np.float32)      # facing direction
    s = np.stack([-f[:, 2], np.zeros(n), f[:, 0]], 1)                              # sideways
    up = np.array([0, 1, 0], np.float32)
    h = size / 2
    seat = p + up * 42
    c = [seat - s * h - f * h, seat + s * h - f * h, seat + s * h + f * h, seat - s * h + f * h,
         seat - s * h - f * h, seat + s * h - f * h, seat + s * h - f * h + up * back, seat - s * h - f * h + up * back]
    v = np.stack(c, 1).reshape(-1, 3)
    base = (np.arange(n) * 8)[:, None]
    t = (base + np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7])).reshape(-1, 3).astype(np.uint32)
    col = np.repeat(recs["rgb"].astype(np.float32), 4, axis=0)
    return v, t, col


def render(stadium, texfile, view="tv", size=(1600, 900), groups=None, skip=("Sky",), light=True,
           eye=None, target=None, fov=None, crowd_recs=None):
    W, H = size
    if view in VIEWS:
        e, t, fv = VIEWS[view]
    else:
        e, t, fv = VIEWS["tv"]
    eye = e if eye is None else eye
    target = t if target is None else target
    fov = fv if fov is None else fov
    P, U0, U1, NR, T, TM = gather(stadium, groups, skip)
    flat_col = None
    if crowd_recs is not None and len(crowd_recs):
        sv, stri, flat_col = seat_geometry(crowd_recs)
        n0 = len(P)
        P = np.concatenate([P, sv])
        U0 = np.concatenate([U0, np.zeros((len(sv), 2), np.float32)])
        U1 = np.concatenate([U1, np.zeros((len(sv), 2), np.float32)])
        NR = np.concatenate([NR, np.tile(np.array([0, 1, 0], np.float32), (len(sv), 1))])
        TM = np.concatenate([TM, -1 - np.arange(len(stri), dtype=np.int32)])
        T = np.concatenate([T, stri + n0])
    out = np.zeros((H, W, 3), np.uint8)
    out[:] = (40, 60, 90)
    if view == "top":
        # orthographic from above, +X right, -Z (main camera side) at the bottom of the picture
        span = 30000.0
        s = W / span
        v = np.stack([W / 2 + P[:, 0] * s, H / 2 + P[:, 2] * s, P[:, 1]], 1).astype(np.float32)
        key, ids = raster.raster(v, T, W, H, persp=False)
        ys, xs = np.nonzero(ids >= 0)
        tid = ids[ys, xs]
        a, b, c = v[T[tid, 0]], v[T[tid, 1]], v[T[tid, 2]]
        px, py = xs + 0.5, ys + 0.5
        den = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        den = np.where(den == 0, 1, den)
        w0 = ((b[:, 0] - px) * (c[:, 1] - py) - (b[:, 1] - py) * (c[:, 0] - px)) / den
        w1 = ((c[:, 0] - px) * (a[:, 1] - py) - (c[:, 1] - py) * (a[:, 0] - px)) / den
        w2 = 1 - w0 - w1
    else:
        eye, r, u, f, fx = look_at(eye, target, fov, W, H)
        d = P.astype(np.float64) - eye
        x, y, z = d @ r, d @ u, d @ f
        v = np.stack([(W / 2) * z + fx * x, (H / 2) * z - fx * y, z], 1).astype(np.float32)
        key, ids = raster.raster(v, T, W, H, persp=True, near=20.0)
        ys, xs = np.nonzero(ids >= 0)
        tid = ids[ys, xs]
        # barycentrics from the ray / triangle-plane intersection in world space
        dirs = (np.outer((xs + 0.5 - W / 2) / fx, r) - np.outer((ys + 0.5 - H / 2) / fx, u) + f)
        A, B, C = (P[T[tid, k]].astype(np.float64) for k in range(3))
        e1, e2 = B - A, C - A
        pv = np.cross(dirs, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        tv = eye - A
        w1 = np.einsum("ij,ij->i", tv, pv) / det
        qv = np.cross(tv, e1)
        w2 = np.einsum("ij,ij->i", dirs, qv) / det
        w0 = 1 - w1 - w2
    W3 = np.stack([w0, w1, w2], 1).astype(np.float32)
    uv0 = np.einsum("ik,ikj->ij", W3, U0[T[tid]])
    uv1 = np.einsum("ik,ikj->ij", W3, U1[T[tid]])
    nrm = np.einsum("ik,ikj->ij", W3, NR[T[tid]])
    cache = TexCache(texfile)
    col = np.full((len(tid), 3), 128.0, np.float32)
    mats = TM[tid]
    if flat_col is not None:
        fsel = np.nonzero(mats < 0)[0]
        col[fsel] = flat_col[-1 - mats[fsel]]
    for ma in np.unique(mats[mats >= 0]):
        sel = np.nonzero(mats == ma)[0]
        shader, slots = stadium.materials[ma]
        sl = dict(slots)
        img = cache.get(sl.get("diffuseTexture", ""))
        c = sample(img, uv0[sel])[:, :3].astype(np.float32) if img is not None else np.full((len(sel), 3), 200.0, np.float32)
        if light:
            lm = cache.get(sl.get("ambientTexture", ""))
            if lm is not None:
                c = c * (sample(lm, uv1[sel])[:, :3].astype(np.float32) / 128.0)
        col[sel] = c
    if not light:
        # simple headlight so unlit previews still show shape
        ln = np.linalg.norm(nrm, axis=1)
        shade = np.where(ln > 0, 0.55 + 0.45 * np.abs(nrm[:, 1]) / np.maximum(ln, 1e-6), 1.0)
        col *= shade[:, None]
    out[ys, xs] = np.clip(col, 0, 255).astype(np.uint8)
    return out


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    opts = {"view": "tv", "size": "1600x900", "groups": None, "skip": "Sky", "light": True}
    i = 3
    while i < len(a):
        if a[i] == "--no-light":
            opts["light"] = False
            i += 1
        else:
            opts[a[i][2:]] = a[i + 1]
            i += 2
    st = rx3.Stadium.parse(open(a[0], "rb").read())
    tx = rx3.TextureFile.parse(open(a[1], "rb").read())
    w, h = map(int, opts["size"].split("x"))
    groups = set(opts["groups"].split(",")) if opts["groups"] else None
    img = render(st, tx, opts["view"], (w, h), groups, tuple(opts["skip"].split(",")), opts["light"])
    Image.fromarray(img).save(a[2])
    print("wrote", a[2])


if __name__ == "__main__":
    main()
