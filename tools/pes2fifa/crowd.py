# SPDX-License-Identifier: GPL-2.0-or-later
"""PES audience areas (audiarea.bin) -> FIFA 16 crowd placement (crowd_<id>_<tod>.dat).

PES describes the stands as quads with a seat-colour map; FIFA wants one 32-byte record per
seat. Seats are generated row by row on each quad, snapped to the real steps of the
converted stand geometry, coloured from the PES seat map, and given the supporter-type
bytes of the nearest seat in the stock stadium (keeps the game's own home/away layout).

FIFA record: pos 3f, facing angle f (degrees, atan2(-fz, fx)), seat RGB 3B, then
type a/b, tier, attendance, 0, 0, 0, flag, variant, 0 0 0 0.
"""
import os
import struct

import numpy as np
from PIL import Image

import fpk
import raster

HEADER = struct.Struct("<4sHI")
REC = np.dtype([("pos", "<f4", 3), ("angle", "<f4"), ("rgb", "u1", 3), ("type", "u1", 2), ("tier", "u1"),
                ("attend", "u1"), ("z0", "u1", 3), ("flag", "u1"), ("variant", "u1"), ("z1", "u1", 4)])
assert REC.itemsize == 32


def read_dat(data):
    magic, ver, count = HEADER.unpack_from(data, 0)
    if magic != b"CRWD":
        raise ValueError("not a CRWD file")
    return np.frombuffer(data, dtype=REC, count=count, offset=HEADER.size).copy(), ver


def write_dat(recs, ver=261):
    return HEADER.pack(b"CRWD", ver, len(recs)) + np.ascontiguousarray(recs, dtype=REC).tobytes()


def read_audiarea(data):
    """-> list of dicts: side, level, verts (4,3) [bottom-left, top-left, top-right, bottom-right]
    in PES metres, uvs (4,2), kind (PES home/away + stance value)."""
    quads = []
    for p in struct.unpack_from("<12I", data, 8):
        if not p:
            continue
        side, level, groups, start = struct.unpack_from("<BBHI", data, p)
        q = start
        for _ in range(groups):
            n = struct.unpack_from("<I", data, q + 24)[0]
            q += 28
            for _i in range(n):
                _idx, rows, kind, _k2 = struct.unpack_from("<I3f", data, q)
                v = np.array(struct.unpack_from("<12f", data, q + 16), np.float32).reshape(4, 3)
                uv = np.array(struct.unpack_from("<8f", data, q + 64), np.float32).reshape(4, 2)
                q += 96
                quads.append(dict(side=side, level=level, verts=v, uvs=uv, kind=kind, rows=rows))
    return quads


def find_steps(profile, step_cm, min_tread=35.0, min_rise=7.0):
    """profile: heights sampled every step_cm along the slope (uphill). -> list of (start, end)
    sample indices of treads, or [] when the stand has no recognisable steps."""
    h = np.asarray(profile, np.float64)
    ok = np.isfinite(h)
    if ok.sum() < len(h) * 0.6:
        return []
    h = np.interp(np.arange(len(h)), np.nonzero(ok)[0], h[ok])
    rise = np.diff(h)
    # a riser is a short stretch where the height climbs steeply
    steep = rise > step_cm * 0.9          # > ~42 degrees
    treads, start = [], 0
    i = 0
    n = len(rise)
    while i < n:
        if steep[i]:
            j = i
            while j < n and steep[j]:
                j += 1
            if h[j] - h[i] >= min_rise:
                treads.append((start, i))
                start = j
            i = j
        else:
            i += 1
    treads.append((start, n))
    treads = [(a, b) for a, b in treads if (b - a) * step_cm >= min_tread]
    if len(treads) < 3:
        return []
    depth = np.array([(b - a) * step_cm for a, b in treads])
    if np.median(depth) > 160:            # terraces that deep are concourses, not seat rows
        return []
    return treads


class HeightField:
    """Highest stand surface below a sloped cap, rasterised top-down for one seating quad."""

    def __init__(self, pos, tris, tri_lo, tri_hi, quad, res=4.0, margin=60.0, cap=90.0, floor=260.0):
        lo, hi = quad.min(0), quad.max(0)
        m = ((tri_hi[:, 0] >= lo[0] - margin) & (tri_lo[:, 0] <= hi[0] + margin) &
             (tri_hi[:, 2] >= lo[2] - margin) & (tri_lo[:, 2] <= hi[2] + margin) &
             (tri_hi[:, 1] >= lo[1] - floor) & (tri_lo[:, 1] <= hi[1] + cap))
        t = tris[m]
        # plane through the quad: y = a*x + b*z + c (least squares over its 4 corners)
        A = np.c_[quad[:, 0], quad[:, 2], np.ones(4)]
        self.plane = np.linalg.lstsq(A, quad[:, 1], rcond=None)[0]
        if len(t):
            p = pos[t]                                   # (n,3,3)
            py = p[:, :, 0] * self.plane[0] + p[:, :, 2] * self.plane[1] + self.plane[2]
            keep = (p[:, :, 1] <= py + cap).all(1) & (p[:, :, 1] >= py - floor).any(1)
            t = t[keep]
        self.x0, self.z0, self.res = lo[0] - margin, lo[2] - margin, res
        self.w = int(np.ceil((hi[0] - lo[0] + 2 * margin) / res)) + 1
        self.h = int(np.ceil((hi[2] - lo[2] + 2 * margin) / res)) + 1
        if len(t):
            used, inv = np.unique(t, return_inverse=True)
            v = pos[used]
            sv = np.stack([(v[:, 0] - self.x0) / res, (v[:, 2] - self.z0) / res, v[:, 1]], 1)
            self.map, _ = raster.raster(sv, inv.reshape(-1, 3), self.w, self.h, want_ids=False)
        else:
            self.map = np.full((self.h, self.w), -np.inf, np.float32)

    def plane_y(self, x, z):
        return x * self.plane[0] + z * self.plane[1] + self.plane[2]

    def height(self, x, z):
        """Surface height at points; NaN where nothing was hit."""
        ix = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.w - 1)
        iz = np.clip(((np.asarray(z) - self.z0) / self.res).astype(int), 0, self.h - 1)
        y = self.map[iz, ix].astype(np.float64)
        return np.where(np.isfinite(y), y, np.nan)


def seat_colours(image, uv):
    h, w = image.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
    y = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
    return image[y, x, :3]


def generate(quads, geom_pos, geom_tris, seat_image, seat_pitch=55.0, row_pitch=80.0, log=print):
    """quads: FIFA-space dicts (verts (4,3) cm). -> dict of per-seat arrays."""
    p = geom_pos[geom_tris]
    tri_lo, tri_hi = p.min(1), p.max(1)
    out = dict(pos=[], face=[], uv=[], quad=[])
    stepped = flat = 0
    for qi, q in enumerate(quads):
        v = q["verts"].astype(np.float64)
        b0, t0, t1, b1 = v
        bottom_mid, top_mid = (b0 + b1) / 2, (t0 + t1) / 2
        slope = top_mid - bottom_mid
        horiz = np.array([slope[0], 0.0, slope[2]])
        depth = np.linalg.norm(horiz)
        if depth < 30 or min(np.linalg.norm(b1 - b0), np.linalg.norm(t1 - t0)) < 20:
            continue
        hf = HeightField(geom_pos, geom_tris, tri_lo, tri_hi, v)
        step = 2.0
        n = max(2, int(depth / step))
        s = (np.arange(n) + 0.5) / n
        # steps are found on three lines across the quad and the clearest one is used
        best = []
        for tline in (0.5, 0.3, 0.7):
            lo_pt = b0 + (b1 - b0) * tline
            hi_pt = t0 + (t1 - t0) * tline
            line = lo_pt + np.outer(s, hi_pt - lo_pt)
            treads = find_steps(hf.height(line[:, 0], line[:, 2]), depth / n)
            if len(treads) > len(best):
                best = treads
        if best:
            stepped += 1
            row_s = [s[min(n - 1, int(a + (b - a) * 0.6))] for a, b in best]
        else:
            flat += 1
            k = max(1, int(round(depth / row_pitch)))
            row_s = [(i + 0.5) / k for i in range(k)]
        face = -horiz / depth                       # downhill, toward the pitch
        for rs in row_s:
            left = b0 + (t0 - b0) * rs
            right = b1 + (t1 - b1) * rs
            width = np.linalg.norm((right - left)[[0, 2]])
            k = int(width / seat_pitch)
            if k < 1:
                continue
            t = (np.arange(k) + 0.5) / k
            pts = left + np.outer(t, right - left)
            y = hf.height(pts[:, 0], pts[:, 2])
            plane = hf.plane_y(pts[:, 0], pts[:, 2])
            # seats that miss the step (aisle gap, hole) fall back to the row's median height
            med = np.nanmedian(y) if np.isfinite(y).any() else np.nan
            bad = ~np.isfinite(y) | (np.abs(y - med) > 35)
            y = np.where(bad, med if np.isfinite(med) else plane, y)
            pts[:, 1] = y
            uv_l = q["uvs"][0] + (q["uvs"][1] - q["uvs"][0]) * rs
            uv_r = q["uvs"][3] + (q["uvs"][2] - q["uvs"][3]) * rs
            out["pos"].append(pts)
            out["face"].append(np.tile(face, (k, 1)))
            out["uv"].append(uv_l + np.outer(t, uv_r - uv_l))
            out["quad"].append(np.full(k, qi))
    seats = {k: np.concatenate(v) for k, v in out.items()}
    seats["rgb"] = seat_colours(seat_image, seats["uv"]) if seat_image is not None else np.full((len(seats["pos"]), 3), 150, np.uint8)
    log(f"crowd: {len(seats['pos'])} seats from {len(quads)} areas ({stepped} snapped to steps, {flat} on the slope)")
    return seats


def assign_tiers(quads, seats, max_tiers=3, gap=250.0):
    """Tier per seat from the height bands of the seating areas."""
    spans = sorted((q["verts"][:, 1].min(), q["verts"][:, 1].max()) for q in quads)
    bands = []
    for lo, hi in spans:
        if bands and lo <= bands[-1][1] + gap:
            bands[-1][1] = max(bands[-1][1], hi)
        else:
            bands.append([lo, hi])
    while len(bands) > max_tiers:                        # merge the two closest bands
        i = int(np.argmin([bands[k + 1][0] - bands[k][1] for k in range(len(bands) - 1)]))
        bands[i][1] = bands[i + 1][1]
        del bands[i + 1]
    y = seats["pos"][:, 1]
    tier = np.ones(len(y), np.uint8)
    for i, (lo, _hi) in enumerate(bands):
        tier[y >= lo - 1] = i + 1
    return tier, bands


def build_records(seats, tier, stock_recs, seed=6):
    from scipy.spatial import cKDTree
    n = len(seats["pos"])
    recs = np.zeros(n, REC)
    recs["pos"] = seats["pos"]
    f = seats["face"]
    recs["angle"] = np.degrees(np.arctan2(-f[:, 2], f[:, 0]))
    recs["rgb"] = seats["rgb"]
    recs["tier"] = tier
    recs["attend"] = 255
    rng = np.random.default_rng(seed)
    variants = np.unique(stock_recs["variant"]) if len(stock_recs) else np.array([255, 128], np.uint8)
    common = [v for v in (255, 128) if v in variants] or [int(variants[0])]
    recs["variant"] = rng.choice(common, n)
    if len(stock_recs):
        # supporter type follows the stock stadium's layout: nearest stock seat around the bowl
        ang_s = np.arctan2(stock_recs["pos"][:, 2], stock_recs["pos"][:, 0])
        ang_n = np.arctan2(seats["pos"][:, 2], seats["pos"][:, 0])
        hs = stock_recs["pos"][:, 1] / max(1.0, stock_recs["pos"][:, 1].max())
        hn = seats["pos"][:, 1] / max(1.0, seats["pos"][:, 1].max())
        key_s = np.c_[np.cos(ang_s), np.sin(ang_s), hs * 0.6]
        key_n = np.c_[np.cos(ang_n), np.sin(ang_n), hn * 0.6]
        _d, idx = cKDTree(key_s).query(key_n)
        recs["type"] = stock_recs["type"][idx]
        recs["flag"] = stock_recs["flag"][idx]
    else:
        recs["type"] = (128, 128)
    return recs


def load_pes(pes_root, st):
    """-> (quads in PES space, seat colour image or None)"""
    base = os.path.join(pes_root, "Asset", "model", "bg", st, "audi")
    pk = os.path.join(base, "#Win", f"audiarea_{st}.fpk")
    if not os.path.exists(pk):
        return [], None
    data = next((d for n, d in fpk.read(pk) if n.lower().endswith("audiarea.bin")), None)
    if data is None:
        return [], None
    image = None
    png = os.path.join(base, "sourceimages", "au_seat.png")
    if os.path.exists(png):
        image = np.asarray(Image.open(png).convert("RGB"))
    return read_audiarea(data), image
