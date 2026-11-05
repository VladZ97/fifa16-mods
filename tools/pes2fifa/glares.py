"""Floodlight glare files (data/sceneassets/fx/glares_<id>_<tod>_<n>.lnx).

They are XML particle systems whose emitters sit at the stock stadium's floodlights. In a
converted stadium those points hang in mid-air, so the stadium-light groups are moved onto
the inner edge of the new roof (same emitter count, same angle around the pitch); sun
flares and street lights are left alone.
"""
import re

import numpy as np

GROUP = re.compile(r'(<particleGroup name="([^"]+)">)(.*?)(</particleGroup>)', re.S)
CENTER = re.compile(r'(name="vCenter" value="\{)([^}]+)(\}")')
MOVE_PREFIXES = ("fx_stadium", "fx_vbeam")


def roof_ring(pos, roof_y, bins=180, drop=150.0):
    """Inner roof edge per angle bin -> (bins, 3) points; None when the stadium has no roof."""
    top = pos[pos[:, 1] >= roof_y]
    if len(top) < bins:
        return None
    ang = np.arctan2(top[:, 2], top[:, 0])
    rad = np.hypot(top[:, 0], top[:, 2])
    b = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    ring = np.full((bins, 3), np.nan)
    order = np.argsort(rad)
    first = np.unique(b[order], return_index=True)
    for bi, oi in zip(*first):
        p = top[order[oi]]
        ring[bi] = (p[0], p[1] - drop, p[2])
    ok = np.isfinite(ring[:, 0])
    if ok.sum() < bins * 0.5:
        return None
    idx = np.arange(bins)
    for c in range(3):                                   # fill empty bins around the circle
        ring[:, c] = np.interp(idx, idx[ok], ring[ok, c], period=bins)
    return ring


def relocate(text, ring):
    bins = len(ring)
    moved = 0

    def fix_group(m):
        nonlocal moved
        if not m.group(2).startswith(MOVE_PREFIXES):
            return m.group(0)

        def fix_center(c):
            nonlocal moved
            x, y, z = (float(v) for v in c.group(2).split(","))
            b = int((np.arctan2(z, x) + np.pi) / (2 * np.pi) * bins) % bins
            moved += 1
            return f"{c.group(1)}{ring[b][0]:.3f},{ring[b][1]:.3f},{ring[b][2]:.3f}{c.group(3)}"
        return m.group(1) + CENTER.sub(fix_center, m.group(3)) + m.group(4)

    return GROUP.sub(fix_group, text), moved
