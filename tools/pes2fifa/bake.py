# SPDX-License-Identifier: GPL-2.0-or-later
"""Per-vertex sky lighting bake for converted stadiums.

PES models carry no lightmaps; FIFA's stadium shaders multiply the diffuse texture by an
"ambientTexture" looked up through the second UV set. Instead of unwrapping 700k triangles
into a lightmap atlas, the light level is baked per vertex and stored in t1.u, and the
ambient texture is a 256x4 grey ramp: the lookup then interpolates like vertex lighting.

Sky visibility is measured with orthographic depth maps from directions spread over the
upper hemisphere (shadow-map style), which also captures the roof's shade over the stands.
"""
import numpy as np

import raster
import rx3
import textures

RAMP_W = 256
RAMP_PAD = 8                       # texels kept clear at both ends so wrap filtering cannot bleed


def hemisphere(n, seed=3):
    """n roughly even directions over the upper hemisphere (y up), biased away from the horizon."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    y = 0.08 + 0.92 * (i / n)                       # elevation sine, skipping the lowest 5 degrees
    phi = i * 2.399963229728653 + rng.uniform(0, 0.3)
    r = np.sqrt(np.maximum(0.0, 1 - y * y))
    return np.stack([r * np.cos(phi), y, r * np.sin(phi)], 1)


def sky_visibility(pos, nrm, tris, occ_pos=None, occ_tris=None, dirs=48, res=4096, bias=45.0, log=print):
    """-> (N,) irradiance factor in 0..1 for vertices (pos, nrm); occluders = the same mesh plus
    optional extra geometry. 1 = flat ground under an open sky."""
    if occ_pos is not None and len(occ_pos):
        all_pos = np.concatenate([pos, occ_pos])
        all_tris = np.concatenate([tris, occ_tris + len(pos)])
    else:
        all_pos, all_tris = pos, tris
    centre = (all_pos.min(0) + all_pos.max(0)) / 2
    radius = float(np.linalg.norm(all_pos - centre, axis=1).max()) * 1.01
    scale = res / (2 * radius)
    rel = (all_pos - centre).astype(np.float64)
    n = len(pos)
    lit = np.zeros(n)
    norm = 0.0
    D = hemisphere(dirs)
    for k, d in enumerate(D):
        up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
        u = np.cross(up, d)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        sx = rel @ u * scale + res / 2
        sy = rel @ v * scale + res / 2
        depth = rel @ d                                   # larger = closer to the sky
        key, _ = raster.raster(np.stack([sx, sy, depth], 1), all_tris, res, res, want_ids=False)
        # receiver: vertex pushed a little along its normal to stay clear of its own surface
        rp = rel[:n] + nrm * 12.0
        px = np.clip((rp @ u * scale + res / 2).astype(int), 0, res - 1)
        py = np.clip((rp @ v * scale + res / 2).astype(int), 0, res - 1)
        vis = (rp @ d) >= key[py, px] - bias
        cos = np.clip(nrm @ d, 0.0, None)
        lit += vis * cos
        norm += d[1]                                      # what an upward-facing open vertex collects
    return np.clip(lit / max(norm, 1e-6), 0.0, 1.0)


def to_u(level):
    """Light level 0..1 -> t1.u on the ramp texture."""
    return (RAMP_PAD + np.clip(level, 0, 1) * (RAMP_W - 2 * RAMP_PAD - 1) + 0.5) / RAMP_W


def ramp_texture(name, lo, hi, tint=(1.0, 1.0, 1.0)):
    """256x4 single-mip ramp: grey level lo at u=pad .. hi at u=1-pad."""
    x = np.arange(RAMP_W)
    t = np.clip((x - RAMP_PAD) / (RAMP_W - 2 * RAMP_PAD - 1), 0, 1)
    g = np.clip(lo + (hi - lo) * t, 0, 255)
    rgba = np.zeros((4, RAMP_W, 4), np.uint8)
    for c in range(3):
        rgba[:, :, c] = np.clip(g * tint[c], 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    tex = textures.encode_rgba(name, rgba, alpha=False)
    tex.mips = tex.mips[:1]
    return tex
