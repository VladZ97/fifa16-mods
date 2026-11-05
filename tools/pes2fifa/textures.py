"""Texture conversion: PES FTEX (DXT1/DXT5 pass-through, anything else re-encoded) -> RX3 textures."""
import io

import numpy as np
from PIL import Image

import ftex
import rx3


def dxt5_to_dxt1(data):
    """Drop the alpha half of DXT5 blocks and make the colour half valid opaque DXT1.

    DXT5 colour blocks are always in 4-colour mode; DXT1 reads c0 <= c1 as 3 colours +
    transparent, so those blocks get their endpoints swapped and the indices remapped.
    """
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 16)[:, 8:].copy()
    c = b[:, :4].copy().view("<u2")              # (n, 2) endpoints
    idx = b[:, 4:].copy().view("<u4")[:, 0]
    c0, c1 = c[:, 0].copy(), c[:, 1].copy()
    swap = c0 < c1
    c[swap, 0], c[swap, 1] = c1[swap], c0[swap]
    idx = np.where(swap, idx ^ 0x55555555, idx)
    same = c0 == c1
    idx = np.where(same, 0, idx)
    out = np.empty((len(b), 8), np.uint8)
    out[:, :4] = c.view(np.uint8).reshape(-1, 4)
    out[:, 4:] = idx.astype("<u4").view(np.uint8).reshape(-1, 4)
    return out.tobytes()


def tone_factor(peak, curve):
    """Brightness factor for colours whose largest channel is `peak` (0..255 array).
    curve = (a, gamma): peak_out / 255 = a * (peak / 255) ** gamma, never brighter than the input."""
    a, gamma = curve
    p = np.maximum(np.asarray(peak, np.float64), 1.0) / 255.0
    return np.minimum(1.0, a * p ** gamma / p)


def scale_colour(data, block_bytes, curve):
    """Apply a tone curve to the colour endpoints of DXT1 (8-byte) / DXT5 (16-byte) blocks.
    Each endpoint is scaled as a whole, so hues are kept."""
    if curve is None:
        return data
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, block_bytes).copy()
    o = block_bytes - 8
    c = b[:, o:o + 4].copy().view("<u2").astype(np.uint32)            # (n, 2)
    r, g, bl = (c >> 11) & 31, (c >> 5) & 63, c & 31
    peak = np.maximum(np.maximum(r * 255.0 / 31, g * 255.0 / 63), bl * 255.0 / 31)
    f = tone_factor(peak, curve)
    r = np.minimum(31, np.rint(r * f)).astype(np.uint32)
    g = np.minimum(63, np.rint(g * f)).astype(np.uint32)
    bl = np.minimum(31, np.rint(bl * f)).astype(np.uint32)
    n = ((r << 11) | (g << 5) | bl).astype("<u2")
    if block_bytes == 8:
        was4 = c[:, 0] > c[:, 1]
        idx = b[:, 4:8].copy().view("<u4")[:, 0]
        collapsed = was4 & (n[:, 0] <= n[:, 1])                       # would flip into 3-colour mode
        idx = np.where(collapsed, 0, idx)
        b[:, 4:8] = idx.astype("<u4").view(np.uint8).reshape(-1, 4)
        n[collapsed, 1] = n[collapsed, 0]
    b[:, o:o + 4] = n.view(np.uint8).reshape(-1, 4)
    return b.tobytes()


def mean_colour(tex):
    """Average RGB (0..255) of an rx3.Texture, taken from a small mip."""
    lvl = 0
    while lvl + 1 < len(tex.mips) and max(tex.width >> lvl, tex.height >> lvl) > 32:
        lvl += 1
    w, h = max(1, tex.width >> lvl), max(1, tex.height >> lvl)
    try:
        a = ftex.decode_bc(tex.mips[lvl][2], w, h, b"DXT1" if tex.fmt == rx3.TEX_DXT1 else b"DXT5")
    except Exception:
        return np.array([128.0, 128.0, 128.0])
    return a[:, :, :3].reshape(-1, 3).mean(0)


def mip_record(w, h, block_bytes, data):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    return (bw * block_bytes, bh, data)


def solid(name, rgb, size=4):
    """A flat-colour DXT1 texture (one mip)."""
    r, g, b = (int(v) for v in rgb)
    c = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    block = int(c).to_bytes(2, "little") * 2 + b"\0\0\0\0"
    n = (size // 4) ** 2
    return rx3.Texture(name, rx3.TEX_DXT1, size, size, [mip_record(size, size, 8, block * n)])


def encode_rgba(name, rgba, alpha):
    """RGBA uint8 (h, w, 4) -> DXT1/DXT5 texture with a full mip chain (Pillow's BCn encoder)."""
    h, w = rgba.shape[:2]
    im = Image.fromarray(rgba, "RGBA")
    mips = []
    lw, lh = w, h
    while True:
        level = im if (lw, lh) == (w, h) else im.resize((lw, lh), Image.LANCZOS)
        buf = io.BytesIO()
        level.save(buf, format="DDS", pixel_format="DXT5" if alpha else "DXT1")
        data = buf.getvalue()[128:]
        mips.append(mip_record(lw, lh, 16 if alpha else 8, data))
        if lw == 1 and lh == 1:
            break
        lw, lh = max(1, lw // 2), max(1, lh // 2)
    return rx3.Texture(name, rx3.TEX_DXT5 if alpha else rx3.TEX_DXT1, w, h, mips)


def from_ftex(name, ft, max_dim=1024, alpha=False, curve=None):
    """Convert a PES texture. DXT1/DXT5 sources keep their texel blocks (mips are dropped to
    reach max_dim); other formats are decoded and re-encoded."""
    level = 0
    while level + 1 < len(ft.mips) and max(ft.mip_size(level)) > max_dim:
        level += 1
    fmt = ft.format_name
    if fmt in ("DXT1", "DXT5") and not ft.cube:
        mips = []
        for lv in range(level, len(ft.mips)):
            w, h = ft.mip_size(lv)
            data = ft.mips[lv]
            if fmt == "DXT5" and not alpha:
                data = dxt5_to_dxt1(data)
            bb = 16 if (fmt == "DXT5" and alpha) else 8
            need = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bb
            if len(data) < need:
                break
            mips.append(mip_record(w, h, bb, scale_colour(data[:need], bb, curve)))
        w, h = ft.mip_size(level)
        out_fmt = rx3.TEX_DXT5 if (fmt == "DXT5" and alpha) else rx3.TEX_DXT1
        return rx3.Texture(name, out_fmt, w, h, mips)
    rgba = ft.rgba(level).copy()
    if curve is not None:
        f = tone_factor(rgba[:, :, :3].max(2), curve)
        rgba[:, :, :3] = np.clip(rgba[:, :, :3] * f[:, :, None], 0, 255).astype(np.uint8)
    w, h = ft.mip_size(level)
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        rgba = np.asarray(Image.fromarray(rgba, "RGBA").resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS))
    return encode_rgba(name, rgba, alpha)


def alpha_coverage(ft):
    """-> (fraction of texels with alpha < 128, fraction with 16 <= alpha < 240) on a small mip."""
    if ft.format_name == "DXT1":
        return 0.0, 0.0
    level = 0
    while level + 1 < len(ft.mips) and max(ft.mip_size(level)) > 128:
        level += 1
    try:
        a = ft.rgba(level)[:, :, 3]
    except Exception:
        return 0.0, 0.0
    return float((a < 128).mean()), float(((a >= 16) & (a < 240)).mean())
