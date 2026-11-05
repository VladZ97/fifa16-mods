#!/usr/bin/env python3
"""Convert compressed DDS UI images (DXT1/DXT5) to the uncompressed 32-bit BGRA layout the
PS4 build of FIFA 16 loads for data/ui/imgAssets (every crest the game and the FIP patch
ship is uncompressed; DXT5 crests from PC scoreboard mods render blank).

  dds_ps4.py convert <src.dds> <dst.dds>      (dst may equal src)
  dds_ps4.py check <file.dds>
Pure Python, no Pillow: BC1/BC3 block decode, box-filtered mip chain, header cloned from the
patch's own crests (flags 0x81007, caps 0x401008, masks R 0xff0000 G 0xff00 B 0xff A 0xff000000).
"""
import struct, sys


def _c565(v):
    r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31
    return (r * 255 // 31, g * 255 // 63, b * 255 // 31)


def _bc1_block(blk, out, x0, y0, w, h, alpha_from=None):
    c0, c1, bits = struct.unpack_from("<HHI", blk, 0)
    p0, p1 = _c565(c0), _c565(c1)
    if c0 > c1 or alpha_from is not None:
        pal = [p0, p1, tuple((2 * a + b) // 3 for a, b in zip(p0, p1)), tuple((a + 2 * b) // 3 for a, b in zip(p0, p1))]
        apal = [255, 255, 255, 255]
    else:
        pal = [p0, p1, tuple((a + b) // 2 for a, b in zip(p0, p1)), (0, 0, 0)]
        apal = [255, 255, 255, 0]
    for j in range(4):
        for i in range(4):
            x, y = x0 + i, y0 + j
            if x >= w or y >= h:
                continue
            idx = (bits >> (2 * (j * 4 + i))) & 3
            r, g, b = pal[idx]
            a = alpha_from[j * 4 + i] if alpha_from is not None else apal[idx]
            o = (y * w + x) * 4
            out[o:o + 4] = bytes((b, g, r, a))


def _bc3_alpha(blk):
    a0, a1 = blk[0], blk[1]
    bits = int.from_bytes(blk[2:8], "little")
    if a0 > a1:
        pal = [a0, a1] + [((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)]
    else:
        pal = [a0, a1] + [((5 - k) * a0 + k * a1) // 5 for k in range(1, 5)] + [0, 255]
    return [pal[(bits >> (3 * n)) & 7] for n in range(16)]


def decode(d):
    """-> (w, h, bgra bytes of the top level)"""
    assert d[:4] == b"DDS ", "not DDS"
    h, w = struct.unpack_from("<II", d, 12)
    pf_flags, fourcc, bits = struct.unpack_from("<I4sI", d, 80)
    off = 128
    if fourcc == b"DX10":
        raise ValueError("DX10 header not supported")
    if pf_flags & 0x4 and fourcc in (b"DXT1", b"DXT5", b"DXT3"):
        out = bytearray(w * h * 4)
        bw, bh = (w + 3) // 4, (h + 3) // 4
        bs = 8 if fourcc == b"DXT1" else 16
        for by in range(bh):
            for bx in range(bw):
                blk = d[off:off + bs]; off += bs
                if fourcc == b"DXT1":
                    _bc1_block(blk, out, bx * 4, by * 4, w, h)
                elif fourcc == b"DXT5":
                    _bc1_block(blk[8:], out, bx * 4, by * 4, w, h, _bc3_alpha(blk[:8]))
                else:  # DXT3: explicit 4-bit alpha
                    abits = int.from_bytes(blk[:8], "little")
                    alpha = [((abits >> (4 * n)) & 15) * 17 for n in range(16)]
                    _bc1_block(blk[8:], out, bx * 4, by * 4, w, h, alpha)
        return w, h, bytes(out)
    if bits == 32 and not (pf_flags & 0x4):
        masks = struct.unpack_from("<4I", d, 92)
        raw = d[off:off + w * h * 4]
        if masks == (0xff0000, 0xff00, 0xff, 0xff000000):
            return w, h, raw
        raise ValueError(f"unsupported 32-bit masks {masks}")
    raise ValueError(f"unsupported DDS: fourcc={fourcc} bits={bits}")


def box_down(w, h, px):
    nw, nh = max(1, w // 2), max(1, h // 2)
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        for x in range(nw):
            acc = [0, 0, 0, 0]; n = 0
            for dy in (0, 1):
                for dx in (0, 1):
                    sx, sy = min(w - 1, 2 * x + dx), min(h - 1, 2 * y + dy)
                    o = (sy * w + sx) * 4
                    for c in range(4):
                        acc[c] += px[o + c]
                    n += 1
            o = (y * nw + x) * 4
            out[o:o + 4] = bytes(v // n for v in acc)
    return nw, nh, bytes(out)


def encode(w, h, px, mips=4):
    levels = [(w, h, px)]
    while len(levels) < mips and (levels[-1][0] > 1 or levels[-1][1] > 1):
        levels.append(box_down(*levels[-1]))
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<7I", hdr, 4, 124, 0x81007, h, w, w * h, 0, len(levels))
    struct.pack_into("<II4sIIIII", hdr, 76, 32, 0x41, b"\0\0\0\0", 32, 0xff0000, 0xff00, 0xff, 0xff000000)
    struct.pack_into("<4I", hdr, 108, 0x401008 if len(levels) > 1 else 0x1000, 0, 0, 0)
    return bytes(hdr) + b"".join(l[2] for l in levels)


def convert_bytes(d):
    pf_flags, fourcc = struct.unpack_from("<I4s", d, 80)
    if not (pf_flags & 0x4):
        raise ValueError("already uncompressed")
    w, h, px = decode(d)
    return encode(w, h, px)


def main():
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "convert":
        d = open(a[1], "rb").read(); open(a[2], "wb").write(convert_bytes(d)); print("wrote", a[2])
    elif len(a) >= 2 and a[0] == "check":
        for p in a[1:]:
            d = open(p, "rb").read()
            h, w, pitch, depth, mips = struct.unpack_from("<5I", d, 12)
            pf_flags, fourcc, bits = struct.unpack_from("<I4sI", d, 80)
            print(f"{p}: {w}x{h} mips={mips} fourcc={fourcc} pf_flags={pf_flags:#x} bits={bits} size={len(d)}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
