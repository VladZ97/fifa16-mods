#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Convert FIFA 16 PC hair / hairlod RX3 meshes to the PS4 vertex layout.

PC files store positions as 3 x float32 (stride 40); every stock PS4 hair stores 4 x
float16 with w = 1.0 (stride 36). The PS4 build reads the declared layout for the main
draw but another hair path assumes the console layout, so PC hair meshes explode across
the pitch in replays/close-ups. This rewrites the vertex sections, the format strings
and the section table; indices, bone palettes and textures are untouched.

  hair_ps4.py convert <src dir> <dst dir> [--only id1,id2]   (recursively, hair_* and hairlod_*)
  hair_ps4.py check <file.rx3>                                 print layout summary
"""
import os, struct, sys

PC = "p0:00:00:0001:3f32 n0:0C:00:0001:3s10n g0:10:00:0001:3s10n t0:14:00:0001:2f16 t1:18:00:0001:2f16 i0:1C:00:0001:4u16 w0:24:00:0001:4u8n"
PS4 = "p0:00:00:0001:4f16 n0:08:00:0001:3s10n g0:0C:00:0001:3s10n t0:10:00:0001:2f16 t1:14:00:0001:2f16 i0:18:00:0001:4u16 w0:20:00:0001:4u8n"
SEC_FMT, SEC_VTX = 0xc28193f0, 0x00587aa1


def sections(d):
    ver, size, count = struct.unpack_from("<III", d, 4)
    return ver, size, [list(struct.unpack_from("<IIII", d, 16 + 16 * i)) for i in range(count)]


def align16(n):
    return (n + 15) & ~15


def convert_bytes(d):
    """Return (new bytes, note). Raises ValueError when the file is not a PC-layout hair."""
    if d[:4] != b"RX3l":
        raise ValueError("not RX3l")
    ver, fsize, secs = sections(d)
    if not secs:
        raise ValueError("no sections")
    # section payloads in file order
    order = sorted(range(len(secs)), key=lambda i: secs[i][1])
    payloads = {}
    fmt_count = vtx_count = 0
    for i in order:
        sid, off, sz, z = secs[i]
        p = d[off:off + sz]
        if sid == SEC_FMT:
            s = p[16:].split(b"\0")[0].decode("latin1")
            if s == PS4:
                raise ValueError("already PS4 layout")
            if s != PC:
                raise ValueError("unexpected vertex format: " + s)
            p = p[:16] + PS4.encode("latin1") + p[16 + len(PC):]
            fmt_count += 1
        elif sid == SEC_VTX:
            ssize, count, stride, one = struct.unpack_from("<IIII", p, 0)
            if stride != 40 or ssize != sz:
                raise ValueError(f"vertex header stride={stride} size={ssize}/{sz}")
            out = bytearray()
            for v in range(count):
                b = p[16 + v * 40: 16 + (v + 1) * 40]
                x, y, z_ = struct.unpack_from("<3f", b, 0)
                out += struct.pack("<4e", x, y, z_, 1.0) + b[12:40]
            new_size = align16(16 + len(out))
            p = struct.pack("<IIII", new_size, count, 36, one) + bytes(out)
            p += b"\0" * (new_size - len(p))
            vtx_count += 1
        payloads[i] = p
    if fmt_count == 0 or vtx_count == 0:
        raise ValueError("no vertex data")
    # rebuild: header + table + payloads at the original order, 16-byte aligned
    table_end = 16 + 16 * len(secs)
    off = align16(table_end) if secs[order[0]][1] != table_end else table_end
    off = secs[order[0]][1]  # keep the original first payload offset
    new = bytearray(d[:off])
    newsecs = [list(s) for s in secs]
    for i in order:
        p = payloads[i]
        newsecs[i][1] = len(new)
        newsecs[i][2] = len(p)
        new += p
    for i, (sid, o, sz, z) in enumerate(newsecs):
        struct.pack_into("<IIII", new, 16 + 16 * i, sid, o, sz, z)
    struct.pack_into("<I", new, 8, len(new))
    return bytes(new), f"{vtx_count} mesh(es), {len(d)} -> {len(new)} bytes"


def check(path):
    d = open(path, "rb").read()
    ver, fsize, secs = sections(d)
    print(f"{os.path.basename(path)}: size={len(d)} (hdr {fsize}) sections={len(secs)}")
    for sid, off, sz, z in secs:
        if sid == SEC_FMT:
            print("   fmt:", d[off + 16:off + sz].split(b"\0")[0].decode("latin1"))
        if sid == SEC_VTX:
            ssize, count, stride, one = struct.unpack_from("<IIII", d, off)
            print(f"   vtx: count={count} stride={stride} size={ssize}/{sz}")
            n = min(count, 2)
            for v in range(n):
                b = d[off + 16 + v * stride: off + 16 + (v + 1) * stride]
                pos = struct.unpack_from("<4e", b, 0) if stride == 36 else struct.unpack_from("<3f", b, 0)
                print(f"      v{v} pos={tuple(round(x, 3) for x in pos)}")


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    if a[0] == "check":
        for p in a[1:]:
            check(p)
        return
    if a[0] != "convert":
        sys.exit(__doc__)
    src, dst = a[1], a[2]
    only = set(a[4].split(",")) if len(a) > 4 and a[3] == "--only" else None
    done = skipped = failed = 0
    for root, dirs, files in os.walk(src):
        for fn in files:
            if fn.startswith("._") or not fn.endswith(".rx3") or "textures" in fn:
                continue
            if not (fn.startswith("hair_") or fn.startswith("hairlod_")):
                continue
            if only and fn.split("_")[1] not in only:
                continue
            sp = os.path.join(root, fn)
            rel = os.path.relpath(sp, src)
            try:
                new, note = convert_bytes(open(sp, "rb").read())
            except ValueError as e:
                skipped += 1
                print(f"skip {rel}: {e}")
                continue
            except Exception as e:
                failed += 1
                print(f"FAIL {rel}: {e}")
                continue
            tp = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            open(tp, "wb").write(new)
            done += 1
            if done % 500 == 0:
                print(f"... {done} converted")
    print(f"converted {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
