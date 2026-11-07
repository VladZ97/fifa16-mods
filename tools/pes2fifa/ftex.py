# SPDX-License-Identifier: GPL-2.0-or-later
"""Fox Engine FTEX texture reader (PES 2018-2021 single-file textures) and RGBA decoding."""
import io
import struct
import zlib

import numpy as np
from PIL import Image

# ftex pixel format -> (name, DDS fourcc or None, DXGI format for a DX10 header or None)
FORMATS = {
    0: ("RGBA8", None, None),
    1: ("R8", None, 61),
    2: ("DXT1", b"DXT1", None),
    3: ("DXT3", b"DXT3", None),
    4: ("DXT5", b"DXT5", None),
    8: ("BC4", None, 80),
    9: ("BC5", None, 83),
    10: ("BC6H", None, 95),
    11: ("BC7", None, 98),
}


class Ftex:
    def __init__(self, data):
        d = data
        (magic, self.version, self.fmt, self.width, self.height, self.depth, self.mipcount, _nrt,
         _flags, _u1, _u2, self.textype, ftexs, _u3) = struct.unpack_from("<4sfHHHHBBHIIIBB", d, 0)
        if magic != b"FTEX":
            raise ValueError("not FTEX")
        if ftexs:
            raise ValueError("split .ftexs textures are not supported")
        self.cube = bool(self.textype & 4)
        self.mips = []
        for j in range(max(1, self.mipcount)):
            off, usize, csize, _index, _fn, chunks = struct.unpack_from("<IIIBBH", d, 64 + 16 * j)
            self.mips.append(self._read(d, off, chunks, usize, csize))

    @staticmethod
    def _read(d, off, chunks, usize, csize):
        if chunks == 0:
            return bytes(d[off:off + usize]) if csize == 0 else zlib.decompress(d[off:off + csize])
        out = []
        for i in range(chunks):
            c, u, o = struct.unpack_from("<HHI", d, off + 8 * i)
            packed = not (o & 0x80000000)
            o &= 0x7fffffff
            buf = d[off + o:off + o + c]
            if packed:
                try:
                    buf = zlib.decompress(buf)
                except zlib.error:          # some exporters store a chunk raw without flagging it
                    if c != u:
                        raise
            out.append(bytes(buf))
        return b"".join(out)

    @property
    def format_name(self):
        return FORMATS.get(self.fmt, (f"fmt{self.fmt}",))[0]

    def mip_size(self, level):
        return max(1, self.width >> level), max(1, self.height >> level)

    def rgba(self, level=0):
        """Decode one mip level to an (h, w, 4) uint8 array."""
        w, h = self.mip_size(level)
        data = self.mips[level]
        if self.fmt == 0:
            a = np.frombuffer(data, dtype=np.uint8, count=w * h * 4).reshape(h, w, 4)
            return a[:, :, [2, 1, 0, 3]].copy()          # stored BGRA
        if self.fmt == 1:
            a = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
            return np.dstack([a, a, a, np.full_like(a, 255)])
        return decode_bc(data, w, h, *FORMATS[self.fmt][1:])


def dds_header(w, h, fourcc=None, dxgi=None, mips=1, linear=0):
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<7I", hdr, 4, 124, 0x1007 | 0x80000 | (0x20000 if mips > 1 else 0), h, w, linear, 0, mips)
    struct.pack_into("<II4s", hdr, 76, 32, 4, fourcc if fourcc else b"DX10")
    struct.pack_into("<I", hdr, 108, 0x1000 | (0x400008 if mips > 1 else 0))
    if not fourcc:
        hdr += struct.pack("<5I", dxgi, 3, 0, 1, 0)
    return bytes(hdr)


def decode_bc(data, w, h, fourcc=None, dxgi=None):
    """Decode block-compressed texel data through Pillow's DDS reader -> (h, w, 4) uint8."""
    im = Image.open(io.BytesIO(dds_header(w, h, fourcc, dxgi, 1, len(data)) + bytes(data)))
    a = np.asarray(im.convert("RGBA"))
    return a


def load(path):
    return Ftex(open(path, "rb").read())
