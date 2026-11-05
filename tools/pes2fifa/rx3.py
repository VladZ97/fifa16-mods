"""FIFA 16 RX3 stadium model + texture container reader/writer (PS4/PC layout, little endian).

`Stadium.parse(bytes)` / `Stadium.build()` round-trip a stock `stadium_<id>.rx3` byte for
byte (see `selftest`), so anything built from the same structures has a layout the game
already accepts. `TextureFile` does the same for `stadium_<id>_<tod>_textures.rx3`.
"""
import struct
from dataclasses import dataclass, field

import numpy as np

SEC_HDR, SEC_VFMT, SEC_NAMES = 0x22b2be36, 0xc28193f0, 0x4c9b9eb2
SEC_IDX, SEC_VTX, SEC_PRIM = 0x005878f4, 0x00587aa1, 0xd48d7880
SEC_PROP, SEC_COLL, SEC_MAT = 0x28da5ce2, 0xf074fbb1, 0x075bd958
SEC_GROUP, SEC_GNAME = 0x7e2480ec, 0x0dc3ffd4
SEC_TEXHDR, SEC_TEX = 0x6bd085dc, 0x7a0b60da
NAME_GROUPTAG = 0x28f7f341

FMT_STD = "p0:00:00:0001:3f32 t0:0C:00:0001:2f16 t1:10:00:0001:2f16 n0:14:00:0001:3s10n"   # stride 24
FMT_UV1 = "p0:00:00:0001:3f32 t0:0C:00:0001:2f16"                                           # stride 16


def pad16(b):
    return b + b"\0" * (-len(b) % 16)


def read_sections(d):
    if d[:4] != b"RX3l":
        raise ValueError("not RX3l")
    ver, size, count = struct.unpack_from("<III", d, 4)
    out = []
    for i in range(count):
        sid, off, sz, _z = struct.unpack_from("<IIII", d, 16 + 16 * i)
        out.append((sid, d[off:off + sz]))
    return out


def write_sections(secs):
    table = 16 + 16 * len(secs)
    off = table
    head = bytearray()
    body = bytearray()
    for sid, payload in secs:
        head += struct.pack("<IIII", sid, off, len(payload), 0)
        body += payload
        off += len(payload)
    return struct.pack("<4sIII", b"RX3l", 4, off, len(secs)) + bytes(head) + bytes(body)


def cstr(b, p):
    e = b.index(b"\0", p)
    return b[p:e].decode("latin1"), e + 1


@dataclass
class Mesh:
    name: str
    fmt: str
    stride: int
    vdata: bytes            # count * stride bytes
    indices: np.ndarray     # uint16
    prim: int = 4           # 4 = triangle list

    @property
    def vcount(self):
        return len(self.vdata) // self.stride

    def positions(self):
        a = np.frombuffer(self.vdata, dtype=np.uint8).reshape(-1, self.stride)
        if self.fmt.startswith("p0:00:00:0001:4f16"):
            return a[:, :8].copy().view("<f2").astype(np.float32)[:, :3]
        return a[:, :12].copy().view("<f4")

    def element(self, tag):
        """-> (offset, type string) of a vertex element such as 't0' or 'n0', or None."""
        for part in self.fmt.split():
            f = part.split(":")
            if f[0] == tag:
                return int(f[1], 16), f[4]
        return None


@dataclass
class Group:
    name: str
    entries: list           # (mesh index, material index, bbmin(3), bbmax(3))
    bbmin: tuple = (0.0, 0.0, 0.0)
    bbmax: tuple = (0.0, 0.0, 0.0)


@dataclass
class Stadium:
    meshes: list = field(default_factory=list)
    props: list = field(default_factory=list)        # (name, (x,y,z), (rx,ry,rz))
    collisions: list = field(default_factory=list)   # (name, float32 array (n,3,3))
    materials: list = field(default_factory=list)    # (shader, [(slot, texture name)])
    textures: list = field(default_factory=list)     # texture names, index = material reference
    groups: list = field(default_factory=list)
    collision_group: bool = True                     # trailing "CollisionGeometry" group name

    # ---------------------------------------------------------------- parse
    @classmethod
    def parse(cls, d):
        s = cls()
        secs = read_sections(d)
        names = {}
        fmts, idxs, vtxs, prims, mats, groups, gnames, props, colls = [], [], [], [], [], [], [], [], []
        for sid, p in secs:
            if sid == SEC_NAMES:
                _size, count = struct.unpack_from("<II", p, 0)
                q = 16
                for _ in range(count):
                    t, ln = struct.unpack_from("<II", p, q)
                    names.setdefault(t, []).append(p[q + 8:q + 8 + ln - 1].decode("latin1"))
                    q += 8 + ln
            elif sid == SEC_VFMT:
                fmts.append(cstr(p, 16)[0])
            elif sid == SEC_IDX:
                _size, count, isz, _z = struct.unpack_from("<IIII", p, 0)
                if isz != 2:
                    raise ValueError("32-bit indices not supported")
                idxs.append(np.frombuffer(p, dtype="<u2", count=count, offset=16).copy())
            elif sid == SEC_VTX:
                _size, count, stride, _one = struct.unpack_from("<IIII", p, 0)
                vtxs.append((stride, p[16:16 + count * stride]))
            elif sid == SEC_PRIM:
                prims.append(struct.unpack_from("<I", p, 0)[0])
            elif sid == SEC_PROP:
                v = struct.unpack_from("<I6fI", p, 0)
                props.append((v[1:4], v[4:7]))
            elif sid == SEC_COLL:
                name, q = cstr(p, 16)
                _one, ntri = struct.unpack_from("<II", p, q)
                tris = np.frombuffer(p, dtype="<f4", count=ntri * 9, offset=q + 8).reshape(-1, 3, 3).copy()
                colls.append((name, tris))
            elif sid == SEC_MAT:
                _size, ntex = struct.unpack_from("<II", p, 0)
                shader, q = cstr(p, 16)
                slots = []
                for _ in range(ntex):
                    slot, q = cstr(p, q)
                    slots.append((slot, struct.unpack_from("<I", p, q)[0]))
                    q += 4
                mats.append((shader, slots))
            elif sid == SEC_GROUP:
                bb = struct.unpack_from("<8f", p, 80)
                n = struct.unpack_from("<I", p, 112)[0]
                ents = []
                for j in range(n):
                    e = struct.unpack_from("<8fII", p, 120 + 40 * j)
                    ents.append((e[8], e[9], e[0:3], e[4:7]))
                groups.append((bb[0:3], bb[4:7], ents))
            elif sid == SEC_GNAME:
                gnames.append(cstr(p, 16)[0])
        mesh_names = names.get(SEC_PRIM, [])
        for i, fmt in enumerate(fmts):
            stride, vdata = vtxs[i]
            s.meshes.append(Mesh(mesh_names[i] if i < len(mesh_names) else "", fmt, stride, vdata, idxs[i], prims[i]))
        s.textures = names.get(SEC_TEX, [])
        s.props = [(n, pos, rot) for n, (pos, rot) in zip(names.get(SEC_PROP, []), props)]
        s.collisions = colls
        s.materials = [(sh, [(slot, s.textures[ti]) for slot, ti in slots]) for sh, slots in mats]
        s.groups = [Group(gnames[i], ents, bmin, bmax) for i, (bmin, bmax, ents) in enumerate(groups)]
        s.collision_group = len(gnames) > len(groups)
        return s

    # ---------------------------------------------------------------- build
    def build(self):
        tex_index = {n: i for i, n in enumerate(self.textures)}
        secs = []
        hdr = struct.pack("<IIII", len(self.meshes), 0, 0, 0)
        for m in self.meshes:
            hdr += struct.pack("<IIII", len(pad16(b"\0" * (16 + 2 * len(m.indices)))), len(m.indices), 2, 0)
        secs.append((SEC_HDR, hdr))
        for m in self.meshes:
            body = m.fmt.encode("latin1") + b"\0"
            size = len(pad16(b"\0" * (16 + len(body))))
            secs.append((SEC_VFMT, pad16(struct.pack("<IIII", size, len(body), 0, 0) + body)))
        entries = [(SEC_PROP, n) for n, _p, _r in self.props]
        entries += [(SEC_PRIM, m.name) for m in self.meshes]
        entries += [(SEC_TEX, n) for n in self.textures]
        entries += [(NAME_GROUPTAG, "") for _ in self.groups]
        body = b"".join(struct.pack("<II", t, len(n) + 1) + n.encode("latin1") + b"\0" for t, n in entries)
        size = len(pad16(b"\0" * (16 + len(body))))
        secs.append((SEC_NAMES, pad16(struct.pack("<IIII", size, len(entries), 0, 0) + body)))
        for m in self.meshes:
            raw = np.asarray(m.indices, dtype="<u2").tobytes()
            size = len(pad16(b"\0" * (16 + len(raw))))
            secs.append((SEC_IDX, pad16(struct.pack("<IIII", size, len(m.indices), 2, 0) + raw)))
        for m in self.meshes:
            size = len(pad16(b"\0" * (16 + len(m.vdata))))
            secs.append((SEC_VTX, pad16(struct.pack("<IIII", size, m.vcount, m.stride, 1) + m.vdata)))
        for m in self.meshes:
            secs.append((SEC_PRIM, struct.pack("<IIII", m.prim, 0, 0, 0)))
        for _n, pos, rot in self.props:
            secs.append((SEC_PROP, struct.pack("<I6fI", 32, *pos, *rot, 0)))
        for name, tris in self.collisions:
            body = name.encode("latin1") + b"\0" + struct.pack("<II", 1, len(tris)) + np.asarray(tris, dtype="<f4").tobytes()
            size = len(pad16(b"\0" * (16 + len(body))))
            secs.append((SEC_COLL, pad16(struct.pack("<IIII", size, 1, 0, 0) + body)))
        for shader, slots in self.materials:
            body = shader.encode("latin1") + b"\0"
            for slot, tex in slots:
                body += slot.encode("latin1") + b"\0" + struct.pack("<I", tex_index[tex])
            size = len(pad16(b"\0" * (16 + len(body))))
            secs.append((SEC_MAT, pad16(struct.pack("<IIII", size, len(slots), 0, 0) + body)))
        for g in self.groups:
            body = struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
            body += struct.pack("<8f", *g.bbmin, 1.0, *g.bbmax, 1.0)
            body += struct.pack("<II", len(g.entries), 0xffffffff)
            for mi, ma, bmin, bmax in g.entries:
                body += struct.pack("<8fII", *bmin, 1.0, *bmax, 1.0, mi, ma)
            size = len(pad16(b"\0" * (16 + len(body))))
            secs.append((SEC_GROUP, pad16(struct.pack("<IIII", size, 1, 0xffffffff, 0) + body)))
        for i, g in enumerate(self.groups):
            body = g.name.encode("latin1") + b"\0" + struct.pack("<I", i)
            size = len(pad16(b"\0" * (16 + len(body))))
            secs.append((SEC_GNAME, pad16(struct.pack("<IIII", size, 1, 0, 0) + body)))
        if self.collision_group:
            body = b"CollisionGeometry\0"
            size = len(pad16(b"\0" * (16 + len(body) + 4)))
            secs.append((SEC_GNAME, (struct.pack("<IIII", size, 0, 0, 0) + body).ljust(size, b"\0")))
        return write_sections(secs)

    def group(self, name):
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def refresh_bounds(self):
        """Recompute per-entry and per-group bounding boxes from the vertex data."""
        for g in self.groups:
            ents = []
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for mi, ma, _a, _b in g.entries:
                p = self.meshes[mi].positions()
                a, b = p.min(0), p.max(0)
                lo, hi = np.minimum(lo, a), np.maximum(hi, b)
                ents.append((mi, ma, tuple(map(float, a)), tuple(map(float, b))))
            g.entries = ents
            if ents:
                g.bbmin, g.bbmax = tuple(map(float, lo)), tuple(map(float, hi))


# ------------------------------------------------------------------ textures
# format ids seen in stock PS4 stadium texture files
TEX_DXT1, TEX_DXT5, TEX_ARGB = 0, 2, 3


@dataclass
class Texture:
    name: str
    fmt: int
    width: int
    height: int
    mips: list              # list of (pitch, rows, data bytes)
    kind: int = 1           # header byte 4 (1 = 2D)
    flags: int = 1          # header u16 at 6
    faces: int = 1          # header u16 at 12


class TextureFile:
    def __init__(self):
        self.textures = []

    @classmethod
    def parse(cls, d):
        t = cls()
        names = []
        raw = []
        for sid, p in read_sections(d):
            if sid == SEC_NAMES:
                _size, count = struct.unpack_from("<II", p, 0)
                q = 16
                for _ in range(count):
                    _t, ln = struct.unpack_from("<II", p, q)
                    names.append(p[q + 8:q + 8 + ln - 1].decode("latin1"))
                    q += 8 + ln
            elif sid == SEC_TEX:
                raw.append(p)
        for i, p in enumerate(raw):
            _size, kind, fmt, flags, w, h, faces, nm = struct.unpack_from("<IBBHHHHH", p, 0)
            q = 16
            mips = []
            for _ in range(nm * max(1, faces if kind != 1 else 1)):
                pitch, rows, size, _z = struct.unpack_from("<IIII", p, q)
                mips.append((pitch, rows, p[q + 16:q + 16 + size]))
                q += 16 + size
            t.textures.append(Texture(names[i] if i < len(names) else "", fmt, w, h, mips, kind, flags, faces))
        return t

    def build(self):
        payloads = []
        for x in self.textures:
            body = b"".join(struct.pack("<IIII", pitch, rows, len(data), 0) + data for pitch, rows, data in x.mips)
            size = len(pad16(b"\0" * (16 + len(body))))
            nm = len(x.mips) // (x.faces if x.kind != 1 else 1)
            head = struct.pack("<IBBHHHHH", size, x.kind, x.fmt, x.flags, x.width, x.height, x.faces, nm)
            payloads.append((head, pad16(head + body)))
        hdr = struct.pack("<IIII", len(self.textures), 0, 0, 0) + b"".join(h for h, _p in payloads)
        secs = [(SEC_TEXHDR, hdr)] + [(SEC_TEX, p) for _h, p in payloads]
        body = b"".join(struct.pack("<II", SEC_TEX, len(x.name) + 1) + x.name.encode("latin1") + b"\0"
                        for x in self.textures)
        size = len(pad16(b"\0" * (16 + len(body))))
        secs.append((SEC_NAMES, pad16(struct.pack("<IIII", size, len(self.textures), 0, 0) + body)))
        return write_sections(secs)


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def selftest(model_path, tex_path=None):
    d = open(model_path, "rb").read()
    out = Stadium.parse(d).build()
    print(f"model  : {len(d)} -> {len(out)} bytes, identical={out == d}, first diff={first_diff(d, out)}")
    if tex_path:
        d = open(tex_path, "rb").read()
        out = TextureFile.parse(d).build()
        print(f"texture: {len(d)} -> {len(out)} bytes, identical={out == d}, first diff={first_diff(d, out)}")


if __name__ == "__main__":
    import sys
    selftest(*sys.argv[1:3])
