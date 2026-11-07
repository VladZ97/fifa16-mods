# SPDX-License-Identifier: GPL-2.0-or-later
"""Fox Engine FMDL reader (PES 2016-2021 static models), numpy based.

Only what a stadium conversion needs: LOD 0 geometry (positions, normals, colours, UV sets,
triangle indices), the material instance of each mesh (shader, technique, texture slots,
parameters) and the mesh group names. Bones/skinning are ignored.
"""
import struct
from dataclasses import dataclass, field

import numpy as np

ENTRY_SIZE = {0: 48, 1: 8, 2: 32, 3: 48, 4: 16, 5: 68, 6: 4, 7: 4, 8: 4, 9: 8, 10: 8, 11: 4,
              12: 8, 13: 32, 14: 16, 16: 16, 17: 8, 18: 8, 20: 128}

# vertex datum types / formats
T_POS, T_WEIGHT, T_NORMAL, T_COLOR, T_BONEIDX, T_UV0, T_TANGENT = 0, 1, 2, 3, 7, 8, 14
F_3F32, F_2F32, F_4F16, F_2F16, F_4U8N, F_4U8 = 1, 2, 6, 7, 8, 9


@dataclass
class Material:
    name: str
    shader: str
    technique: str
    textures: dict            # role -> texture path (as stored, e.g. /Assets/.../x.tga)
    params: dict              # name -> 4 floats


@dataclass
class Mesh:
    index: int
    group: str
    material: Material
    alpha_flags: int
    shadow_flags: int
    pos: np.ndarray           # (N,3) float32
    normal: np.ndarray        # (N,3) float32 or None
    color: np.ndarray         # (N,4) float32 0..1 or None
    uvs: list                 # list of (N,2) float32
    tris: np.ndarray          # (M,3) uint32
    visible: bool = True
    path: list = field(default_factory=list)   # mesh group names from the root down
    skinned: bool = False                      # has bone indices (characters, flags)


class Fmdl:
    def __init__(self, data):
        self.d = d = bytes(data)
        if d[:4] != b"FMDL":
            raise ValueError("not FMDL")
        (self.version, desc_off, _b0, _b1, n0, n1, s0off, _s0len, s1off, _s1len) = struct.unpack_from(
            "<IQQQIIIIII", d, 4)
        self.b0, self.b1 = {}, {}
        p = desc_off
        for _ in range(n0):
            bid, count, off = struct.unpack_from("<HHI", d, p); p += 8
            self.b0[bid] = (count, s0off + off)
        for _ in range(n1):
            bid, off, length = struct.unpack_from("<III", d, p); p += 12
            start = s1off + off
            if length > len(d) - start or bid == 3:
                length = len(d) - start
            self.b1[bid] = (start, length)
        self._parse()

    def _entries(self, bid, fmt):
        if bid not in self.b0:
            return []
        count, off = self.b0[bid]
        size = ENTRY_SIZE[bid]
        return [struct.unpack_from(fmt, self.d, off + i * size) for i in range(count)]

    def _parse(self):
        d = self.d
        strings = []
        for bid, length, off in self._entries(12, "<HHI"):
            start = self.b1[bid][0] + off
            strings.append(d[start:start + length].decode("utf-8", "replace"))
        self.strings = strings

        # materials
        mats = [(strings[t], strings[s]) for s, t in self._entries(8, "<HH")]   # (technique, shader)
        textures = [(strings[f], strings[dr]) for f, dr in self._entries(6, "<HH")]
        assigns = [(strings[n], ref) for n, ref in self._entries(7, "<HH")]
        params = []
        if 0 in self.b1:
            start, length = self.b1[0]
            params = [struct.unpack_from("<4f", d, start + 16 * i) for i in range(length // 16)]
        self.materials = []
        for (name, _p0, mid, tcount, pcount, tfirst, pfirst, _p1) in self._entries(4, "<HHHBBHHI"):
            tech, shader = mats[mid]
            tex = {}
            for i in range(tfirst, tfirst + tcount):
                role, tid = assigns[i]
                fn, dr = textures[tid]
                tex[role] = dr + fn
            par = {}
            for i in range(pfirst, pfirst + pcount):
                role, pid = assigns[i]
                if pid < len(params):
                    par[role] = params[pid]
            self.materials.append(Material(strings[name], shader, tech, tex, par))

        # buffers and vertex formats
        buf = [off for (_eof, _len, off, _pad) in self._entries(14, "<IIII")]
        vbase = self.b1[2][0] if 2 in self.b1 else 0
        mesh_formats = self._entries(10, "<BBBBI")     # bufferID, entryCount, stride, type, offset
        vertex_formats = self._entries(11, "<BBH")     # type, format, offset
        format_sets = []
        for (mcount, vcount, _uvf, _uvc, mfirst, vfirst) in self._entries(9, "<BBBBHH"):
            slots = []
            for i in range(mfirst, mfirst + mcount):
                bid, ecount, stride, _typ, boff = mesh_formats[i]
                slots += [(buf[bid] + boff, stride)] * ecount
            fs = []
            for i in range(vcount):
                typ, fmt, voff = vertex_formats[vfirst + i]
                base, stride = slots[i]
                fs.append((typ, fmt, vbase + base + voff, stride))
            format_sets.append(fs)
        lods = self._entries(17, "<II")

        # mesh groups
        groups = self._entries(1, "<HHhh")              # name, invisible, parent, unknown
        mesh_group = {}
        for (gid, count, first, _bbox, _u) in self._entries(2, "<4xHHHH4xH14x"):
            for i in range(first, first + count):
                mesh_group[i] = gid

        def group_path(gid):
            out, seen = [], set()
            while gid is not None and 0 <= gid < len(groups) and gid not in seen:
                seen.add(gid)
                out.append(strings[groups[gid][0]])
                gid = groups[gid][2]
            return out[::-1]

        def group_visible(gid):
            seen = set()
            while gid is not None and 0 <= gid < len(groups) and gid not in seen:
                seen.add(gid)
                if groups[gid][1]:
                    return False
                gid = groups[gid][2]
            return True

        self.meshes = []
        for mi, (alpha, shadow, mat, _bg, fmt_id, vcount, ffirst, fcount, flod) in enumerate(
                self._entries(3, "<BB2xHHHH4xIIQ16x")):
            pos = normal = color = None
            uvs = {}
            skinned = any(typ == T_BONEIDX for typ, _f, _o, _s in format_sets[fmt_id])
            for typ, fmt, off, stride in format_sets[fmt_id]:
                def view(dtype, n):
                    return np.ndarray((vcount, n), dtype=dtype, buffer=d, offset=off,
                                      strides=(stride, np.dtype(dtype).itemsize))
                if typ == T_POS and fmt == F_3F32:
                    pos = view("<f4", 3).copy()
                elif typ == T_NORMAL and fmt == F_4F16:
                    normal = view("<f2", 4)[:, :3].astype(np.float32)
                elif typ == T_COLOR and fmt == F_4U8N:
                    color = view("u1", 4).astype(np.float32) / 255.0
                elif T_UV0 <= typ <= T_UV0 + 3:
                    if fmt == F_2F16:
                        uvs[typ - T_UV0] = view("<f2", 2).astype(np.float32)
                    elif fmt == F_2F32:
                        uvs[typ - T_UV0] = view("<f4", 2).copy()
            if pos is None:
                continue
            lod_first, lod_count = lods[flod] if flod < len(lods) else (0, fcount)
            ibase = vbase + buf[2] + (ffirst + lod_first) * 2
            idx = np.frombuffer(d, dtype="<u2", count=lod_count - lod_count % 3, offset=ibase)
            tris = idx.astype(np.uint32).reshape(-1, 3)
            gid = mesh_group.get(mi)
            path = group_path(gid)
            self.meshes.append(Mesh(mi, path[-1] if path else "", self.materials[mat], alpha, shadow,
                                    pos, normal, color, [uvs[k] for k in sorted(uvs)], tris,
                                    group_visible(gid), path, skinned))


def load(path):
    return Fmdl(open(path, "rb").read())
