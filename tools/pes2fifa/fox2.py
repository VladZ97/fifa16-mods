# SPDX-License-Identifier: GPL-2.0-or-later
"""Fox Engine binary .fox2 scene reader (PES): just enough to get model placements.

Entities carry a class name and typed properties; strings are stored as 64-bit hashes that
resolve through the file's own string table.
"""
import struct

import numpy as np

# property data types -> (struct format, size)
SIMPLE = {0: ("b", 1), 1: ("B", 1), 2: ("h", 2), 3: ("H", 2), 4: ("i", 4), 5: ("I", 4), 6: ("q", 8), 7: ("Q", 8),
          8: ("f", 4), 9: ("d", 8), 10: ("?", 1)}
T_STRING, T_PATH, T_ENTITYPTR, T_VEC3, T_VEC4, T_QUAT, T_MAT3, T_MAT4, T_COLOR, T_FILEPTR, T_HANDLE, T_LINK = \
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22


class Entity:
    def __init__(self, cls, address):
        self.cls = cls
        self.address = address
        self.props = {}

    def get(self, name, default=None):
        v = self.props.get(name)
        if v is None or not v:
            return default
        return v[0]


def parse(data):
    """-> list of Entity. Property values are lists (containers are flattened)."""
    magic, _m2, count, strtab, _data_off = struct.unpack_from("<IIIII", data, 0)
    if magic != 0x786f62f2:
        raise ValueError("not a fox2 file")
    strings = {}
    p = strtab
    while p + 12 <= len(data):
        h, ln = struct.unpack_from("<QI", data, p)
        if h == 0:
            break
        strings[h] = data[p + 12:p + 12 + ln].decode("utf-8", "replace")
        p += 12 + ln

    def s(h):
        return strings.get(h, "" if h in (0, 0xb8a0bf169f98) else f"#{h:x}")

    def read_value(typ, q):
        if typ in SIMPLE:
            f, size = SIMPLE[typ]
            return struct.unpack_from("<" + f, data, q)[0], size
        if typ in (T_STRING, T_PATH, T_FILEPTR):
            return s(struct.unpack_from("<Q", data, q)[0]), 8
        if typ in (T_ENTITYPTR, T_HANDLE):
            return struct.unpack_from("<Q", data, q)[0], 8
        if typ in (T_VEC3, T_VEC4, T_QUAT, T_COLOR):
            return struct.unpack_from("<4f", data, q), 16
        if typ == T_MAT3:
            return struct.unpack_from("<9f", data, q), 36
        if typ == T_MAT4:
            return struct.unpack_from("<16f", data, q), 64
        if typ == T_LINK:
            a, b, c, h = struct.unpack_from("<QQQQ", data, q)
            return (s(a), s(b), s(c), h), 32
        return None, 0

    out = []
    p = 32
    for _ in range(count):
        hsize = struct.unpack_from("<H", data, p)[0]
        address = struct.unpack_from("<I", data, p + 10)[0]
        cls_hash, n_static, n_dyn, _off, _static_size, data_size = struct.unpack_from("<QHHIII", data, p + 28)
        ent = Entity(s(cls_hash), address)
        q = p + hsize
        for _i in range(n_static + n_dyn):
            name_hash, typ, container, arr, _o, size = struct.unpack_from("<QBBHHH", data, q)
            name = s(name_hash)
            v = q + 32
            vals = []
            if container == 0:                       # static array of `arr` values
                for _k in range(arr):
                    val, step = read_value(typ, v)
                    if step == 0:
                        break
                    vals.append(val)
                    v += step
            elif container == 2:                     # string map: key hash + value
                for _k in range(arr):
                    key = s(struct.unpack_from("<Q", data, v)[0])
                    val, step = read_value(typ, v + 8)
                    if step == 0:
                        break
                    vals.append((key, val))
                    v += 8 + step
                    v += (-(v - q)) % 16
            else:                                    # dynamic array / list / fixed array
                for _k in range(arr):
                    val, step = read_value(typ, v)
                    if step == 0:
                        break
                    vals.append(val)
                    v += step
            ent.props[name] = vals
            q += size
        out.append(ent)
        p += data_size
    return out


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def placements(data):
    """-> list of (model path, 4x4 world matrix) for every model instance in the scene."""
    ents = parse(data)
    by_addr = {e.address: e for e in ents}

    def local(e):
        m = np.eye(4)
        t = by_addr.get(e.get("transform", 0))
        if t is not None:
            tr = t.get("transform_translation", (0, 0, 0, 0))
            rq = t.get("transform_rotation_quat", (0, 0, 0, 1))
            sc = t.get("transform_scale", (1, 1, 1, 0))
            m[:3, :3] = quat_matrix(rq) @ np.diag(sc[:3])
            m[:3, 3] = tr[:3]
        return m

    def world(e, depth=0):
        m = local(e)
        parent = by_addr.get(e.get("parent", 0))
        if parent is not None and depth < 16:
            m = world(parent, depth + 1) @ m
        return m

    out = []
    for e in ents:
        f = e.get("modelFile")
        if not f:
            continue
        out.append((f, world(e)))
        for addr in e.props.get("locators", []):          # further instances of the same model
            loc = by_addr.get(addr)
            if loc is not None:
                out.append((f, world(loc)))
    return out
