# SPDX-License-Identifier: GPL-2.0-or-later
"""Fox Engine FPK / FPKD archive reader (PES 2015-2021, Windows packages)."""
import struct


def read(path):
    """-> list of (name, bytes). Entry names are stored in the clear in PES packages."""
    d = open(path, "rb").read()
    if d[:6] != b"foxfpk" or d[7:10] != b"win":
        raise ValueError(f"{path}: not a Windows FPK")
    count, refs = struct.unpack_from("<II", d, 36)
    out = []
    for i in range(count):
        off, _, size, _, noff, _, nlen, _ = struct.unpack_from("<IIIIIIII", d, 48 + 48 * i)
        name = d[noff:noff + nlen].decode("utf-8", "replace")
        out.append((name, d[off:off + size]))
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        for n, b in read(p):
            print(f"{len(b):>10}  {n}")
