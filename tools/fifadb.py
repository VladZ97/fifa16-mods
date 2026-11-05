#!/usr/bin/env python3
"""Minimal reader for FIFA 16 t3db databases (fifa_ng_db.db + -meta.xml).

  fifadb.py <db> <table> [field,field,...]      dump rows as TSV
  fifadb.py <db> teamleague                       teamid<TAB>leagueid from leagueteamlinks

Layout (verified on the FIP16 V9.2 db): 'DB\\0\\x08', u32 0, u32 file size, u32 0, u32 table
count, u32 crc, then <count> x {char[4] shortname, u32 offset} (offsets relative to the end
of the directory). Table: 40-byte header {u32 prevcrc, u32, u32 record bytes, u32 record bits,
u32, u16 records, u16 records, u16, u16, u32 field count, u32, u32 crc}, then field
descriptors {u32 type (0 string, 3 int ...), u32 bit offset, char[4] shortname, u32 depth(bits)}
and the records, bit-packed little-endian (LSB first) per record. Integers are stored as
value - rangelow (rangelow from the meta XML); strings are byte-aligned UTF-8.
"""
import re, struct, sys


def load_meta(meta_path):
    meta = open(meta_path, encoding="utf-8").read()
    tables = {}
    for tm in re.finditer(r'<table name="([^"]+)" shortname="([^"]+)".*?</table>', meta, re.S):
        fields = {}
        for fm in re.finditer(r'<field name="([^"]+)" shortname="([^"]+)" type="([^"]+)" depth="(\d+)"[^>]*?rangelow="(-?\d+)"', tm.group(0)):
            fields[fm.group(2)] = (fm.group(1), fm.group(3), int(fm.group(4)), int(fm.group(5)))
        tables[tm.group(1)] = (tm.group(2), fields)
    return tables


def read_table(db_path, table_name):
    d = open(db_path, "rb").read()
    assert d[:4] == b"DB\x00\x08", "not a t3db"
    count, = struct.unpack_from("<I", d, 16)
    base = 24 + 8 * count
    tables = load_meta(db_path.replace(".db", "-meta.xml"))
    sn, fields = tables[table_name]
    off = None
    for i in range(count):
        if d[24 + 8 * i:28 + 8 * i].decode("latin1") == sn:
            off = base + struct.unpack_from("<I", d, 28 + 8 * i)[0]
    if off is None:
        raise KeyError(table_name)
    rec_bytes, = struct.unpack_from("<I", d, off + 8)
    n_rec, = struct.unpack_from("<H", d, off + 22)
    n_fields, = struct.unpack_from("<I", d, off + 28)
    descs = []
    for i in range(n_fields):
        ftype, bitoff = struct.unpack_from("<II", d, off + 40 + 16 * i)
        short = d[off + 48 + 16 * i:off + 52 + 16 * i].decode("latin1")
        depth, = struct.unpack_from("<I", d, off + 52 + 16 * i)
        f = fields.get(short, (short, "?", depth, 0))
        descs.append((f[0], ftype, bitoff, depth, f[3]))
    data = off + 40 + 16 * n_fields
    rows = []
    for r in range(n_rec):
        rec = d[data + r * rec_bytes:data + (r + 1) * rec_bytes]
        val = int.from_bytes(rec, "little")
        row = {}
        for name, ftype, bitoff, depth, rangelow in descs:
            if ftype == 0:  # string, byte-aligned
                raw = rec[bitoff // 8: bitoff // 8 + depth]
                row[name] = raw.split(b"\0")[0].decode("utf-8", errors="replace")
            else:  # integers are stored as value - rangelow
                row[name] = ((val >> bitoff) & ((1 << depth) - 1)) + rangelow
        rows.append(row)
    return rows


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__); return
    db, what = a[0], a[1]
    if what == "teamleague":
        for row in read_table(db, "leagueteamlinks"):
            print(f"{row['teamid']}\t{row['leagueid']}")
        return
    rows = read_table(db, what)
    cols = a[2].split(",") if len(a) > 2 else list(rows[0].keys())
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
