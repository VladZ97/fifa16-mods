# SPDX-License-Identifier: GPL-2.0-or-later
"""Stock FIFA 16 asset access: pulls entries out of the game's BIG archives (update overlay
first) and caches them decoded under ~/FIFA16Mods/stock/<entry path>."""
import glob
import json
import os
import sys

HOME = os.path.expanduser("~")
MODS_ROOT = os.path.join(HOME, "FIFA16Mods")
CACHE = os.path.join(MODS_ROOT, "stock")
INDEX = os.path.join(CACHE, ".big_index.json")
SKILL_SCRIPTS = os.path.join(HOME, ".claude", "skills", "fifa-pc-patch-on-ps4", "scripts")


def game_dirs():
    cfg = json.load(open(os.path.join(MODS_ROOT, "mods.json")))
    game = cfg["game"]
    return [game, game + "-UPDATE"]          # later entries win


def _big4():
    if SKILL_SCRIPTS not in sys.path:
        sys.path.insert(0, SKILL_SCRIPTS)
    import big4
    return big4


def index():
    if os.path.exists(INDEX):
        return json.load(open(INDEX))
    big4 = _big4()
    idx = {}
    for d in game_dirs():
        for b in sorted(glob.glob(os.path.join(d, "**", "*.big"), recursive=True)):
            try:
                entries = big4.entries(b)
            except Exception:
                continue
            for name, off, size in entries:
                idx[name.lower()] = (b, off, size)
    os.makedirs(CACHE, exist_ok=True)
    json.dump(idx, open(INDEX, "w"))
    return idx


def exists(entry):
    return entry.lower() in index()


def path(entry):
    """Path of the decoded stock file (extracted on first use)."""
    entry = entry.lower()
    out = os.path.join(CACHE, entry)
    if os.path.exists(out):
        return out
    b, off, size = index()[entry]
    with open(b, "rb") as f:
        f.seek(off)
        raw = f.read(size)
    data = _big4().decode(raw)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    return out


def read(entry):
    return open(path(entry), "rb").read()


def find(*parts):
    return sorted(n for n in index() if all(p in n for p in parts))


if __name__ == "__main__":
    for n in find(*sys.argv[1:]):
        print(n)
