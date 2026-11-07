#!/usr/bin/env -S python3 -I
"""fifamod - mod manager for FIFA 16 (PS4, CUSA02138) running in shadPS4 on macOS.

Every mod lives in its own folder under <store>/mods/<slug>/files, normalised to the
game's layout (data/ui/..., data/sceneassets/..., dlc/..., root *.ini).  The base patch
(FIP "Manual Install") is never modified.  `build` assembles <store>/merged from the base
plus every enabled mod (later mods win) using symlinks, and points shadPS4's
/download0 overlay (~/Library/Application Support/shadPS4/download/<TITLE>) at it.

usage:
  fifamod                          interactive menu
  fifamod list                     list mods (order = priority, last wins)
  fifamod install <archive|folder> [--name NAME] [--dry-run] [--no-build]
  fifamod info <mod>               files, base overrides, conflicts
  fifamod enable|disable <mod>     toggle (rebuilds)
  fifamod remove <mod> [--yes]     delete the mod folder (rebuilds)
  fifamod move <mod> top|bottom|up|down|<n>
  fifamod pick <mod> <extras file> <game path>   switch to a variant shipped in extras/
  fifamod exclude <mod> [<folder>] [--remove]   switch off part of a mod (e.g. data/ui/imgAssets)
  fifamod fix <mod>|base           re-run the PS4 asset fixers (PC hair layout -> PS4, ...)
  fifamod scoreboard [<comp id> <mod>|--clear]   per-competition scoreboards (patched emulator)
  fifamod conflicts                files provided by more than one enabled mod
  fifamod build                    rebuild the merged overlay
  fifamod status                   where /download0 points, what is enabled
  fifamod link <download/TITLE path> [--remove]   manage extra emulator builds' overlay links
  fifamod init --base DIR --game DIR [--title CUSA02138] [--mods-dir DIR]

For the FIFA 16 app (compiled into its bundle, FIFAMOD_APP=1 and FIFAMOD_STORE set):
  fifamod configure [--base DIR] [--game DIR] [--mods-dir DIR]   create or update the registry
  fifamod state                    the whole library as JSON
  fifamod details <mod>            one mod's parts, variants and conflicts as JSON
"""
import argparse
import datetime as _dt
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

# The store (registry + merged symlink tree) must live on APFS: exFAT fakes symlinks badly.
STORE = Path(os.environ.get("FIFAMOD_STORE", Path(__file__).resolve().parent))
# Inside the FIFA 16 app: nothing is written outside the store and the mods folder (no shadPS4
# download links, no folder next to the game unless one is already there).
APP_MODE = os.environ.get("FIFAMOD_APP") == "1"
MODS_DIR = STORE / "mods"  # overridden by "mods_dir" in mods.json (may be on the SSD)
MERGED = STORE / "merged"
REGISTRY = STORE / "mods.json"
MARKER = ".fifamod-merged"
SHAD_DOWNLOAD = Path.home() / "Library/Application Support/shadPS4/download"

# Where a folder with this name belongs in the game tree, when it is the first
# recognised component of a path inside an archive.
ANCHORS = {
    "data": "", "dlc": "",
    "sceneassets": "data", "ui": "data", "fifarna": "data", "db": "data",
    "loc": "data", "teammanagement": "data",
    "lua": "data/fifarna",
    "game": "data/ui", "fonts": "data/ui", "artassets": "data/ui", "imgassets": "data/ui",
    "layout": "data/ui", "nav": "data/ui", "pow": "data/ui",
    "overlays": "data/ui/game", "globalcomponents": "data/ui/game",
    "components": "data/ui/game", "screens": "data/ui/game", "widgets": "data/ui/game",
}
for _d in ("accessory adboard ball banner body crowd crowdchair faces flag goalnet hair "
           "hairlod heads jerseyfonts kit kitnumbers kitoverlay leaguelogo pedestal prop "
           "shoe simplecloth slc tournament trophy warmup wipe3d").split():
    ANCHORS[_d] = "data/sceneassets"
# Real folder names (case) as the game uses them.
CANON = {"imgassets": "imgAssets"}
JUNK = {".DS_Store", "Thumbs.db", "desktop.ini", "__MACOSX"}
MERGE_LINES = {"data/ui/fonts/fonts.txt"}  # union of lines across layers
# A loose fonts.txt REPLACES the packed one (an empty loose file falls back to it), so the
# merged list must start from the stock entries kept under <store>/stock/.
# FIFAMOD_RESOURCES: read-only files of the app bundle (the store holds user data only there).
RESOURCES = Path(os.environ.get("FIFAMOD_RESOURCES", STORE))
STOCK = RESOURCES / "stock"
TOOLS = RESOURCES / "tools"


def tool(name):
    """A module of tools/ (compiled into the binary in the app; imported from TOOLS otherwise)."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    return importlib.import_module(name)


# ------------------------------------------------------------------ fixers
# Asset conversions the PS4 build needs; run on every installed file (fifamod install / fix).
def fix_hair_layout(path: Path):
    """PC hair/hairlod meshes (3 x float32 positions) -> PS4 layout (4 x float16)."""
    if path.suffix.lower() != ".rx3" or "textures" in path.name.lower():
        return None
    if not (path.name.startswith("hair_") or path.name.startswith("hairlod_")):
        return None
    mod = tool("hair_ps4")
    try:
        new, note = mod.convert_bytes(path.read_bytes())
    except ValueError:
        return None  # already PS4 layout or not a hair mesh
    path.write_bytes(new)
    return "hair layout PC->PS4"


def fix_dds_uncompressed(path: Path):
    """Compressed (DXT) DDS under data/ui/imgAssets -> uncompressed 32-bit, as the PS4 build needs."""
    if path.suffix.lower() != ".dds" or "/data/ui/imgassets/" not in path.as_posix().lower():
        return None
    mod = tool("dds_ps4")
    try:
        new = mod.convert_bytes(path.read_bytes())
    except (ValueError, AssertionError, IndexError):
        return None  # already uncompressed or not convertible
    path.write_bytes(new)
    return "DDS DXT->uncompressed (imgAssets)"


FIXERS = [fix_hair_layout, fix_dds_uncompressed]


def run_fixers(files_root: Path):
    """Apply every fixer to every file under files_root; returns {fix name: count}."""
    counts = {}
    for p in files_root.rglob("*"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        for fx in FIXERS:
            r = fx(p)
            if r:
                counts[r] = counts.get(r, 0) + 1
    return counts


# ----------------------------------------------------------------- registry
def load_reg():
    if not REGISTRY.exists():
        sys.exit(f"not initialised: run  fifamod init --base <patch folder> --game <game folder>")
    reg = json.loads(REGISTRY.read_text())
    reg.setdefault("mods", [])
    global MODS_DIR
    MODS_DIR = Path(reg.get("mods_dir", STORE / "mods"))
    return reg


def save_reg(reg):
    tmp = REGISTRY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(REGISTRY)


def mod_meta_path(slug):
    return MODS_DIR / slug / "mod.json"


def load_mod(slug):
    p = mod_meta_path(slug)
    if not p.exists():
        sys.exit(f"no such mod: {slug}")
    return json.loads(p.read_text())


def save_mod(meta):
    mod_meta_path(meta["slug"]).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def find_mod(reg, name):
    """Accept slug, display name (case-insensitive) or 1-based index from `list`."""
    if name.isdigit() and 1 <= int(name) <= len(reg["mods"]):
        return reg["mods"][int(name) - 1]
    for s in reg["mods"]:
        if s == name:
            return s
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    low = norm(name)
    hits = [s for s in reg["mods"] if norm(s) == low or norm(load_mod(s)["name"]) == low]
    if len(hits) == 1:
        return hits[0]
    hits = [s for s in reg["mods"] if low in norm(s) or low in norm(load_mod(s)["name"])]
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"mod '{name}' not found" + (f" (ambiguous: {', '.join(hits)})" if hits else ""))


def slugify(name):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return s[:80] or "mod"


def mod_files(slug, honour_excludes=True):
    """Relative game paths -> absolute source paths for one mod (minus excluded parts)."""
    root = MODS_DIR / slug / "files"
    out = {}
    excludes = []
    if honour_excludes and mod_meta_path(slug).exists():
        excludes = [e.lower().strip("/") + "/" for e in load_mod(slug).get("exclude", [])]
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and not p.name.startswith("._") and p.name not in JUNK:
                rel = p.relative_to(root).as_posix()
                if any((rel.lower() + "/").startswith(e) or rel.lower().startswith(e) for e in excludes):
                    continue
                out[rel] = p
    return out


# ------------------------------------------------------------------ install
def check_source(src: Path):
    """Refuse what cannot be a mod before anything is copied: a stray folder choice must not
    fill the disk. An archive or a folder of game files only, with room for it in the store."""
    if src.suffix.lower() == ".app" or any(p.suffix.lower() == ".app" for p in src.parents):
        sys.exit(f"{src.name} is an application, not a mod")
    if src.is_dir():
        home = Path.home().resolve()
        system = [Path(p) for p in ("/", "/Applications", "/Library", "/System", "/Users", "/Volumes", "/private", "/usr",
                                    "/opt", "/bin", "/sbin", "/etc", "/var")]
        if (src in system or src == home or src.parent == Path("/Volumes") or src in
                [home / d for d in ("Desktop", "Documents", "Downloads", "Library", "Applications", "Movies", "Music", "Pictures")]
                or any(src.is_relative_to(p) for p in (Path("/System"), Path("/Library"), Path("/Applications"), Path("/usr")))):
            sys.exit(f"{src} is not a mod folder: choose the mod's own folder or archive")
    need = 0
    if src.is_file():
        need = archive_unpacked_size(src) or src.stat().st_size * 3
    else:
        for d, _, files in os.walk(src):
            for f in files:
                try:
                    need += os.lstat(os.path.join(d, f)).st_size
                except OSError:
                    pass
    return pick_tmp_base(need, src.name)


def archive_unpacked_size(src: Path):
    """The unpacked size from the archive's own listing (7zz l summary line), None if unknown."""
    seven = os.environ.get("FIFAMOD_7ZZ") or shutil.which("7zz") or shutil.which("7z")
    if not seven or not os.access(seven, os.X_OK):
        return None
    try:
        r = subprocess.run([seven, "l", "-p-", str(src)], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(r.stdout.splitlines()):
        m = re.match(r"^\S+ \S+\s+(\d+)\s+(?:\d+\s+)?\d+ files", line)
        if m:
            return int(m.group(1))
    return None


def pick_tmp_base(need: int, what: str):
    """Where to unpack: the library's own volume (APFS) when it has room, else the mods folder's
    volume (a 44 GiB patch does not fit on a small internal disk; the SSD holds the mods anyway,
    and unpacking there makes the final move a rename). Refuses, with both numbers, otherwise."""
    margin = need + need // 10 + (1 << 30)
    bases = [STORE]
    try:
        if MODS_DIR.is_dir() and os.stat(MODS_DIR).st_dev != os.stat(STORE).st_dev:
            bases.append(MODS_DIR)
    except OSError:
        pass
    frees = []
    for base in bases:
        free = shutil.disk_usage(base).free
        frees.append((base, free))
        if free >= margin:
            if base is not STORE:
                print(f"unpacking in the mods folder's volume ({free >> 30} GiB free there; "
                      f"{frees[0][1] >> 30} GiB on the library's)")
            return base
    sys.exit(f"not enough free disk space for {what}: about {need >> 20} MiB unpacked, free: " +
             ", ".join(f"{f >> 20} MiB on {b}" for b, f in frees))


def clear_tmp():
    """Leftovers of an install that was killed (only one install runs at a time)."""
    for base in {STORE, MODS_DIR}:
        tmp = base / ".tmp"
        if tmp.is_dir():
            for p in tmp.iterdir():
                if p.name.startswith("fifamod-"):
                    shutil.rmtree(p, ignore_errors=True)


def extract(src: Path, dest: Path):
    if src.is_dir():
        shutil.copytree(src, dest / src.name, symlinks=False)
        return
    bundled = os.environ.get("FIFAMOD_7ZZ")
    seven = bundled if bundled and os.access(bundled, os.X_OK) else shutil.which("7zz") or shutil.which("7z")
    # RAR: the bundled unrar first (RARLAB's own decoder; 7-Zip 23 cannot read every RAR5
    # archive, e.g. FIFA Infinity Patch 9.0: "Unsupported Method"), then 7zz, then bsdtar.
    unrar = os.environ.get("FIFAMOD_UNRAR") or (None if bundled else shutil.which("unrar"))
    if unrar and not os.access(unrar, os.X_OK):
        unrar = None
    cmds = []
    if seven:
        cmds.append([seven, "x", "-y", "-bso0", "-bsp0", "-p-", f"-o{dest}", str(src)])
    if unrar and src.suffix.lower() == ".rar":
        cmds.insert(0, [unrar, "x", "-o+", "-inul", str(src), str(dest) + "/"])
    if os.access("/usr/bin/bsdtar", os.X_OK):  # every Mac has it: zip, 7z, rar
        cmds.append(["/usr/bin/bsdtar", "-xf", str(src), "-C", str(dest)])
    if not cmds:
        sys.exit("need 7zz (brew install sevenzip) or unrar to extract archives")
    # bsdtar cannot follow the volumes of a multi-part archive: its error would only hide 7zz's.
    if any(pattern.match(src.name) for pattern, _ in VOLUME_PATTERNS):
        cmds = [c for c in cmds if c[0] != "/usr/bin/bsdtar"] or cmds
    errors = []
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)  # a SIGTERM here ends it too (see main)
        if r.returncode == 0:
            return
        text = (r.stderr.strip() or r.stdout.strip()).splitlines()
        errors.append(f"{Path(c[0]).name}: " + " | ".join(text[-3:]) if text else f"{Path(c[0]).name}: status {r.returncode}")
        for leftover in dest.iterdir():
            shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink(missing_ok=True)
    sys.exit("extraction failed: " + "; ".join(errors))


def classify(rel_parts):
    """Map a path inside the archive to (game-relative path | None, reason)."""
    low = [p.lower() for p in rel_parts]
    # Prefer an explicit data/ or dlc/ folder near the top.
    for i, p in enumerate(low[:-1]):
        if p in ("data", "dlc") and i <= 3:
            return "/".join(rel_parts[i:]), "anchor:" + p
    for i, p in enumerate(low[:-1]):
        optional = p.startswith("_") and p[1:] in ANCHORS  # "_imgAssets": optional part
        if optional:
            p = p[1:]
        if p in ANCHORS:
            prefix = ANCHORS[p]
            name = CANON.get(p, rel_parts[i].lstrip("_"))
            tail = [name] + list(rel_parts[i + 1:])
            return "/".join(([prefix] if prefix else []) + tail), ("optional:" if optional else "anchor:") + p
    fn = rel_parts[-1]
    if fn.lower().endswith(".ini") and len(rel_parts) <= 3:
        return fn, "root-ini"
    return None, "extra"


def plan_install(extracted: Path, args_skip_optional=False):
    """Return (mapping: game-rel -> src, extras: [src], notes: [str])."""
    mapping, extras, notes = {}, [], []
    optional_parts = set()
    for p in sorted(extracted.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(extracted).parts
        if any(x in JUNK for x in parts) or p.name.startswith("._"):
            continue
        rel, why = classify(parts)
        # "_font.txt" next to .ttf files: the author's "append this to fonts.txt" note.
        if rel and re.fullmatch(r"data/ui/fonts/_?fonts?\.txt", rel, re.I):
            rel = "data/ui/fonts/fonts.txt"
            notes.append(f"fonts/{p.name} treated as a fonts.txt fragment (merged at build)")
        if rel is None:
            extras.append(p)
            continue
        if why.startswith("optional:"):
            if args_skip_optional:
                extras.append(p)
                continue
            optional_parts.add(parts[: [x.lower() for x in parts].index("_" + why[9:]) + 1][-1])
        if rel in mapping:
            notes.append(f"duplicate inside archive, keeping last: {rel}")
        mapping[rel] = p
    # Placeholder rx3 files crash the PS4 build (stack_chk_fail); see the patch skill.
    for op in sorted(optional_parts):
        notes.append(f"optional folder '{op}' installed (use --skip-optional to leave it out)")
    tiny = [r for r, s in mapping.items() if r.endswith(".rx3") and s.stat().st_size <= 32]
    if tiny:
        notes.append(f"{len(tiny)} placeholder .rx3 files (<=32 bytes) dropped: they crash the PS4 build")
        for r in tiny:
            del mapping[r]
    return mapping, extras, notes


VOLUME_PATTERNS = (  # multi-part archives: (regex on the file name, how to name the first part)
    (re.compile(r"^(?P<base>.+)\.part(?P<n>\d+)\.rar$", re.I), lambda m: f"{m['base']}.part{'1'.zfill(len(m['n']))}.rar"),
    (re.compile(r"^(?P<base>.+)\.r(?P<n>\d\d)$", re.I), lambda m: f"{m['base']}.rar"),
    (re.compile(r"^(?P<base>.+\.(?:7z|zip|rar))\.(?P<n>\d{3})$", re.I), lambda m: f"{m['base']}.001"),
)


def first_volume(src: Path):
    """A multi-part archive is extracted from its first part (7zz follows the others, which must
    sit beside it). Returns (file to extract, mod name without the part suffix); any other file
    comes back unchanged. Picking a later part is redirected, not refused."""
    for pattern, first in VOLUME_PATTERNS:
        m = pattern.match(src.name)
        if not m:
            continue
        base = Path(m["base"]).stem if m["base"].lower().endswith((".7z", ".zip", ".rar")) else m["base"]
        candidate = src.with_name(first(m))
        if candidate.is_file():
            if candidate != src:
                print(f"multi-part archive: extracting from {candidate.name}")
            return candidate, base
        sys.exit(f"{src.name} is part of a multi-part archive, but its first part ({candidate.name}) is not beside it")
    return src, src.stem


def cmd_install(args):
    reg = load_reg()
    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        sys.exit(f"not found: {src}")
    if src.is_file():
        src, stem = first_volume(src)
        name = args.name or stem
    else:
        name = args.name or src.name
    tmp_base = check_source(src)
    clear_tmp()
    # Unpack on the library's volume (APFS) when it has room: the exFAT driver mangles
    # non-ASCII names. A big patch unpacks on the mods folder's volume instead (pick_tmp_base).
    (tmp_base / ".tmp").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="fifamod-", dir=tmp_base / ".tmp"))
    try:
        print(f"extracting {src.name} ...")
        extract(src, tmp)
        # If the archive has a single top folder and no explicit name, take its name.
        tops = [p for p in tmp.iterdir() if p.name not in JUNK and not p.name.startswith("._")]
        if not args.name and len(tops) == 1 and tops[0].is_dir():
            name = tops[0].name
        mapping, extras, notes = plan_install(tmp, args.skip_optional)
        if not mapping:
            sys.exit("no game files recognised in this archive; nothing installed")
        slug = slugify(name)
        if slug in reg["mods"]:
            if not args.replace:
                sys.exit(f"mod '{slug}' already installed; use --replace to overwrite it, or --name")
        base_files = mod_files_from_dir(Path(reg["base"]), mapping.keys()) if reg.get("base") else set()
        overrides = sorted(r for r in mapping if r in base_files)
        conflicts = {}
        for other in reg["mods"]:
            if other == slug:
                continue
            o = {r.lower() for r in mod_files(other)}
            hit = [r for r in mapping if r.lower() in o]
            if hit:
                conflicts[other] = hit
        print(f"\nmod:       {name}   (slug: {slug})")
        print(f"game files: {len(mapping)}   overriding base patch: {len(overrides)}   extras: {len(extras)}")
        by_dir = {}
        for r in mapping:
            by_dir[os.path.dirname(r)] = by_dir.get(os.path.dirname(r), 0) + 1
        for d, n in sorted(by_dir.items()):
            print(f"  {n:5d}  {d}/")
        for c, hit in conflicts.items():
            print(f"  conflicts with '{c}': {len(hit)} files (this mod will win if it is later in the order)")
        for n in notes:
            print(f"  note: {app_note(n) if APP_MODE else n}")
        ex_by_dir = {}
        for e in extras:
            d = str(e.relative_to(tmp).parent)
            ex_by_dir.setdefault(d, []).append(e.name)
        for d, names in sorted(ex_by_dir.items()):
            shown = ", ".join(names[:3]) + (f", ... ({len(names)} files)" if len(names) > 3 else "")
            print(f"  extra (kept in extras/, not installed): {d + '/' if d != '.' else ''}{shown}")
        if args.dry_run:
            print("\ndry run, nothing written")
            return
        dest = MODS_DIR / slug
        if dest.exists():
            safe_rmtree(dest)
        files_root = dest / "files"
        for rel, s in mapping.items():
            t = files_root / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), t, copy_function=shutil.copyfile)  # no xattrs -> no ._ files on exFAT
        if extras:
            ex = dest / "extras"
            for e in extras:
                t = ex / e.relative_to(tmp)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(e), t, copy_function=shutil.copyfile)
        meta = {
            "slug": slug, "name": name, "source": str(src),
            "installed": _dt.datetime.now().isoformat(timespec="seconds"),
            "files": len(mapping), "overrides_base": len(overrides), "notes": notes,
            "enabled": True,
        }
        for junk in list(dest.rglob("._*")):  # AppleDouble files macOS leaves on exFAT
            _unlink_any(str(junk))
        print("applying PS4 fixes ...")
        fixes = run_fixers(files_root)
        if fixes:
            meta["fixes"] = fixes
            for k, v in fixes.items():
                print(f"  fixed: {v} file(s): {k}")
        save_mod(meta)
        if slug not in reg["mods"]:
            reg["mods"].append(slug)
        save_reg(reg)
        print(f"\ninstalled to {dest}")
        if not args.no_build:
            build(reg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mod_files_from_dir(root: Path, rels):
    return {r for r in rels if (root / r).is_file()}


# -------------------------------------------------------------------- build
def _unlink_any(path):
    """Delete a file on the exFAT SSD, whose driver lists NFD names but stores NFC."""
    for cand in (path, unicodedata.normalize("NFC", path), unicodedata.normalize("NFD", path)):
        try:
            os.unlink(cand)
            return True
        except FileNotFoundError:
            continue
    return False


def safe_rmtree(p: Path):
    p = Path(p)
    if not p.exists() and not p.is_symlink():
        return
    if not any(root in p.resolve().parents for root in (STORE.resolve(), MODS_DIR.resolve())):
        sys.exit(f"refusing to delete outside the store: {p}")
    if p.is_symlink():
        p.unlink()
        return
    for d, dirs, files in os.walk(p, topdown=False):
        for f in files:
            if not _unlink_any(os.path.join(d, f)) and not f.startswith("._"):
                print(f"warning: could not delete {os.path.join(d, f)}")
        for sd in dirs:
            full = os.path.join(d, sd)
            if os.path.islink(full):
                os.unlink(full)
            else:
                try:
                    os.rmdir(full)
                except OSError as e:
                    print(f"warning: could not remove {full}: {e.strerror}")
    try:
        os.rmdir(p)
    except OSError as e:
        print(f"warning: could not remove {p}: {e.strerror}")


def winners(reg):
    """game-rel path -> (layer name, source path). Base first, enabled mods in order."""
    win = {}  # lowercase game path -> [(slug, source, original path)], last wins
    layers = []
    assigned = set(reg.get("scoreboards", {}).values())
    for slug in reg["mods"]:
        m = load_mod(slug)
        if m.get("enabled", True):
            layers.append(slug)
            for rel, s in mod_files(slug).items():
                win.setdefault(rel.lower(), []).append((slug, s, rel))
        elif slug in assigned:
            # A scoreboard mod that is assigned to a competition but not enabled still needs
            # its fonts loaded at boot (fonts.txt is global); overlays come via scoreboards/.
            n = 0
            for rel, s in mod_files(slug).items():
                if rel.lower().startswith("data/ui/fonts/"):
                    win.setdefault(rel.lower(), []).append((slug, s, rel)); n += 1
            if n:
                layers.append(f"{slug}(fonts only)")
    return win, layers


def build(reg):
    base = Path(reg["base"]) if reg.get("base") else None  # the app runs without a base patch
    if base and not base.is_dir():
        sys.exit(f"base patch folder missing: {base}")
    if reg["mods"] and not MODS_DIR.is_dir():
        sys.exit(f"mods folder missing: {MODS_DIR}")
    win, layers = winners(reg)
    touched_dirs = set()
    for rel in win:  # already lowercase
        d = os.path.dirname(rel)
        while True:
            touched_dirs.add(d)
            if not d:
                break
            d = os.path.dirname(d)
    new = STORE / "merged.new"
    safe_rmtree(new)
    STORE.mkdir(parents=True, exist_ok=True)
    new.mkdir()
    (new / MARKER).write_text(_dt.datetime.now().isoformat() + "\n")
    placed = set()
    n_dirlinks = n_filelinks = 0

    def walk(rel):
        nonlocal n_dirlinks, n_filelinks
        src_dir = base / rel if rel else base
        for e in sorted(os.scandir(src_dir), key=lambda x: x.name):
            if e.name in JUNK or e.name.startswith("._"):
                continue
            r = f"{rel}/{e.name}" if rel else e.name
            rl = r.lower()
            t = new / r
            if e.is_dir(follow_symlinks=True):
                if rl in touched_dirs:
                    t.mkdir()
                    walk(r)
                else:
                    os.symlink(e.path, t)
                    n_dirlinks += 1
            else:
                if rl in win and rl not in MERGE_LINES:
                    continue  # a mod provides it
                if rl in MERGE_LINES:
                    write_merged_lines(rl, e.path, win.get(rl, []), t)
                else:
                    os.symlink(e.path, t)
                    n_filelinks += 1
                placed.add(rl)

    if base:
        walk("")
    for rl, cands in win.items():
        if rl in placed:
            continue
        t = new / cands[-1][2]
        t.parent.mkdir(parents=True, exist_ok=True)
        if rl in MERGE_LINES:
            write_merged_lines(rl, None, cands, t)
        else:
            os.symlink(cands[-1][1], t)
            n_filelinks += 1
        placed.add(rl)

    n_sb = build_scoreboards(reg, new)

    stage_root_ini(reg, win, new)
    old = STORE / "merged.old"
    safe_rmtree(old)
    if MERGED.exists():
        MERGED.rename(old)
    new.rename(MERGED)
    safe_rmtree(old)
    link_download(reg)
    print(f"built {MERGED}: base + {len(layers)} mod(s) [{', '.join(layers) or '-'}], "
          f"{len(win)} mod files, {n_dirlinks} dir links, {n_filelinks} file links"
          + (f", scoreboards for {n_sb} competition(s)" if n_sb else ""))


SCOREBOARD_PREFIXES = ("data/ui/game/overlays/", "data/ui/game/globalcomponents/overlaycomponents_",
                       "data/ui/imgassets/crest")  # crests the scoreboard draws (served only in-match)


def build_scoreboards(reg, new: Path):
    """merged/scoreboards/<competition id>/data/ui/game/... from the assigned mods. The patched
    emulator serves these instead of the normal overlay files once it has seen the match's
    specifictournament_<team>_<comp>_<n>.rx3 request. Assigned mods need not be enabled."""
    n = 0
    for comp, slug in reg.get("scoreboards", {}).items():
        if not mod_meta_path(slug).exists():
            print(f"warning: scoreboard for competition {comp} points at missing mod {slug}")
            continue
        # excluded parts (e.g. the mod's white scoreboard crests) still go into the competition
        # folder: there they are served only while that scoreboard is active
        files = {rel: src for rel, src in mod_files(slug, honour_excludes=False).items()
                 if rel.lower().startswith(SCOREBOARD_PREFIXES)}
        for rel, src in files.items():
            t = new / "scoreboards" / str(comp) / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, t)
        if files:
            n += 1
    if n:
        # team -> league table from the base database. The emulator uses it for the rule:
        # a competition id that is a league applies only when BOTH teams belong to it
        # (friendly between two clubs of the league yes, club vs foreign club no);
        # an id with no league teams is a tournament and applies whenever the match reports it.
        (new / "scoreboards" / "teamleague.txt").write_text(
            "".join(f"{t} {l}\n" for t, l in team_leagues(reg)))
    return n


def team_leagues(reg):
    fdb = tool("fifadb")
    if not reg.get("base"):
        return []
    db = Path(reg["base"]) / "data/db/fifa_ng_db.db"
    if not db.is_file():
        return []
    return [(r["teamid"], r["leagueid"]) for r in fdb.read_table(str(db), "leagueteamlinks")]


def write_merged_lines(rel, base_src, cands, target):
    lines = []
    srcs = [p for p in [STOCK / rel] if p.is_file()]
    srcs += ([Path(base_src)] if base_src else []) + [c[1] for c in cands]
    for s in srcs:
        s = s if isinstance(s, Path) else Path(s)
        for ln in Path(s).read_text(encoding="utf-8-sig", errors="replace").splitlines():
            ln = ln.strip().lstrip("\ufeff")
            if ln and ln not in lines:
                lines.append(ln)
    target.write_text("\n".join(lines) + ("\n" if lines else ""))


def download_links(reg):
    """Every shadPS4 download/<TITLE> path that should point at the merged tree.
    The main shadPS4 folder plus any extra emulator builds listed under "links"."""
    links = [SHAD_DOWNLOAD / reg["title"]]
    for extra in reg.get("links", []):
        links.append(Path(extra).expanduser())
    return links


def link_download(reg):
    if APP_MODE:
        return  # the app hands the merged tree to the emulator itself
    for link in download_links(reg):
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            cur = os.readlink(link)
            if Path(cur) != MERGED:
                reg.setdefault("previous_targets", {})[str(link)] = cur
                save_reg(reg)
            link.unlink()
        elif link.exists():
            bak = link.with_name(link.name + ".pre-fifamod-" + _dt.datetime.now().strftime("%Y%m%d%H%M"))
            link.rename(bak)
            print(f"backed up existing download folder to {bak}")
        os.symlink(MERGED, link)


def stage_root_ini(reg, win, new=None):
    """Root *.ini files are read from /app0: staged into <game>-mods/ (sibling discovery) and,
    for the self-contained FIFA 16.app, into merged/app0/ (served as the /app0 overlay when the
    merged tree is chosen as the mods folder)."""
    sibling = Path(reg["game"]).parent / f"{reg['title']}-mods" if reg.get("game") else None
    targets = [new / "app0"] if new else []
    if sibling and (not APP_MODE or sibling.is_dir()):  # the app never creates it
        targets.insert(0, sibling)
    for mods_dir in targets:
        mods_dir.mkdir(exist_ok=True)
        _stage_ini_into(reg, win, mods_dir)


def _stage_ini_into(reg, win, mods_dir):
    srcs = {p.name: p for p in Path(reg["base"]).glob("*.ini")} if reg.get("base") else {}
    for rl, cands in win.items():
        if "/" not in rl and rl.endswith(".ini"):
            srcs[cands[-1][2]] = cands[-1][1]
    for name, s in srcs.items():
        t = mods_dir / name
        if not t.exists() or t.read_bytes() != Path(s).read_bytes():
            shutil.copy2(s, t)


# ------------------------------------------------------------------- queries
def cmd_list(args):
    reg = load_reg()
    win, _ = winners(reg)
    if not reg["mods"]:
        print("no mods installed.  fifamod install <archive>")
        return
    print(f"base: {reg['base']}")
    print(f"{'#':>2}  {'on':2}  {'files':>6}  {'base':>5}  {'clash':>5}  installed            name")
    for i, slug in enumerate(reg["mods"], 1):
        m = load_mod(slug)
        files = mod_files(slug)
        clash = sum(1 for r in files if len(win.get(r.lower(), [])) > 1 and r.lower() not in MERGE_LINES)
        on = "x" if m.get("enabled", True) else "-"
        print(f"{i:>2}  {on:2}  {len(files):>6}  {m.get('overrides_base', 0):>5}  {clash:>5}  "
              f"{m['installed'][:19]}  {m['name']}   [{slug}]")
    print("order = priority: a later mod overrides an earlier one. 'base' = files that replace the base patch.")


def cmd_info(args):
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    m = load_mod(slug)
    win, _ = winners(reg)
    files = mod_files(slug)
    print(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"folder: {MODS_DIR / slug}")
    for rel in sorted(files):
        others = [c[0] for c in win.get(rel.lower(), []) if c[0] != slug]
        tag = ""
        if (Path(reg["base"]) / rel).is_file():
            tag += " [overrides base]"
        if others:
            winner = win[rel.lower()][-1][0]
            tag += f" [also in: {', '.join(others)}; winner: {winner}]"
        print(f"  {rel}{tag}")


def cmd_conflicts(args):
    reg = load_reg()
    win, _ = winners(reg)
    n = 0
    for rl, cands in sorted(win.items()):
        if len(cands) > 1 and rl not in MERGE_LINES:  # fonts.txt is merged, not a conflict
            n += 1
            print(f"{cands[-1][2]}: {' < '.join(c[0] for c in cands)}  (winner: {cands[-1][0]})")
    print(f"{n} file(s) provided by more than one enabled mod" if n else "no conflicts between enabled mods")


def cmd_status(args):
    reg = load_reg()
    print(f"title:      {reg['title']}")
    print(f"base patch: {reg['base']}")
    print(f"game:       {reg['game']}")
    for link in download_links(reg):
        target = os.readlink(link) if link.is_symlink() else ("(real folder)" if link.exists() else "(missing)")
        ok = link.is_symlink() and Path(target) == MERGED and (MERGED / MARKER).exists()
        print(f"/download0: {link} -> {target}  {'OK' if ok else '!! not pointing at the merged overlay, run: fifamod build'}")
    if (MERGED / MARKER).exists():
        print(f"merged at:  {(MERGED / MARKER).read_text().strip()}")
    win, layers = winners(reg)
    print(f"enabled:    {', '.join(layers) or '-'}   ({len(win)} mod files)")


# ------------------------------------------------------------------- toggles
def set_enabled(args, value):
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    m = load_mod(slug)
    m["enabled"] = value
    save_mod(m)
    print(f"{slug}: {'enabled' if value else 'disabled'}")
    build(reg)


def cmd_remove(args):
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    m = load_mod(slug)
    if not args.yes:
        a = input(f"delete mod '{m['name']}' ({MODS_DIR / slug}) ? [y/N] ").strip().lower()
        if a != "y":
            print("kept")
            return
    safe_rmtree(MODS_DIR / slug)
    reg["mods"] = [s for s in reg["mods"] if s != slug]
    save_reg(reg)
    print(f"removed {slug}")
    build(reg)


def cmd_move(args):
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    order = reg["mods"]
    i = order.index(slug)
    order.pop(i)
    w = args.where.lower()
    if w == "top":
        j = 0
    elif w == "bottom":
        j = len(order)
    elif w == "up":
        j = max(0, i - 1)
    elif w == "down":
        j = min(len(order), i + 1)
    elif w.isdigit():
        j = min(len(order), max(0, int(w) - 1))
    else:
        sys.exit("where: top | bottom | up | down | <position>")
    order.insert(j, slug)
    save_reg(reg)
    cmd_list(args)
    build(reg)


def comp_kind(reg, comp):
    leagues = {l for _, l in team_leagues(reg)}
    if int(comp) in leagues:
        return "league: used when both teams belong to it"
    return "tournament: used whenever the match reports it"


def cmd_scoreboard(args):
    """Assign a scoreboard mod to a competition id, list or clear assignments."""
    reg = load_reg()
    sb = reg.setdefault("scoreboards", {})
    if args.comp is None:
        if not sb:
            print("no per-competition scoreboards assigned.  fifamod scoreboard <competition id> <mod>")
        for comp, slug in sorted(sb.items(), key=lambda kv: int(kv[0])):
            print(f"  competition {comp:>4}: {load_mod(slug)['name']}   [{slug}]  ({comp_kind(reg, comp)})")
        print("competition ids appear in the emulator log as 'Scoreboard: competition <id>' when a match loads.")
        return
    comp = str(int(args.comp))
    reg.pop("scoreboard_rules", None)
    if args.clear or (args.mod and args.mod.lower() in ("none", "default", "-")):
        sb.pop(comp, None)
        print(f"competition {comp}: default scoreboard")
    else:
        if not args.mod:
            sys.exit("usage: fifamod scoreboard <competition id> <mod> | --clear")
        slug = find_mod(reg, args.mod)
        sb[comp] = slug
        print(f"competition {comp}: {load_mod(slug)['name']}  ({comp_kind(reg, comp)})")
    save_reg(reg)
    build(reg)


def cmd_exclude(args):
    """Switch off part of a mod (a folder or file inside its game tree) without deleting it."""
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    m = load_mod(slug)
    ex = m.setdefault("exclude", [])
    if args.path is None:
        print(f"{slug}: excluded: {', '.join(ex) if ex else 'nothing'}")
        return
    path = args.path.strip("/")
    rel, _ = classify(Path(path).parts + ("x",))  # map a loose folder name the same way install does
    path = rel.rsplit("/", 1)[0] if rel else path
    all_files = mod_files(slug, honour_excludes=False)
    hits = [r for r in all_files if r.lower().startswith(path.lower() + "/") or r.lower() == path.lower()]
    if args.remove:
        ex[:] = [e for e in ex if e.lower() != path.lower()]
        print(f"{slug}: {path} included again ({len(hits)} files)")
    else:
        if not hits:
            sys.exit(f"{slug} has no files under {path}; it has: " + ", ".join(sorted({r.split('/')[0] + '/' + r.split('/')[1] for r in all_files if '/' in r})[:8]))
        if path not in ex:
            ex.append(path)
        print(f"{slug}: {path} excluded ({len(hits)} files kept on disk, not installed)")
    m["files"] = len(mod_files(slug))
    save_mod(m)
    build(reg)


def cmd_fix(args):
    """Re-run the PS4 asset fixers on an installed mod (or on the base patch with 'base')."""
    reg = load_reg()
    if args.mod == "base":
        root = Path(reg["base"])
    else:
        slug = find_mod(reg, args.mod)
        root = MODS_DIR / slug / "files"
    fixes = run_fixers(root)
    print(f"{args.mod}: " + (", ".join(f"{v} x {k}" for k, v in fixes.items()) if fixes else "nothing to fix"))
    if args.mod != "base" and fixes:
        m = load_mod(slug)
        for k, v in fixes.items():
            m.setdefault("fixes", {})[k] = m.get("fixes", {}).get(k, 0) + v
        save_mod(m)


def cmd_pick(args):
    """Copy a file from the mod's extras/ over one of its game files (variant switching)."""
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    src = MODS_DIR / slug / "extras" / args.extra
    if not src.is_file():
        hits = [p for p in (MODS_DIR / slug / "extras").rglob("*") if p.is_file() and p.name == args.extra]
        if len(hits) != 1:
            sys.exit(f"extras file not found (or ambiguous): {args.extra}\n" +
                     "\n".join("  " + p.relative_to(MODS_DIR / slug / "extras").as_posix()
                               for p in sorted((MODS_DIR / slug / "extras").rglob("*")) if p.is_file()))
        src = hits[0]
    rel, _ = classify(Path(args.game_path).parts)
    if rel is None:
        sys.exit(f"cannot map {args.game_path} into the game tree")
    t = MODS_DIR / slug / "files" / rel
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, t)
    m = load_mod(slug)
    m.setdefault("picks", {})[rel] = src.relative_to(MODS_DIR / slug / "extras").as_posix()
    m["files"] = len(mod_files(slug))
    save_mod(m)
    print(f"{slug}: {rel} <- extras/{m['picks'][rel]}")
    build(reg)


def cmd_link(args):
    """Register another emulator build's download/<TITLE> path (e.g. a portable user/ dir)."""
    reg = load_reg()
    p = str(Path(args.path).expanduser())
    links = reg.setdefault("links", [])
    if args.remove:
        links[:] = [l for l in links if l != p]
    elif p not in links:
        links.append(p)
    save_reg(reg)
    build(reg)


def cmd_init(args):
    base = Path(args.base).expanduser().resolve()
    game = Path(args.game).expanduser().resolve()
    if not (base / "data").is_dir():
        sys.exit(f"{base} has no data/ folder; is it the patch's Manual Install?")
    if not (game / "eboot.bin").exists():
        sys.exit(f"{game} has no eboot.bin; it should be the .../{args.title} game folder")
    global MODS_DIR
    STORE.mkdir(parents=True, exist_ok=True)
    if args.mods_dir:
        MODS_DIR = Path(args.mods_dir).expanduser().resolve()
    MODS_DIR.mkdir(parents=True, exist_ok=True)
    reg = {"title": args.title, "base": str(base), "game": str(game), "mods_dir": str(MODS_DIR), "mods": []}
    if REGISTRY.exists():
        reg["mods"] = json.loads(REGISTRY.read_text()).get("mods", [])
    # adopt mod folders already present in mods_dir
    for d in sorted(MODS_DIR.iterdir()):
        if (d / "mod.json").exists() and d.name not in reg["mods"]:
            reg["mods"].append(d.name)
    save_reg(reg)
    print(f"store: {STORE}\nmods:  {MODS_DIR}\nbase:  {base}\ngame:  {game}\ntitle: {args.title}")
    build(reg)


def cmd_configure(args):
    """Create or update the registry (the app calls this; unlike init it keeps everything else)."""
    STORE.mkdir(parents=True, exist_ok=True)
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {
        "title": "CUSA02138", "base": "", "game": "", "mods_dir": str(STORE / "mods"), "mods": []}
    if args.base is not None:
        base = str(Path(args.base).expanduser()) if args.base else ""
        if base and not (Path(base) / "data").is_dir():
            sys.exit(f"{base} has no data/ folder: it is not a patch's Manual Install folder")
        reg["base"] = base
    if args.game is not None:
        reg["game"] = str(Path(args.game).expanduser()) if args.game else ""
    if args.mods_dir:
        new = Path(args.mods_dir).expanduser()
        lost = [s for s in reg["mods"] if not (new / s / "mod.json").exists()]
        if lost and str(new) != reg.get("mods_dir"):
            sys.exit(f"{len(lost)} installed mod(s) are not in {new}: move them there first, or remove them")
        reg["mods_dir"] = str(new)
    global MODS_DIR
    MODS_DIR = Path(reg["mods_dir"])
    MODS_DIR.mkdir(parents=True, exist_ok=True)
    for d in sorted(MODS_DIR.iterdir()):  # mod folders already there
        if (d / "mod.json").exists() and d.name not in reg["mods"]:
            reg["mods"].append(d.name)
    save_reg(reg)
    if args.build:
        build(reg)


def cmd_state(args):
    """The library as JSON, for the app. Works with the mods drive disconnected."""
    out = {"store": str(STORE), "merged": str(MERGED), "initialised": REGISTRY.exists(), "mods": [], "scoreboards": []}
    if REGISTRY.exists():
        reg = load_reg()
        base = reg.get("base", "")
        out.update(base=base, base_ok=bool(base) and Path(base).is_dir(), game=reg.get("game", ""),
                   mods_dir=str(MODS_DIR), mods_dir_ok=MODS_DIR.is_dir())
        present = [s for s in reg["mods"] if mod_meta_path(s).exists()]
        clone = dict(reg, mods=present)
        win, layers = winners(clone)
        out["layers"] = layers
        out["conflicts"] = sum(1 for rl, c in win.items() if len(c) > 1 and rl not in MERGE_LINES)
        for slug in reg["mods"]:
            if slug not in present:
                out["mods"].append({"slug": slug, "name": slug.replace("_", " "), "enabled": False, "missing": True})
                continue
            m = load_mod(slug)
            files = mod_files(slug)
            extras = MODS_DIR / slug / "extras"
            out["mods"].append({
                "slug": slug, "name": m["name"], "enabled": m.get("enabled", True), "missing": False,
                "files": len(files), "overrides_base": m.get("overrides_base", 0),
                "clash": sum(1 for r in files if len(win.get(r.lower(), [])) > 1 and r.lower() not in MERGE_LINES),
                "installed": m.get("installed", "")[:10], "excluded": m.get("exclude", []),
                "has_extras": extras.is_dir() and any(extras.iterdir()), "notes": m.get("notes", []),
                "fixes": m.get("fixes", {}), "folder": str(MODS_DIR / slug)})
        leagues = {l for _, l in team_leagues(reg)} if reg.get("scoreboards") else set()
        for comp, slug in sorted(reg.get("scoreboards", {}).items(), key=lambda kv: int(kv[0])):
            out["scoreboards"].append({"comp": int(comp), "slug": slug, "league": int(comp) in leagues})
    if (MERGED / MARKER).exists():
        out["merged_at"] = (MERGED / MARKER).read_text().strip()
    print(json.dumps(out, ensure_ascii=False))


def app_note(note):
    """Install notes speak to the command line; the app has a switch for this one."""
    return note.replace(" (use --skip-optional to leave it out)", "")


def cmd_details(args):
    """One mod as JSON, for the app: its parts (for exclude), variants (for pick) and conflicts."""
    reg = load_reg()
    slug = find_mod(reg, args.mod)
    m = load_mod(slug)
    win, _ = winners(reg)
    every = mod_files(slug, honour_excludes=False)
    excluded = [e.strip("/") for e in m.get("exclude", [])]
    parts = {}
    for rel in every:
        bits = rel.split("/")
        for depth in range(1, min(len(bits), 5)):
            key = "/".join(bits[:depth])
            parts[key] = parts.get(key, 0) + 1
    # A part worth listing: the folder where the tree first branches, and everything below it.
    shown = sorted(k for k in parts if parts[k] < len(every) and not any(
        o.startswith(k + "/") and parts[o] == parts[k] for o in parts))
    conflicts = {}
    for rel in mod_files(slug):
        for c in win.get(rel.lower(), []):
            if c[0] != slug and rel.lower() not in MERGE_LINES:
                entry = conflicts.setdefault(c[0], {"slug": c[0], "files": 0, "wins": 0})
                entry["files"] += 1
                entry["wins"] += win[rel.lower()][-1][0] == slug
    ex_root = MODS_DIR / slug / "extras"
    extras = sorted(p.relative_to(ex_root).as_posix() for p in ex_root.rglob("*")
                    if p.is_file() and not p.name.startswith("._")) if ex_root.is_dir() else []
    variants = []
    for e in extras:
        stem, suffix = Path(e).stem.lower(), Path(e).suffix.lower()
        if not suffix or suffix in (".txt", ".pdf", ".png", ".jpg", ".jpeg", ".url", ".html", ".docx"):
            continue
        targets = sorted(r for r in every if Path(r).suffix.lower() == suffix
                         and (stem.startswith(Path(r).stem.lower()) or Path(r).stem.lower().startswith(stem)))
        variants.append({"file": e, "targets": targets[:20], "in_use": e in m.get("picks", {}).values()})
    print(json.dumps({
        "slug": slug, "name": m["name"], "source": m.get("source", ""), "installed": m.get("installed", ""),
        "folder": str(MODS_DIR / slug), "notes": [app_note(n) for n in m.get("notes", [])], "fixes": m.get("fixes", {}),
        "parts": [{"path": k, "files": parts[k], "excluded": any(
            k.lower() == e.lower() or k.lower().startswith(e.lower() + "/") for e in excluded)} for k in shown],
        "extras": extras, "variants": variants, "picks": m.get("picks", {}),
        "conflicts": sorted(conflicts.values(), key=lambda c: -c["files"])}, ensure_ascii=False))


# --------------------------------------------------------------------- menu
def menu():
    while True:
        print()
        try:
            cmd_list(None)
        except SystemExit as e:
            print(e)
        print("\n[i]nstall  [e]nable  [d]isable  [r]emove  [m]ove  [c]onflicts  [b]uild  [s]tatus  [q]uit")
        c = input("> ").strip().lower()
        try:
            if c in ("q", ""):
                return
            elif c == "i":
                src = input("archive or folder path: ").strip().strip("'\"")
                if src:
                    main(["install", src])
            elif c in ("e", "d", "r", "m"):
                which = input("mod (# or name): ").strip()
                if not which:
                    continue
                if c == "e":
                    main(["enable", which])
                elif c == "d":
                    main(["disable", which])
                elif c == "r":
                    main(["remove", which])
                else:
                    main(["move", which, input("top/bottom/up/down/<n>: ").strip()])
            elif c == "c":
                main(["conflicts"])
            elif c == "b":
                main(["build"])
            elif c == "s":
                main(["status"])
        except SystemExit as e:
            if e.code not in (0, None):
                print(e)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fifamod", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("init"); p.add_argument("--base", required=True); p.add_argument("--game", required=True)
    p.add_argument("--title", default="CUSA02138"); p.add_argument("--mods-dir", help="where mod folders are kept (default <store>/mods)")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("install"); p.add_argument("source"); p.add_argument("--name")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--no-build", action="store_true")
    p.add_argument("--replace", action="store_true"); p.add_argument("--skip-optional", action="store_true",
                   help="leave '_folder' optional parts (e.g. _imgAssets crests) in extras/")
    p.set_defaults(fn=cmd_install)
    p = sub.add_parser("configure"); p.add_argument("--base"); p.add_argument("--game"); p.add_argument("--mods-dir")
    p.add_argument("--build", action="store_true"); p.set_defaults(fn=cmd_configure)
    sub.add_parser("state").set_defaults(fn=cmd_state)
    p = sub.add_parser("details"); p.add_argument("mod"); p.set_defaults(fn=cmd_details)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("info"); p.add_argument("mod"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("enable"); p.add_argument("mod"); p.set_defaults(fn=lambda a: set_enabled(a, True))
    p = sub.add_parser("disable"); p.add_argument("mod"); p.set_defaults(fn=lambda a: set_enabled(a, False))
    p = sub.add_parser("remove"); p.add_argument("mod"); p.add_argument("--yes", "-y", action="store_true"); p.set_defaults(fn=cmd_remove)
    p = sub.add_parser("move"); p.add_argument("mod"); p.add_argument("where"); p.set_defaults(fn=cmd_move)
    p = sub.add_parser("scoreboard", help="assign a scoreboard mod to a competition id (needs the patched emulator)")
    p.add_argument("comp", nargs="?", help="competition id from the emulator log"); p.add_argument("mod", nargs="?")
    p.add_argument("--clear", action="store_true"); p.set_defaults(fn=cmd_scoreboard)
    p = sub.add_parser("exclude", help="switch off part of a mod (folder inside its game tree); --remove to re-include")
    p.add_argument("mod"); p.add_argument("path", nargs="?"); p.add_argument("--remove", action="store_true")
    p.set_defaults(fn=cmd_exclude)
    p = sub.add_parser("fix", help="re-run the PS4 asset fixers (hair layout, ...) on a mod, or 'base'")
    p.add_argument("mod"); p.set_defaults(fn=cmd_fix)
    p = sub.add_parser("pick", help="use a file from the mod's extras/ as a game file (variants)")
    p.add_argument("mod"); p.add_argument("extra", help="file name or path inside extras/")
    p.add_argument("game_path", help="target, e.g. data/ui/game/overlays/overlay_9002.big")
    p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("link", help="also point another emulator build's download/<TITLE> at the merged tree")
    p.add_argument("path"); p.add_argument("--remove", action="store_true"); p.set_defaults(fn=cmd_link)
    sub.add_parser("conflicts").set_defaults(fn=cmd_conflicts)
    sub.add_parser("build").set_defaults(fn=lambda a: build(load_reg()))
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    if not args.cmd:
        menu()
        return
    args.fn(args)


if __name__ == "__main__":
    # Cancel in the app sends SIGTERM: leave through the normal path so temporary files go.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("cancelled"))
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # the app shows progress as it comes
    sys.stderr.reconfigure(encoding="utf-8")
    main()
