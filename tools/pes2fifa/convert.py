"""PES 2021 (Fox Engine) stadium -> FIFA 16 stadium RX3 conversion.

The stock FIFA stadium being replaced is the template: its pitch, ad boards, sky, props and
collision are kept, its stands are replaced by the PES geometry.

Coordinates: PES is metres in a left-handed Y-up frame with the main camera on +Z; FIFA is
centimetres, right-handed, main camera on -Z. So p_fifa = 100 * (x, y, -z): a reflection.
(Checked against seat lettering: the stock crowd file spells MES QUE UN CLUB correctly in a
right-handed render, and the PES seat map only does so with this mapping. The usual Blender
route - import, rotate 180 degrees, export - yields a mirrored stadium.) PES winds triangles
against its normals in its own frame, which after the reflection is exactly FIFA's
convention, so indices are kept. UVs carry over unchanged.
"""
import glob
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np

import bake
import crowd
import fmdl
import fpk
import ftex
import glares
import rx3
import stock
import textures

AXIS = np.array([100.0, 100.0, -100.0], np.float32)
NAXIS = np.array([1.0, 1.0, -1.0], np.float32)
LM_NAME = "lm_pes_ramp"
VERT_DTYPE = np.dtype([("p", "<f4", 3), ("t0", "<f2", 2), ("t1", "<f2", 2), ("n", "<u4")])   # rx3.FMT_STD

DEFAULTS = dict(
    # StadiumWear (dirt decal around the pitch) is shaped for the stock stadium's surroundings
    # and z-fights with the new ground, so it is not kept.
    keep_groups=("Pitch", "AdBoard", "CrowdNet_NoShadowCast", "CrowdNet", "Sky"),
    keep_shaders={},                           # per kept group: only these shaders survive
    exterior=True,                             # convert an exterior add-on when the stadium has one
    exterior_group="Exterior_NoShadowCast",
    exterior_tris=450000,                      # triangle budget for the outside
    exterior_radius=60000.0,                   # cm; the sky dome is ~634 m
    tex_max_exterior=512,
    skip_models=("center2",),                  # PES sideline staff characters
    skip_texture_dirs=("/common/staff/", "/scarecrow/", "/common/character"),
    skip_group_patterns=(r"benchmen", r"_sc20\d\d_", r"scarecrow", r"staff"),   # PES sideline people
    tex_max=1024,
    max_tris=10000,                            # per mesh; stock meshes peak at ~11.5k triangles
    bake=True,                                 # per-vertex sky lighting; False = flat mid level
    bake_dirs=48,
    roof_transmission=0.5,                     # share of sky light that passes roof-level geometry
    bounce=0.10,                               # light level floor from ground / stand bounce
    albedo_match=True,                         # darken PES albedo to the stock stadium's level
    crowd=True,
    glares=True,                               # move floodlight glares onto the new roof edge
    crowd_max=None,                            # seat budget; default = seat count of the stock stadium
    seat_pitch=55.0,                           # cm between seats along a row
    cutout_threshold=0.02,
)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ PES side
class PesStadium:
    def __init__(self, root):
        self.root = root
        pat = os.path.join(glob.escape(root), "Asset", "model", "bg", "st*", "#Win", "st*.fpk")
        mains = [p for p in glob.glob(pat) if re.fullmatch(r"st\d+\.fpk", os.path.basename(p))]
        if not mains:
            raise FileNotFoundError(f"no Asset/model/bg/stNNN/#Win/stNNN.fpk under {root}")
        self.main_fpk = sorted(mains)[0]
        self.st = os.path.basename(self.main_fpk)[:-4]
        self.roots = [root]
        self.exterior = None
        # Stadium-server add-ons: an "exterior" add-on carries the real outside of the stadium
        # (facade, surroundings) and a main package without the placeholder shell.
        for d in sorted(glob.glob(os.path.join(glob.escape(root), "AddOn", "*", "Global"))):
            ext = os.path.join(d, "common", "demo", "fixdemoobj", f"{self.st}_bg", "#Win", f"{self.st}_h_all.fpk")
            if os.path.exists(ext):
                self.exterior = ext
                self.roots.append(d)
                alt = os.path.join(d, "Asset", "model", "bg", self.st, "#Win", f"{self.st}.fpk")
                if os.path.exists(alt):
                    self.main_fpk = alt
                break
        self._tex = {}

    def models(self):
        """-> list of (model name, Fmdl) of the main stadium package."""
        out = []
        for name, data in fpk.read(self.main_fpk):
            if name.lower().endswith(".fmdl"):
                out.append((os.path.basename(name)[:-5], fmdl.Fmdl(data)))
        return out

    def exterior_instances(self):
        """-> list of (model name, Fmdl, 4x4 placement matrix in PES space) from the exterior add-on."""
        if not self.exterior or not os.path.exists(self.exterior + "d"):
            return []
        import fox2
        raw = {os.path.basename(n).lower(): d for n, d in fpk.read(self.exterior) if n.lower().endswith(".fmdl")}
        parsed = {}
        out = []
        for name, data in fpk.read(self.exterior + "d"):
            if not name.lower().endswith(".fox2"):
                continue
            try:
                placed = fox2.placements(data)
            except Exception as e:
                log(f"  ! {os.path.basename(name)}: {e}")
                continue
            for path, matrix in placed:
                key = os.path.basename(path).lower()
                if key not in raw:
                    continue
                if key not in parsed:
                    parsed[key] = fmdl.Fmdl(raw[key])
                out.append((key[:-5], parsed[key], matrix))
        return out

    def texture_file(self, pes_path):
        rest = re.sub(r"^/?Assets/[^/]+/", "", pes_path)
        d, f = os.path.split(rest)
        name = os.path.splitext(f)[0] + ".ftex"
        for root in self.roots:
            for top in ("Asset", "common"):
                p = os.path.join(root, top, d, "#windx11", name)
                if os.path.exists(p):
                    return p
        return os.path.join(self.root, "Asset", d, "#windx11", name)

    def texture(self, pes_path):
        if pes_path not in self._tex:
            f = self.texture_file(pes_path)
            try:
                self._tex[pes_path] = ftex.load(f)
            except Exception as e:           # missing or unsupported
                log(f"  ! texture {pes_path}: {e}")
                self._tex[pes_path] = None
        return self._tex[pes_path]


def base_texture(mat):
    for role in ("Base_Tex_SRGB", "Base_Tex_LIN"):
        if role in mat.textures:
            return mat.textures[role]
    return None


def tex_name(pes_path):
    n = os.path.splitext(os.path.basename(pes_path))[0].lower()
    return re.sub(r"[^a-z0-9_]", "_", n)[:48]


# ------------------------------------------------------------------ geometry helpers
def pack_s10(n):
    v = np.clip(np.rint(n * 511.0), -511, 511).astype(np.int32) & 0x3ff
    return (v[:, 0] | (v[:, 1] << 10) | (v[:, 2] << 20)).astype("<u4")


def split_chunks(tris, pos, max_tris):
    """Recursive median split on the longest axis -> list of triangle index arrays."""
    cent = pos[tris].mean(1)
    out, stack = [], [np.arange(len(tris))]
    while stack:
        idx = stack.pop()
        if len(idx) <= max_tris:
            out.append(idx)
            continue
        c = cent[idx]
        ax = int(np.argmax(c.max(0) - c.min(0)))
        order = np.argsort(c[:, ax], kind="stable")
        h = len(idx) // 2
        stack.append(idx[order[:h]])
        stack.append(idx[order[h:]])
    return out


def make_mesh(name, pos, nrm, uv0, uv1, tris):
    """Build an rx3.Mesh (standard stadium layout) from a triangle soup subset; welds equal vertices."""
    used, inv = np.unique(tris, return_inverse=True)
    v = np.zeros(len(used), VERT_DTYPE)
    v["p"] = pos[used]
    v["t0"] = uv0[used]
    v["t1"] = uv1[used]
    v["n"] = pack_s10(nrm[used])
    raw = v.view(np.dtype((np.void, VERT_DTYPE.itemsize)))
    uniq, first, inv2 = np.unique(raw, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")          # keep original vertex order
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    v = v[first[order]]
    t = rank[inv2[inv.reshape(-1)]].reshape(-1, 3)
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t = t[ok]
    if len(t) == 0:
        return None
    return rx3.Mesh(name, rx3.FMT_STD, VERT_DTYPE.itemsize, v.tobytes(), t.astype("<u2").reshape(-1))


# ------------------------------------------------------------------ conversion
class Part:
    """Geometry of one FIFA material being accumulated from many PES meshes."""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.tris = [], [], [], []
        self.count = 0
        self.sources = set()

    def add(self, pos, nrm, uv, tris):
        self.pos.append(pos)
        self.nrm.append(nrm)
        self.uv.append(uv)
        self.tris.append(tris + self.count)
        self.count += len(pos)

    def arrays(self):
        return (np.concatenate(self.pos), np.concatenate(self.nrm), np.concatenate(self.uv),
                np.concatenate(self.tris))


def classify(mesh, pes, opts, stats, exterior=False):
    """-> (group, shader, base texture path) or None when the mesh is skipped."""
    mat = mesh.material
    base = base_texture(mat)
    if not mesh.visible:
        return stats.skip("hidden mesh group", mesh)
    if mesh.skinned:
        return stats.skip("skinned mesh (character)", mesh)
    if any(re.search(p, g, re.I) for p in opts["skip_group_patterns"] for g in mesh.path):
        return stats.skip("sideline character group", mesh)
    if mesh.shadow_flags == 2:
        return stats.skip("shadow-only mesh", mesh)
    if "3dfw_const" in mat.shader:
        return stats.skip("unlit glow / marker mesh", mesh)
    if base is None:
        return stats.skip(f"no base texture ({mat.shader})", mesh)
    if any(d in base for d in opts["skip_texture_dirs"]):
        return stats.skip("character texture", mesh)
    ft = pes.texture(base)
    if ft is None:
        return stats.skip("texture missing", mesh)
    blend = bool(mesh.alpha_flags & 0x80) or bool(mesh.alpha_flags & 0x10) or "glass" in mat.shader
    if not blend and ((mesh.alpha_flags & 0x20) or "translucent" in mat.shader):
        cut, _soft = textures.alpha_coverage(ft)
        blend = cut > opts["cutout_threshold"]
    if blend:
        return "Alpha", "diffusealpha", base
    if exterior:
        return opts["exterior_group"], "diffuseopaque", base
    if mesh.shadow_flags & 1:
        return "MainStadium_NoShadowCast", "diffuseopaque", base
    return "MainStadium", "diffuseopaque", base


class Stats:
    def __init__(self):
        self.skipped = defaultdict(lambda: [0, 0])
        self.used = [0, 0]

    def skip(self, why, mesh):
        s = self.skipped[why]
        s[0] += 1
        s[1] += len(mesh.tris)
        return None


def is_exterior(key):
    return key[0].startswith("Exterior") or key[0] == "ExteriorAlpha"


def add_mesh(parts, key, model, m, matrix=None, sink=None, keep_radius=None):
    """Transform one PES mesh into FIFA space and append it to its part. -> triangles added."""
    pos = m.pos.astype(np.float64)
    nrm = m.normal.astype(np.float64) if m.normal is not None else np.tile((0.0, 1.0, 0.0), (len(pos), 1))
    tris = m.tris.astype(np.int64)
    if matrix is not None:
        r = matrix[:3, :3]
        pos = pos @ r.T + matrix[:3, 3]
        nrm = nrm @ np.linalg.inv(r)                       # inverse transpose for normals
        if np.linalg.det(r) < 0:
            tris = tris[:, ::-1]
    pos = (pos * AXIS).astype(np.float32)
    nrm = nrm * NAXIS
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 1e-6, nrm / np.maximum(ln, 1e-6), (0, 1, 0)).astype(np.float32)
    if keep_radius is not None:
        far = np.hypot(pos[:, 0], pos[:, 2]) > keep_radius
        tris = tris[~far[tris].all(1)]
        if len(tris) == 0:
            return 0
    if sink is not None:
        # FIFA draws its own pitch at y = 0; PES ground lying in the same plane under it flickers
        x0, x1, z0, z1 = sink
        under = ((pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 2] > z0) & (pos[:, 2] < z1) &
                 (pos[:, 1] > -30.0) & (pos[:, 1] < 3.5))
        pos[under, 1] = -12.0
        if key[0] == "Alpha":
            # PES lays blended decals (worn grass, painted lines) a few millimetres above its
            # turf; that gap is too small for FIFA's depth range and shimmers, so lift them
            flat = (~under) & (np.abs(pos[:, 1]) < 6.0) & (nrm[:, 1] > 0.9)
            pos[flat, 1] += 2.5
    part = parts.setdefault(key, Part())
    part.sources.add(model)
    part.add(pos, nrm, m.uvs[0], tris)
    n = len(tris)
    if m.alpha_flags & 0x20:                               # two-sided: add the back faces
        part.add(pos, -nrm, m.uvs[0], tris[:, ::-1])
        n *= 2
    return n


def convert_geometry(pes, opts, sink=None):
    """-> (parts {(group, shader, base path): Part}, stats)"""
    stats = Stats()
    parts = OrderedDict()
    for model, f in pes.models():
        if any(model.startswith(s) for s in opts["skip_models"]):
            stats.skipped[f"model {model}"][0] += len(f.meshes)
            stats.skipped[f"model {model}"][1] += sum(len(m.tris) for m in f.meshes)
            continue
        for m in f.meshes:
            if len(m.tris) == 0 or not m.uvs:
                continue
            key = classify(m, pes, opts, stats)
            if key is None:
                continue
            stats.used[0] += 1
            stats.used[1] += add_mesh(parts, key, model, m, None, sink)
    if opts["exterior"] and pes.exterior:
        convert_exterior(pes, parts, stats, opts)
    return parts, stats


def convert_exterior(pes, parts, stats, opts):
    """Add the exterior add-on: fixed models first, then placed instances nearest first until
    the triangle budget is used up. Anything beyond the sky dome is left out."""
    radius = opts["exterior_radius"]
    budget = opts["exterior_tris"]
    inst = pes.exterior_instances()

    def dist(item):
        return float(np.hypot(item[2][0, 3], item[2][2, 3]))

    def fixed(item):
        return np.allclose(item[2], np.eye(4), atol=1e-4)

    order = sorted(inst, key=lambda it: (not fixed(it), dist(it)))
    used = placed = left_out = 0
    for model, f, matrix in order:
        if dist((model, f, matrix)) * 100.0 > radius:
            left_out += 1
            continue
        tri_count = sum(len(m.tris) for m in f.meshes)
        if not fixed((model, f, matrix)) and used + tri_count > budget:
            left_out += 1
            continue
        got = 0
        for m in f.meshes:
            if len(m.tris) == 0 or not m.uvs:
                continue
            key = classify(m, pes, opts, stats, exterior=True)
            if key is None:
                continue
            if key[0] == "Alpha":
                key = ("ExteriorAlpha",) + key[1:]
            got += add_mesh(parts, key, model, m, None if fixed((model, f, matrix)) else matrix, None, radius)
        if got:
            used += got
            placed += 1
    stats.used[1] += used
    log(f"exterior: {placed} model placements / {used} triangles from {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pes.exterior))))))))}"
        f" ({left_out} left out: beyond {radius / 100:.0f} m or over the {budget} triangle budget)")


def convert_jumbotron(pes, template, opts):
    """PES big screens (tv/#Win/tv_<st>.fpk) -> one mesh for FIFA's Jumbotron group, drawn with the
    stock jumbotron material (the game feeds it the live picture). Returns (mesh, material) or None.

    PES wraps one strip of video around the whole bowl; FIFA maps the full picture to u,v 0..1,
    so the strip is cut into 16:9 panels around the ring."""
    g = template.group("Jumbotron")
    path = os.path.join(pes.root, "Asset", "model", "bg", pes.st, "tv", "#Win", f"tv_{pes.st}.fpk")
    if g is None or not g.entries or not os.path.exists(path):
        return None
    smi, sma, _a, _b = g.entries[0]
    stock_mesh = template.meshes[smi]
    e = stock_mesh.element("t1")
    raw = np.frombuffer(stock_mesh.vdata, dtype=np.uint8).reshape(-1, stock_mesh.stride)
    lm_uv = raw[:, e[0]:e[0] + 4].copy().view("<f2").astype(np.float32).mean(0) if e else np.array([0.5, 0.5], np.float32)
    P, N, V = [], [], []
    for name, data in fpk.read(path):
        if not name.lower().endswith(".fmdl"):
            continue
        for m in fmdl.Fmdl(data).meshes:
            if not len(m.tris) or not m.uvs:
                continue
            t = m.tris.reshape(-1)
            P.append(m.pos[t] * AXIS)
            nr = m.normal[t] * NAXIS if m.normal is not None else np.tile((0, 1, 0), (len(t), 1))
            N.append(nr.astype(np.float32))
            V.append(m.uvs[0][t, 1])
    if not P:
        return None
    pos, nrm, v = np.concatenate(P), np.concatenate(N), np.concatenate(V)
    centre = pos.mean(0)
    theta = np.arctan2(pos[:, 2] - centre[2], pos[:, 0] - centre[0])
    radius = float(np.hypot(pos[:, 0] - centre[0], pos[:, 2] - centre[2]).mean())
    span = float(np.ptp(theta))
    height = float(np.percentile(pos[:, 1], 98) - np.percentile(pos[:, 1], 2))
    slope = height / max(1e-3, float(np.ptp(v)))            # screen height incl. tilt is longer; good enough
    if span > np.pi:                                        # a ring: panels all the way round
        panels = max(1, int(round(2 * np.pi * radius / (max(height, 100.0) * 16 / 9))))
        u = (theta + np.pi) / (2 * np.pi) * panels
    else:                                                   # ordinary screens: one picture each
        panels = 1
        u = (theta - theta.min()) / max(span, 1e-6)
    tri_u = u.reshape(-1, 3)
    # a triangle that straddles the +-pi seam has to be unwrapped before it is assigned a panel
    wrap = (tri_u.max(1) - tri_u.min(1)) > panels / 2
    tri_u[wrap] = np.where(tri_u[wrap] < panels / 2, tri_u[wrap] + panels, tri_u[wrap])
    k = np.floor(tri_u.mean(1))
    tri_u = np.clip(tri_u - k[:, None], 0.0, 1.0)
    uv0 = np.stack([tri_u.reshape(-1), (v - v.min()) / max(1e-6, float(np.ptp(v)))], 1).astype(np.float32)
    uv1 = np.tile(lm_uv, (len(pos), 1)).astype(np.float32)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 1e-6, nrm / np.maximum(ln, 1e-6), (0, 1, 0)).astype(np.float32)
    mesh = make_mesh("pes_jumbotron", pos.astype(np.float32), nrm, uv0, uv1, np.arange(len(pos)).reshape(-1, 3))
    if mesh is None:
        return None
    log(f"jumbotron: {len(mesh.indices) // 3} triangles, {panels} panel(s)")
    return mesh, template.materials[sma]


def build_model(template, parts, opts, extra=None):
    """Splice converted parts into the stock template -> (rx3.Stadium, {texture name: (base path, alpha)})."""
    out = rx3.Stadium(props=list(template.props), collisions=list(template.collisions),
                      collision_group=template.collision_group)
    mat_index = {}
    tex_needed = OrderedDict()               # converted texture name -> [base path, needs alpha]

    def material(mat):
        key = (mat[0], tuple(mat[1]))
        if key not in mat_index:
            mat_index[key] = len(out.materials)
            out.materials.append((mat[0], list(mat[1])))
            for _slot, t in mat[1]:
                if t not in out.textures:
                    out.textures.append(t)
        return mat_index[key]

    new_groups = defaultdict(list)
    arrays = {key: part.arrays() for key, part in parts.items()}
    levels = {}
    if opts["bake"] and arrays:
        keys = list(arrays)
        offs = np.cumsum([0] + [len(arrays[k][0]) for k in keys])
        pos = np.concatenate([arrays[k][0] for k in keys])
        nrm = np.concatenate([arrays[k][1] for k in keys])
        opaque = [i for i, k in enumerate(keys) if k[0] not in ("Alpha", "ExteriorAlpha")]
        solid = np.concatenate([arrays[keys[i]][3] + offs[i] for i in opaque])
        inside = np.concatenate([np.full(len(arrays[keys[i]][3]), not is_exterior(keys[i])) for i in opaque])
        # Roofs over the stands would leave every seat black under a pure sky-visibility bake.
        # Real roofs let light through or bounce it back, so geometry above the top seat row
        # only blocks part of the sky light.
        roof_y = opts.get("roof_y")
        if roof_y is None:
            roof_y = pos[:, 1].min() + 0.7 * (pos[:, 1].max() - pos[:, 1].min())
        below = (pos[solid][:, :, 1] < roof_y).any(1)
        log(f"baking sky light for {len(pos)} vertices ({opts['bake_dirs']} directions, roof above {roof_y / 100:.0f} m) ...")
        vis_all = bake.sky_visibility(pos, nrm, solid, dirs=opts["bake_dirs"], log=log)
        vis_low = bake.sky_visibility(pos, nrm, solid[below], dirs=opts["bake_dirs"], log=log) if not below.all() else vis_all
        t = opts["roof_transmission"]
        bounce = opts["bounce"] * (0.5 + 0.5 * np.clip(-nrm[:, 1], 0, 1))
        vis = np.clip(bounce + (1 - opts["bounce"]) * (t * vis_low + (1 - t) * vis_all), 0, 1)
        # the roof skin itself: thin two-sided sheets self-shadow into speckle, so it gets a
        # smooth level instead (lit top, softly glowing underside)
        roof = pos[:, 1] >= roof_y
        vis[roof] = 0.5 + 0.4 * np.clip(nrm[roof, 1], 0, 1)
        log(f"  light level: mean {vis.mean():.2f}, dark (<0.2) {100 * (vis < 0.2).mean():.0f}%, bright (>0.6) {100 * (vis > 0.6).mean():.0f}%")
        # reference level: what open, up-facing stand surfaces (where the seats are) received
        tri_n = np.cross(pos[solid[:, 1]] - pos[solid[:, 0]], pos[solid[:, 2]] - pos[solid[:, 0]]).astype(np.float64)
        tri_a = np.linalg.norm(tri_n, axis=1)
        seatlike = (tri_n[:, 1] > 0.5 * tri_a) & below & (tri_a > 0) & inside
        if seatlike.any():
            tl = vis[solid[seatlike]].mean(1)
            o = np.argsort(tl)
            cw = np.cumsum(tri_a[seatlike][o]) / tri_a[seatlike].sum()
            opts["seat_level"] = float(tl[o][np.searchsorted(cw, 0.6)])
        for i, k in enumerate(keys):
            levels[k] = vis[offs[i]:offs[i + 1]]
    names = {}                                       # base path -> texture name (unique)
    for (group, shader, base), part in parts.items():
        if base not in names:
            name, n = tex_name(base), 1
            while name in tex_needed and tex_needed[name][0] != base:
                n += 1
                name = f"{tex_name(base)[:44]}_{n}"
            names[base] = name
        name = names[base]
        entry = tex_needed.setdefault(name, [base, False, 0])
        entry[1] = entry[1] or shader == "diffusealpha"
        entry[2] = max(entry[2], opts["tex_max_exterior"] if is_exterior((group,)) else opts["tex_max"])
        pos, nrm, uv, tris = arrays[(group, shader, base)]
        level = levels.get((group, shader, base))
        uv1 = np.full((len(pos), 2), 0.5, np.float32)
        uv1[:, 0] = bake.to_u(level if level is not None else 0.6)
        ma = (shader, [("ambientTexture", LM_NAME), ("diffuseTexture", name)])
        for ci, idx in enumerate(split_chunks(tris, pos, opts["max_tris"])):
            mesh = make_mesh(f"pes_{name}_{group}_{ci}", pos, nrm, uv, uv1, tris[idx])
            if mesh is not None:
                new_groups["Alpha" if group == "ExteriorAlpha" else group].append((mesh, ma))

    for gname, mesh, ma in (extra or []):
        new_groups[gname].append((mesh, (ma[0], list(ma[1]))))

    placed = set()

    def emit_new(gname):
        g = rx3.Group(gname, [])
        for mesh, ma in new_groups.get(gname, []):
            out.meshes.append(mesh)
            g.entries.append((len(out.meshes) - 1, material(ma), (0, 0, 0), (0, 0, 0)))
        placed.add(gname)
        if g.entries:
            out.groups.append(g)

    keep = set(opts["keep_groups"])
    for g in template.groups:
        if g.name in new_groups and g.name not in placed:
            emit_new(g.name)
        elif g.name in keep:
            ng = rx3.Group(g.name, [])
            only = opts["keep_shaders"].get(g.name)
            for mi, ma, _a, _b in g.entries:
                if only and template.materials[ma][0] not in only:
                    continue
                out.meshes.append(template.meshes[mi])
                ng.entries.append((len(out.meshes) - 1, material(template.materials[ma]), (0, 0, 0), (0, 0, 0)))
            out.groups.append(ng)
    for gname in list(new_groups):
        if gname not in placed:
            emit_new(gname)
    # the sky has to stay the last drawn group when present
    sky = [g for g in out.groups if g.name == "Sky"]
    out.groups = [g for g in out.groups if g.name != "Sky"] + sky
    out.refresh_bounds()
    return out, tex_needed


def part_area(part):
    pos, _n, _uv, tris = part.arrays()
    p = pos[tris].astype(np.float64)
    return float(np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2)


def weighted_percentile(values, weights, q):
    values, weights = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    order = np.argsort(values)
    cw = np.cumsum(weights[order]) / max(weights.sum(), 1e-9)
    return float(values[order][min(len(values) - 1, np.searchsorted(cw, q))])


def albedo_curve(template, stock_tex, parts, pes):
    """Tone curve (a, gamma) that maps PES albedo brightness onto the stock stadium's range.

    FIFA's stadium textures are far darker than PES's PBR albedo (white concrete is ~190 in
    PES, ~110 in FIFA) and the game's lighting is tuned for them; unscaled, stands blow out
    to white and saturated colours turn neon. The curve is fitted through the area-weighted
    brightness percentiles of both stadiums, so dark paint is barely touched and bright
    concrete comes down the most."""
    tex = {t.name: t for t in stock_tex.textures}
    sl, sa = [], []
    for g in template.groups:
        if g.name not in ("MainStadium", "Roof"):
            continue
        for mi, ma, _a, _b in g.entries:
            m = template.meshes[mi]
            name = dict(template.materials[ma][1]).get("diffuseTexture")
            if name not in tex or m.prim != 4:
                continue
            p = m.positions()[m.indices[:len(m.indices) // 3 * 3].reshape(-1, 3).astype(int)].astype(np.float64)
            sa.append(np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2)
            sl.append(textures.mean_colour(tex[name]).max())
    pl, pa = [], []
    for (group, _shader, base), part in parts.items():
        if group == "Alpha" or is_exterior((group,)):
            continue
        ft = pes.texture(base)
        lvl = 0
        while lvl + 1 < len(ft.mips) and max(ft.mip_size(lvl)) > 32:
            lvl += 1
        pa.append(part_area(part))
        pl.append(ft.rgba(lvl)[:, :, :3].reshape(-1, 3).mean(0).max())
    if len(sl) < 3 or len(pl) < 3:
        return None
    qs = np.arange(0.10, 0.96, 0.05)
    x = np.log(np.maximum([weighted_percentile(pl, pa, q) for q in qs], 4.0) / 255.0)
    y = np.log(np.maximum([weighted_percentile(sl, sa, q) for q in qs], 4.0) / 255.0)
    gamma, loga = np.polyfit(x, y, 1) if np.ptp(x) > 1e-3 else (1.0, float(np.mean(y - x)))
    gamma = float(np.clip(gamma, 0.5, 1.0))
    a = float(np.exp(np.mean(y - gamma * x)))
    curve = (a, gamma)
    probe = [60, 128, 190, 255]
    log("albedo: PES -> FIFA brightness " + ", ".join(
        f"{v}->{v * float(textures.tone_factor(np.array([v]), curve)[0]):.0f}" for v in probe)
        + f"  (a={a:.2f}, gamma={gamma:.2f})")
    return curve


def stock_light_levels(template, stock_tex):
    """Lightmap grey levels of the stock stadium for this time of day ->
    (shade level = area-weighted 10th percentile, seating level = mean over up-facing stand surfaces).
    The converted stadium's baked light is scaled to land on the same levels, which is what the
    game's exposure and crowd lighting are tuned for."""
    import render
    cache = render.TexCache(stock_tex, 128)
    lum, area, up = [], [], []
    for g in template.groups:
        if g.name not in ("MainStadium", "Roof"):
            continue
        for mi, ma, _a, _b in g.entries:
            m = template.meshes[mi]
            lm = cache.get(dict(template.materials[ma][1]).get("ambientTexture", ""))
            uv1 = render.decode_attr(m, "t1")
            if lm is None or uv1 is None or m.prim != 4:
                continue
            t = m.indices[:len(m.indices) // 3 * 3].reshape(-1, 3).astype(int)
            p = m.positions()[t].astype(np.float64)
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            a = np.linalg.norm(n, axis=1) / 2
            lum.append(render.sample(lm, uv1[t].mean(1))[:, :3].astype(np.float64).mean(1))
            area.append(a)
            up.append(n[:, 1] / np.maximum(2 * a, 1e-9) > 0.5)
    if not lum:
        return 20.0, 128.0
    lum, area, up = np.concatenate(lum), np.concatenate(area), np.concatenate(up)
    order = np.argsort(lum)
    cw = np.cumsum(area[order]) / area.sum()
    lo = float(lum[order][np.searchsorted(cw, 0.10)])
    seat = float((lum[up] * area[up]).sum() / max(area[up].sum(), 1e-9)) if up.any() else float(np.average(lum, weights=area))
    return lo, max(seat, lo + 10.0)


def build_textures(stock_tex, template, model, tex_needed, pes, opts, tod, curve=None):
    """-> rx3.TextureFile for one time of day."""
    out = rx3.TextureFile()
    referenced_by_template = {t for _sh, slots in template.materials for _s, t in slots}
    wanted = set(model.textures)
    for t in stock_tex.textures:
        if t.name in wanted or t.name not in referenced_by_template:
            out.textures.append(t)
    have = {t.name for t in out.textures}
    lo, seat = stock_light_levels(template, stock_tex)
    ref = opts.get("seat_level") or 0.5
    hi = lo + (seat - lo) / max(ref, 0.05)                 # grey level a fully open surface would get
    log(f"  ambient (time of day {tod}): stock shade {lo:.0f}, stock seating {seat:.0f} -> ramp {lo:.0f}..{min(hi, 255):.0f}"
        f" (seating light level {ref:.2f})")
    out.textures.append(bake.ramp_texture(LM_NAME, lo, hi))
    for name, (base, alpha, max_dim) in tex_needed.items():
        if name in have:
            continue
        out.textures.append(textures.from_ftex(name, pes.texture(base), max_dim or opts["tex_max"], alpha, curve))
    missing = wanted - {t.name for t in out.textures}
    if missing:
        raise RuntimeError(f"textures referenced but not written: {sorted(missing)[:8]}")
    return out


def convert(pes_root, stadium_id, out_dir, **overrides):
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    pes = PesStadium(pes_root)
    log(f"PES stadium {pes.st}: {pes.main_fpk}")
    template = rx3.Stadium.parse(stock.read(f"data/sceneassets/stadium/stadium_{stadium_id}.rx3"))
    sink = None
    pitch = template.group("Pitch")
    if pitch and pitch.entries:
        pp = np.concatenate([template.meshes[mi].positions() for mi, _ma, _a, _b in pitch.entries])
        sink = (float(pp[:, 0].min()), float(pp[:, 0].max()), float(pp[:, 2].min()), float(pp[:, 2].max()))
    if pes.exterior:
        log(f"exterior add-on: {pes.exterior}")
        if not any(g.name == opts["exterior_group"] for g in template.groups):
            opts["exterior_group"] = "MainStadium_NoShadowCast"
    parts, stats = convert_geometry(pes, opts, sink)
    quads, _img = crowd.load_pes(pes.root, pes.st)
    if quads:
        opts["roof_y"] = max(float(q["verts"][:, 1].max()) for q in quads) * 100.0 + 300.0
    log(f"converted {stats.used[0]} meshes / {stats.used[1]} triangles into {len(parts)} materials")
    for why, (n, t) in sorted(stats.skipped.items(), key=lambda kv: -kv[1][1]):
        log(f"  skipped {n:3} meshes / {t:7} triangles: {why}")
    extra = []
    jumbo = convert_jumbotron(pes, template, opts)
    if jumbo:
        extra.append(("Jumbotron", jumbo[0], jumbo[1]))
    model, tex_needed = build_model(template, parts, opts, extra)
    sdir = os.path.join(out_dir, "data", "sceneassets", "stadium")
    os.makedirs(sdir, exist_ok=True)
    data = model.build()
    mpath = os.path.join(sdir, f"stadium_{stadium_id}.rx3")
    open(mpath, "wb").write(data)
    v = sum(m.vcount for m in model.meshes)
    t = sum(len(m.indices) for m in model.meshes) // 3
    log(f"model: {len(model.meshes)} meshes, {v} vertices, {t} triangles, {len(model.materials)} materials, "
        f"{len(data) / 1e6:.1f} MB -> {mpath}")
    for g in model.groups:
        gt = sum(len(model.meshes[mi].indices) for mi, _m, _a, _b in g.entries) // 3
        log(f"  group {g.name:28} {len(g.entries):4} meshes {gt:8} triangles")
    written = [mpath]
    curve, have_curve = None, False
    for entry in stock.find(f"sceneassets/stadium/stadium_{stadium_id}_", "_textures.rx3"):
        tod = int(os.path.basename(entry).split("_")[2])
        stock_tex = rx3.TextureFile.parse(stock.read(entry))
        if not have_curve:
            curve = albedo_curve(template, stock_tex, parts, pes) if opts["albedo_match"] else None
            have_curve = True
        tf = build_textures(stock_tex, template, model, tex_needed, pes, opts, tod, curve)
        data = tf.build()
        tpath = os.path.join(sdir, os.path.basename(entry))
        open(tpath, "wb").write(data)
        log(f"textures (time of day {tod}): {len(tf.textures)} textures, {len(data) / 1e6:.1f} MB -> {tpath}")
        written.append(tpath)
    if opts["crowd"]:
        written += convert_crowd(pes, parts, stadium_id, out_dir, opts)
    if opts["glares"]:
        written += convert_glares(parts, stadium_id, out_dir, opts)
    return written


def convert_glares(parts, stadium_id, out_dir, opts):
    entries = stock.find(f"sceneassets/fx/glares_{stadium_id}_", ".lnx")
    roof_y = opts.get("roof_y")
    if not entries or roof_y is None:
        return []
    pos = np.concatenate([part.arrays()[0] for (group, _s, _t), part in parts.items()
                          if group != "Alpha" and not is_exterior((group,))])
    ring = glares.roof_ring(pos, roof_y)
    if ring is None:
        log("glares: no roof found, stock floodlight glares left in place")
        return []
    fdir = os.path.join(out_dir, "data", "sceneassets", "fx")
    os.makedirs(fdir, exist_ok=True)
    written, moved = [], 0
    for entry in entries:
        text, n = glares.relocate(stock.read(entry).decode("latin1"), ring)
        moved += n
        path = os.path.join(fdir, os.path.basename(entry))
        open(path, "w", encoding="latin1", newline="").write(text)
        written.append(path)
    log(f"glares: {moved} floodlight emitters moved to the roof edge in {len(written)} files "
        f"(ring radius {np.hypot(ring[:, 0], ring[:, 2]).mean() / 100:.0f} m, height {ring[:, 1].mean() / 100:.0f} m)")
    return written


def convert_crowd(pes, parts, stadium_id, out_dir, opts):
    quads, seat_image = crowd.load_pes(pes.root, pes.st)
    entries = stock.find(f"sceneassets/crowdplacement/crowd_{stadium_id}_", ".dat")
    if not quads or not entries:
        log("crowd: no PES audience areas or no stock crowd file; stock crowd left in place")
        return []
    for q in quads:
        q["verts"] = q["verts"] * AXIS
    # solid stand geometry the seats are snapped to
    P, T, base = [], [], 0
    for (group, _shader, _tex), part in parts.items():
        if group == "Alpha" or is_exterior((group,)):
            continue
        pos, _n, _uv, tris = part.arrays()
        P.append(pos)
        T.append(tris + base)
        base += len(pos)
    geom_pos, geom_tris = np.concatenate(P), np.concatenate(T)
    stock_recs, ver = crowd.read_dat(stock.read(entries[0]))
    budget = opts["crowd_max"] or len(stock_recs)
    pitch = opts["seat_pitch"]
    seats = crowd.generate(quads, geom_pos, geom_tris, seat_image, pitch, log=log)
    if len(seats["pos"]) > budget:
        pitch *= len(seats["pos"]) / budget * 1.01
        log(f"crowd: over the budget of {budget} seats, widening seat spacing to {pitch:.0f} cm")
        seats = crowd.generate(quads, geom_pos, geom_tris, seat_image, pitch, log=log)
    tier, bands = crowd.assign_tiers(quads, seats)
    log("crowd: tiers " + ", ".join(f"{i + 1}: {lo / 100:.1f}-{hi / 100:.1f} m" for i, (lo, hi) in enumerate(bands)))
    recs = crowd.build_records(seats, tier, stock_recs)
    cdir = os.path.join(out_dir, "data", "sceneassets", "crowdplacement")
    os.makedirs(cdir, exist_ok=True)
    written = []
    for entry in entries:
        path = os.path.join(cdir, os.path.basename(entry))
        open(path, "wb").write(crowd.write_dat(recs, ver))
        written.append(path)
    log(f"crowd: {len(recs)} seats (stock had {len(stock_recs)}) -> {cdir}")
    return written
