#!/usr/bin/env python3
"""pes2fifa - convert a PES 2021 / Football Life stadium folder into a FIFA 16 stadium mod.

  pes2fifa.py convert <PES stadium folder> --id <FIFA stadium id> --out <folder>
                      [--tex-max 1024] [--max-tris 10000]
  pes2fifa.py preview <folder from convert> --id <id> [--views tv,high,top] [--tod 1] [--out <png dir>]
  pes2fifa.py inspect <PES stadium folder>          list models, materials and textures

<PES stadium folder> is the stadium-server style folder that holds Asset/model/bg/stNNN/...
The FIFA stadium with the given id is replaced; its pitch, ad boards, props and collision
are reused. Output is a game-layout tree (data/sceneassets/stadium/...).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("pes")
    c.add_argument("--id", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tex-max", type=int, dest="tex_max")
    c.add_argument("--max-tris", type=int, dest="max_tris")
    p = sub.add_parser("preview")
    p.add_argument("folder")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--views", default="tv,high,top")
    p.add_argument("--tod", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--size", default="1600x900")
    i = sub.add_parser("inspect")
    i.add_argument("pes")
    a = ap.parse_args()

    if a.cmd == "convert":
        import convert
        convert.convert(a.pes, a.id, a.out, tex_max=a.tex_max, max_tris=a.max_tris)
    elif a.cmd == "preview":
        from PIL import Image
        import render
        import rx3
        sdir = os.path.join(a.folder, "data", "sceneassets", "stadium")
        if not os.path.isdir(sdir):
            sdir = a.folder
        st = rx3.Stadium.parse(open(os.path.join(sdir, f"stadium_{a.id}.rx3"), "rb").read())
        tx = rx3.TextureFile.parse(open(os.path.join(sdir, f"stadium_{a.id}_{a.tod}_textures.rx3"), "rb").read())
        crowd_recs = None
        cpath = os.path.join(os.path.dirname(sdir), "crowdplacement", f"crowd_{a.id}_{a.tod}.dat")
        if os.path.exists(cpath):
            import crowd
            crowd_recs = crowd.read_dat(open(cpath, "rb").read())[0]
        out = a.out or os.path.join(a.folder, "preview")
        os.makedirs(out, exist_ok=True)
        w, h = map(int, a.size.split("x"))
        for view in a.views.split(","):
            size = (min(w, h), min(w, h)) if view == "top" else (w, h)
            img = render.render(st, tx, view, size, crowd_recs=crowd_recs)
            path = os.path.join(out, f"{view}_{a.tod}.png")
            Image.fromarray(img).save(path)
            print("wrote", path)
    elif a.cmd == "inspect":
        import convert
        pes = convert.PesStadium(a.pes)
        for model, f in pes.models():
            print(f"{model}: {len(f.meshes)} meshes, {sum(len(m.tris) for m in f.meshes)} triangles")
            for m in f.meshes:
                base = convert.base_texture(m.material)
                print(f"   {len(m.tris):7} tris  alpha={m.alpha_flags:3} shadow={m.shadow_flags:2} "
                      f"{m.material.shader:34} {base}")


if __name__ == "__main__":
    main()
