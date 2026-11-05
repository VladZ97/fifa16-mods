# pes2fifa — PES 2021 / Football Life stadium → FIFA 16 (PS4, shadPS4)

Converts a PES stadium-server folder straight into FIFA 16 game files. No Blender, no
Windows tools; pure Python (numpy, Pillow, scipy) plus a 100-line C rasteriser that is
compiled on first use.

```
pes2fifa convert "<PES stadium folder>" --id <FIFA stadium id> --out <folder>
pes2fifa preview <folder> --id <id> [--views tv,high,top,corner,pitch,far] [--tod 1|3]
pes2fifa inspect "<PES stadium folder>"
fifamod install <folder> --name "My stadium"        # then it is a normal mod
```

`<PES stadium folder>` is the one that contains `Asset/model/bg/stNNN/...`.
`--id` is the FIFA stadium that gets replaced (database table `stadiums`; Camp Nou = 6,
Bernabéu = 2, Old Trafford = 1). That stock stadium is the template: its pitch, ad boards,
sky, sideline props and ball collision are reused.

## What gets written (game layout)

| file | content |
|---|---|
| `data/sceneassets/stadium/stadium_<id>.rx3` | PES geometry in FIFA's scene groups + kept stock groups |
| `data/sceneassets/stadium/stadium_<id>_<tod>_textures.rx3` | PES textures (DXT blocks passed through, darkened), stock textures still in use, light ramp |
| `data/sceneassets/crowdplacement/crowd_<id>_<tod>.dat` | one seat record per spectator, from PES audience areas |
| `data/sceneassets/fx/glares_<id>_<tod>_<n>.lnx` | stock floodlight glares moved to the new roof edge |

## How it works

- **Geometry** (`fpk.py`, `fmdl.py`, `convert.py`): the main package `stNNN.fpk` is unpacked,
  every FMDL mesh is read (LOD 0), character meshes and hidden/shadow-only meshes are dropped,
  meshes are merged per material and cut into chunks of at most 10k triangles (stock meshes
  stay below ~11.5k).
- **Coordinates**: `p_fifa = 100 * (x, y, -z)`. PES is left-handed with the camera on +Z, FIFA
  right-handed with the camera on -Z, so this is a reflection and the triangle order is kept.
  The usual Blender recipe (import, rotate 180°, export) produces a mirrored stadium; the seat
  lettering is the proof (see the note at the top of `convert.py`).
- **Materials**: opaque → `diffuseopaque`, blended / glass / cut-out → `diffusealpha` in the
  `Alpha` group, PES "no shadow" meshes → `MainStadium_NoShadowCast`. Two-sided PES meshes get
  explicit back faces.
- **Lighting** (`bake.py`): PES has no lightmaps. Sky light is baked per vertex (48 shadow-map
  directions) and stored in the second UV set; the "lightmap" is a 256x4 grey ramp, so the
  lookup behaves like vertex lighting. The ramp's levels are taken from the stock stadium's
  own lightmaps for each time of day, because the game's exposure and the crowd's brightness
  follow them. Roof-level geometry only blocks half of the light, otherwise fully roofed
  stadiums bake black.
- **Texture brightness**: PES albedo is far brighter than FIFA's stadium textures; the DXT
  colour endpoints are scaled so both stadiums have the same level.
- **Crowd** (`crowd.py`): `audiarea.bin` quads → rows snapped to the real steps of the stand
  (top-down height field per quad), seats along each row, colours from `au_seat.png`, facing
  angle `atan2(-fz, fx)`, supporter type copied from the nearest stock seat. The seat count is
  kept at the stock stadium's (63k for Camp Nou) by widening the seat spacing.
- **Jumbotron**: PES TV screens become the `Jumbotron` group with the stock material, cut
  into 16:9 panels.
- **Exterior** (`fox2.py`): stadium-server exterior add-ons (`AddOn/<name>/Global/common/demo/
  fixdemoobj/stNNN_bg/#Win/stNNN_h_all.fpk` + `.fpkd`) hold the real outside of the stadium:
  facade, ground, town blocks, trees, cars, each placed through `.fox2` scene files (one model,
  many locators). They go into `Exterior_NoShadowCast`, nearest first, within a triangle
  budget (`exterior_tris`, 450k) and the sky dome radius, with textures capped at 512 px.
  Without the add-on the main package only has a plain placeholder shell around the bowl.
- **Pitch edge**: FIFA draws its own pitch at y = 0, so PES ground inside that footprint is
  sunk 12 cm (two coplanar surfaces flicker), and the stock dirt decal around the pitch is
  dropped because it was shaped for the old stadium.
- **Files are verified by construction**: `rx3.py` re-builds stock stadium and texture files
  byte for byte (`python3 rx3.py <stadium.rx3> <textures.rx3>`).

## Checking a conversion without the game

`pes2fifa preview` renders the written files (not the intermediate data): geometry, UVs,
texture bindings, light ramp and crowd seats. A mirrored stadium, wrong seat heights or
missing textures show up here in a few seconds.

## Known limits

- Ball collision, sideline characters and benches are still the stock stadium's.
- The pitch is FIFA's own; PES pitch-side turf is kept only outside it.
- Exterior glow strips and distant skyline cards (unlit PES shaders) are skipped; trees use
  plain alpha blending.
- Lighting is ambient only (no baked sun direction); night uses the same bake with the
  stock night levels.
- PES banners, flags, corner flags and sideline staff are not converted.
