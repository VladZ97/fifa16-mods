# FIFA 16 (shadPS4) mod manager

`fifamod` keeps every mod in its own folder and builds the shadPS4 `/download0` overlay
from the base patch (FIP "Manual Install") plus the enabled mods.

- Tool + registry + merged overlay: `~/FIFA16Mods/` (APFS; exFAT cannot hold symlinks)
- Mod folders:  `/Volumes/SanDiskSSD/FIFA16Mods/mods/<slug>/files/...` (game layout),
  `extras/` (readme, previews), `mod.json`
- Registry: `~/FIFA16Mods/mods.json` (order = priority, last wins)
- Overlay: `~/Library/Application Support/shadPS4/download/CUSA02138 -> ~/FIFA16Mods/merged`

```
fifamod                              interactive menu
fifamod install ~/Downloads/x.rar    (--dry-run to preview the mapping, --name "Nice name")
fifamod list | info <mod> | conflicts | status
fifamod enable|disable|remove <mod>  (number or name; rebuilds automatically)
fifamod move <mod> top|bottom|up|down|<n>
fifamod build                        rebuild after editing files by hand
```

Archives are mapped by folder name: `data/...` and `dlc/...` as-is; `game/`, `fonts/` ->
`data/ui/`; `overlays/`, `globalcomponents/` -> `data/ui/game/`; `kit/`, `heads/`, `ball/`...
-> `data/sceneassets/`; root `*.ini` -> `<game>-mods/` (served as /app0). `fonts.txt`
is merged as a union of lines across layers; a `_font.txt` next to a .ttf is treated as a
fonts.txt fragment. Placeholder `.rx3` files (<=32 bytes) are dropped because they crash
the PS4 build.

To go back to the plain patch: `fifamod disable` every mod, or point the download symlink
back at `/Volumes/SanDiskSSD/Manual Install`.

## Variants and optional parts

- Folders whose name starts with `_` (e.g. `_imgAssets`) are optional parts the author ships;
  they are installed by default, `--skip-optional` leaves them in `extras/`.
- Alternative files the author ships beside the main ones (e.g. `Overlay_9002/overlay_9002_NAME.big`)
  stay in `extras/`. Switch with
  `fifamod pick <mod> overlay_9002_NAME.big data/ui/game/overlays/overlay_9002.big` (rebuilds).
- Scoreboard mods replace the same `overlay_90xx.big` ids, so two of them overlap; the one
  lower in `fifamod list` wins. Enable only the one you want, or `fifamod move` it to the bottom.

## fonts.txt

A loose `data/ui/fonts/fonts.txt` replaces the packed one (an empty loose file falls back
to it). If the merged list lacks the 12 stock fonts the UI loads none of them and the game
crashes with a null write on a SportsUtil::AsyncCallback thread when the first scoreboard
draws. `fifamod build` therefore seeds the list from `stock/data/ui/fonts/fonts.txt`
(extracted from data_startup.big), then adds the base patch's and each mod's entries.

## Asset fixers (automatic on install)

`fifamod install` runs PS4 fixers on every installed file and records them in `mod.json`
("fixes"); `fifamod fix <mod>` or `fifamod fix base` re-runs them.

- **Hair layout**: PC `hair_*`/`hairlod_*` meshes store positions as 3 x float32 (stride 40);
  every stock PS4 hair uses 4 x float16 (stride 36). The PS4 build draws PC hair correctly
  most of the time but one hair path assumes the console layout and the mesh explodes across
  the pitch in replays/close-ups. `tools/hair_ps4.py` rewrites the vertex sections; the FIP
  base patch's 6,768 hair meshes are in the mod "PS4 hair layout (all facepack hair)".
- **Placeholder .rx3** files (<= 32 bytes) are dropped at install; they crash the PS4 build.

## Per-competition scoreboards (patched emulator)

Both custom emulator builds (Custom-metal, Custom-perf) carry a hook in `sceKernelOpen`
(file_system.cpp, "fifamod" comment): when the game loads a match it requests
`data/sceneassets/tournament/specifictournament_<team>_<comp>_<n>.rx3`; the hook remembers
`<comp>` (logged as `Scoreboard: competition <id>`, visible even in play mode) and, when the
game later opens `data/ui/game/overlays/*` or `globalcomponents/overlaycomponents_*`, serves
`download0/scoreboards/<comp>/<same path>` if it exists. `fifamod scoreboard <comp> <mod>`
assigns a mod (enabled or not) to a competition and `fifamod build` materialises the folder.
Unassigned competitions get the merged tree's normal overlays (the last enabled scoreboard
mod, or FIP's own if none is enabled; assigned-but-disabled mods still contribute their fonts).
The request cascades through fallbacks (`<team>_<comp>_<stadium>`, `0_<comp>_<stadium>`, ...
`0_0_0`); the hook ignores comp 0. The hook also reads the two teams from
`imgAssets/crest/stadium/stadium_<home>_<comp>_<stadium>_<away>.dds` and applies a rule derived
from `scoreboards/teamleague.txt` (team -> league, generated from the base database by
`tools/fifadb.py` at build): a competition id that is a league is used only when BOTH teams
belong to it (two Spanish clubs in a friendly yes, Barcelona vs Bayern no); an id with no league
teams is a tournament and is used whenever the match reports it. Nothing to configure per mod
beyond `fifamod scoreboard <id> <mod>`.
Ids seen so far: 53 = La Liga (kick-off uses the home team's league), 39 = MLS,
223 = UEFA Champions League in FIP V9 (tournament mode).

## Switching off part of a mod

`fifamod exclude <mod> <folder>` keeps the files on disk but leaves them out of the merged
overlay and the scoreboard folders (`--remove` re-includes). Example: the La Liga scoreboard
ships its author's dark club crests under `data/ui/imgAssets`; `fifamod exclude laliga
data/ui/imgAssets` keeps FIP's crests instead.

## Scoreboard crests (white in-match crests)

Scoreboard mods ship monochrome white crests (`_imgAssets/crest*/dark`) for their score bar.
They are DXT5 on PC; the PS4 build renders compressed UI DDS blank, so the `fix_dds_uncompressed`
fixer (tools/dds_ps4.py, pure Python BC1/BC3 decode) rewrites them to uncompressed 32-bit at
install. They are excluded from the global merge (`fifamod exclude laliga data/ui/imgAssets`) but
still go into `scoreboards/<comp>/`; the emulator serves `imgAssets/crest*` from there only while
a match is in progress (from the dressing request until a results / team-select / kick-off-hub
screen loads). The scoreboard itself keeps its competition until the next match loads: the
FluxHub component is also loaded at half time, so it must never be a reset signal.

## Balls

The FIP base patch has the low-poly stand-in ball meshes (UV layout mismatch -> atlas shows as
tiny splashes). The mod "FIP balls (PS4 fixed meshes+textures)" carries the 59 fixed meshes the
old Metal overlay linked (stripified / original high-poly from fip/_overwritten_by_mac_apply).
High-poly originals are fine on the Metal build; the Vulkan build needs strips <= ~9k indices.
