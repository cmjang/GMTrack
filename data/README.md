# Data layout

The JSON files under `data/` are organized by lifecycle. Motion payloads remain in
the shared `motions_*` directories and are not duplicated when a manifest moves
between lifecycle directories.

| Directory | Purpose |
|---|---|
| `datasets/` | All raw and converted motion payloads, grouped by dataset/source |
| `current/` | Active formal training source and its validated Stage-II splits |
| `test/` | Held-out evaluation manifests; never include them in training |

For formal runs, start at [`current/README.md`](current/README.md). Intermediate
rebuild manifests, clearance reports, and selection reports are reproducible output
and are intentionally not retained under `data/`; tools write them under `logs/`.

All conversion commands and manifests now reference `datasets/` directly. The former
top-level `raw` and `motions_*` compatibility names have been removed.

Manifest clip paths are relative to the directory containing the JSON file. Keep
that in mind when moving manifests between `data/current` and `data/test`; nesting a
manifest one level deeper requires rewriting its clip paths.
Never edit or relocate a strict Stage-II four-file bundle without regenerating its
shared artifact hash.
