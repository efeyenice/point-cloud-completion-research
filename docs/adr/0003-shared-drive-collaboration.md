# Shared-Drive collaboration setup (Colab + "Shared with me")

> **Status:** the *data-location* half of this ADR (download fresh to local each session) is
> **superseded by ADR-0004** — data is now read from the shared `Ortak/data/shapenetcore/`.
> The Drive shortcut/mount setup and the outputs-to-Ortak parts below still apply.

The project's Google Drive folder "Point Cloud Completion" (containing `Efe/`, `Ortak/`,
`Pelin/`) is a **"Shared with me" folder, NOT a Google Workspace Shared Drive.** Colab does
not auto-mount "Shared with me" folders, so each teammate must add a one-time shortcut
(Drive web → Shared with me → right-click the folder → Organize → Add shortcut to Drive →
My Drive). After that it resolves at the same path for everyone —
`/content/drive/MyDrive/Point Cloud Completion/Ortak` — and the data-generation notebook
reads/writes that shared path.

Data is downloaded fresh from HuggingFace (only the 3 category zips) to the **local** Colab
disk each session — not stored in the shared folder — keeping shared storage tiny and making
runs reproducible from a fixed source. Only the small outputs (colored PLYs + `manifest.csv`
+ `errors.log`, ~100–200 MB) are written to `Ortak/colored_pc/`, which doubles as the
transfer mechanism: nothing is moved by hand; both teammates just see the same files.

Why recorded: the manual shortcut step is invisible in the code and will silently break a
teammate's run if skipped (`MyDrive` would resolve to the wrong account); and "download to
local, not the shared Drive" is a deliberate choice (vs caching the big zips in the shared
folder) worth remembering. Note: CloudCompare's point sampling has no CLI seed, so re-runs
yield *equivalent* not byte-identical clouds — for identical data a teammate reads the PLYs
already in `Ortak/` rather than regenerating.
