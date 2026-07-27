# Colab research ops: HF-native storage, resumable per-unit runs, W&B tracking, git as source of truth

The 200-model/category data-gen (`partialGeneration`, `RUN_TAG="s4"`, `N_PER_CAT=200`) kept **dying on
Colab disconnects with nothing durable written until the very end** → full re-runs, repeatedly. Root
causes are structural, not Colab's fault (disconnects are unavoidable): results accumulate in **RAM**
(manifest/rows lists), intermediates live in **ephemeral `/content`**, the pipeline is **stage-by-stage**
(nothing is complete until a whole stage is), and run state is in RAM (hence the `NameError: SYNSETS` on
a kernel restart). Meanwhile Drive `Ortak` had become a **catch-all** (source data + `runs_s5` + the
notebooks themselves), and notebooks **drift** between git and Drive (the stale/misnamed branch and
out-of-date CLAUDE.md this month). As experiments scale (the s6 curve = many runs), this is unsustainable.

**Decision:** adopt an HF-native, resumable, tracked operating model. One home for each kind of artifact:

| Artifact | Home | How |
|---|---|---|
| Code + notebooks | **git** | Colab↔GitHub native open/save; **no notebooks on Drive**; strip outputs before commit (nbstripout) so the repo doesn't bloat (the 10 MB notebook goes away) |
| Decisions / rationale | **git** | `docs/adr/`, `CONTEXT.md`, `notes/handoff-*` (unchanged — working well) |
| Generated datasets | **HF dataset repo** | one **`{model_id}.npz`** per model (24 partials + GT as arrays); `huggingface_hub.CommitScheduler` background-syncs the local folder every few min |
| Run checkpoints / results | **HF** | `CommitScheduler` on the run folder (best/latest `.pt`, `metrics.csv`, panels) |
| Live metrics / curves | **W&B** | `wandb.init/log` in the loop — cloud dashboards survive disconnects + compare runs; `metrics.csv → HF` is the self-owned backup (no lock-in) |
| Source data (ShapeNetCore) | **HF** | already (`snapshot_download`) |
| Scratch / intermediates | **Colab local disk** | ephemeral by design; never the source of truth |
| Drive | **retired / optional mirror** | no longer a dependency |

**The resilience pattern (every long Colab job — data-gen AND training):**
1. **Per-unit atomic commit** to durable storage — a unit is "done" only once its bytes are on HF, never
   "it was in RAM once." Unit = one model (data-gen) / one checkpoint interval (training).
2. **Idempotent resumable driver** — re-running the driver cell skips units already present in the HF
   repo. That *is* the entire recovery procedure after a disconnect.
3. **Live-sync** via `CommitScheduler` (loss window ≈ the sync interval, ~minutes).
4. **Anti-idle** keep-alive to reduce disconnect *frequency* (secondary; resumability is the real fix).

Training already has `run_training(resume=True)` (latest.pt) + the curve driver's skip-recorded rows;
this extends it with a `CommitScheduler` on the run root and W&B logging. **Data-gen needs the rewrite**
to the per-model-atomic + `CommitScheduler` pattern — the acute fix, done first.

## Considered options / the honest tradeoffs
- **Stay on Drive, just add resilience.** Rejected: leaves the coupling and the drift; Drive FUSE is slow
  on many small files (ADR-0003); "too Drive-dependent" was the explicit complaint.
- **Hybrid (Drive live-checkpoints, HF final artifacts).** Rejected as the *target* (still Drive mid-run),
  but kept as the **fallback** if `CommitScheduler` proves too heavy for large data pushes.
- **Parquet / WebDataset data format.** Deferred: `npz`-per-model is trivially atomic and needs the least
  loader change now; parquet is a later "release" packaging for public sharing.
- **Self-owned ledger instead of W&B.** Viable and lock-in-free, but no live dashboard / cross-run view;
  W&B chosen for the numerical+visual, disconnect-proof tracking the advisor asked for, with the HF
  `metrics.csv` mirror as the escape hatch.

## Consequences
- A disconnect costs **one unit**, not the whole run; recovery = re-run the same cell.
- New deps/secrets: `HF_TOKEN` + `WANDB_API_KEY` in Colab secrets; `huggingface_hub`, `wandb`.
- HF repos to create (private): a **dataset** repo for generated data, a repo for **run artifacts**.
- `PairDataset` switches from reading PLYs to reading the per-model `npz`.
- **Watch:** `CommitScheduler` cadence vs HF rate limits on large data pushes — coarsen the interval or
  fall back to hybrid if it bites. Log what a crash could still lose (the sub-interval window).
