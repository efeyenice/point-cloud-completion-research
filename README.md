# Point Cloud Completion Research

Does adding **per-point RGB/color** improve 3D point-cloud completion, and **under what conditions**? A
controlled, baseline-first study (PCN from scratch, three arms A/B/C).

**Latest result (2026-07-28):** the color-input harm **shrinks with data but doesn't flip** —
+6.6% → +4.0% worse CD as training grows 20 → 150 models/category — see the
[weekly report](./reports/2026-07-28-scale-curve/report.pdf).

- **Handoff / current state:** [CLAUDE.md](./CLAUDE.md)
- **Decisions:** [docs/adr/](./docs/adr/) · **Domain glossary:** [CONTEXT.md](./CONTEXT.md)
- **How we co-research with Claude (adopt this style):** [docs/WORKFLOW.md](./docs/WORKFLOW.md)
- **Weekly reports (numbers regenerated from the run artifacts):** [reports/](./reports/)

## Structure
```
CLAUDE.md          handoff / current state (AGENTS.md points here)
CONTEXT.md         domain glossary
docs/adr/          architecture decisions (0001–0008)
docs/WORKFLOW.md   how we co-research with Claude
notebooks/         Colab: dataGeneration (s1) -> partialGeneration (s4) -> pcnTraining (s5)
reports/           weekly research reports (LaTeX template + per-week source, figures, PDF)
tools/             pc_resilient.py, pc_hf_data.py (+ tests), mesh-sampling/
notes/             research journal (handoffs) + key finding
archive/           reading-phase + source materials
```

## Where things live (ADR-0008)
Code → this repo · datasets + run artifacts → HF [`efeyenice/pc-completion-data`](https://huggingface.co/datasets/efeyenice/pc-completion-data)
· live metrics → **Weights & Biases** · demo → HF Space `efeyenice/pc-completion-runs`. *(Google Drive retired.)*

## Notebooks (run on Colab, opened from GitHub)
`notebooks/dataGeneration.ipynb` (s1) → `partialGeneration.ipynb` (s4) → `pcnTraining.ipynb` (s5).

Every long run is **resumable**: it commits per-unit to HF and skips finished units on restart, so a Colab
disconnect costs one unit, not the whole run.
