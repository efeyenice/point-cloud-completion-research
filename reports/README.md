# Weekly research reports

One LaTeX report per week, advisor-facing, honest and baseline-relative. This is the
**Report** step of the loop in `docs/WORKFLOW.md`, made concrete.

## Layout
```
reports/
  template/weeklyreport.sty     # shared style: title block, TL;DR / process boxes, colors
  <date>-<slug>/                # one directory per report, e.g. 2026-07-28-scale-curve/
    main.tex                    # the report (single file)
    make_figures.py             # pulls data from HF -> figures/ + generated/
    figures/                    # generated figure PDFs/PNGs (committed)
    generated/numbers.tex       # every number quoted in prose, as macros (committed)
    generated/runs_table.tex    # data tables (committed)
    Makefile                    # `make figures` (needs network once) + `make` (offline)
    report.pdf                  # the committed deliverable
```

## Rules that keep reports honest
- **Numbers are macros, never typed.** `make_figures.py` derives every quoted number from
  the run artifacts on HF and writes `generated/numbers.tex`; prose uses `\gapConefifty`
  etc., so text cannot drift from data.
- **Figures are scripted** from the same artifacts (no hand-edited images; qualitative
  panels are reused from the run outputs, recomposed only).
- **Claims stay conservative**: baseline-relative, seed-spread shown, caveats in the text,
  no extrapolation beyond the tested range.
- The **PDF is committed** so the record is readable without a TeX install.

## Building
```
cd reports/<week>
make figures   # regenerates figures/ + generated/ from HF (network, first time only)
make           # latexmk -> build/ -> report.pdf
```

## Starting a new week
1. Copy the latest week's directory; rename to `<date>-<slug>`.
2. Update the metadata block at the top of `main.tex` (`\reportweek`, `\reportdates`,
   `\reportquestion`, …).
3. Point `make_figures.py` at the new run artifacts; keep the numbers-as-macros rule.
4. Write, `make`, review the PDF, commit (including `report.pdf`).
