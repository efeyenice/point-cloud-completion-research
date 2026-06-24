# Read ShapeNet data from the shared Drive (supersedes the "download-to-local" half of ADR-0003)

The ShapeNet category zips are stored once in the shared Drive at
`Ortak/data/shapenetcore/<synset>.zip`. The data-generation notebook reads those zips
**directly from the mounted shared folder** and selectively extracts only the ~20 models per
category it needs to local disk — it does **not** re-download from HuggingFace and does **not**
copy whole zips. The HuggingFace download is kept as an **optional, normally-unused cell** that
only (re)populates the shared folder if a zip goes missing; a missing zip otherwise **hard-stops**
with a helpful message rather than silently re-downloading.

Why this supersedes ADR-0003's "download to local each session": the decisive reason is that
**teammate Pelin has no HuggingFace access** — ShapeNetCore is gated (account + license + token).
Reading the shared zips lets her run the notebook with zero HF setup.

Trade-offs accepted: it does not save much wall-clock time (reading a few GB off the Drive mount
is comparable to downloading); and all 58 zips (~24 GB) currently sit in the shared folder while
only 3 are used (trimming the rest is optional housekeeping). Reliability note: reading a zip's
central directory + members over the Google Drive FUSE mount is generally fine at this scale; if
it proves slow/flaky, the fallback is to copy the 3 zips to local disk first, then extract.
