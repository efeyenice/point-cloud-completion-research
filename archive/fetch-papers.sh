#!/usr/bin/env bash
# Re-download all source papers into papers/ from arXiv.
# The PDFs are git-ignored (240 MB, re-downloadable); this script repopulates them.
# Usage:  ./fetch-papers.sh
set -euo pipefail
cd "$(dirname "$0")/papers"

# arXiv-id  filename  (see ../READING-LIST.md for what each is and why it matters)
papers=(
  "1612.00593 PointNet"
  "1706.02413 PointNet++"
  "1706.03762 AttentionIsAllYouNeed"
  "1712.07262 FoldingNet"
  "1801.07829 DGCNN"
  "1808.00671 PCN"
  "2012.09164 PointTransformer"
  "2108.08839 PoinTr"
  "2108.04444 SnowflakeNet"
  "2301.04545 AdaPoinTr"
  "2404.06814 ComPC"
  "2502.19896 GenPC"
  "2605.25553 ComPose"
  "2406.15811 PointDreamer"
  "2307.14726 P2C"
  "2006.11239 DDPM"
  "2104.03670 PVD"
  "2112.03530 PDR"
  "2104.05666 ViPC"
  "2209.09552 XMFnet"
  "2407.02887 EGIInet"
  "2412.08271 CLIP-PointCompletion"
  "2210.05891 JointColorSemanticSceneCompletion"
  "2404.08312 GPN-GenerativePointNeRF"
)

for entry in "${papers[@]}"; do
  id="${entry%% *}"; name="${entry##* }"
  if [ -f "${name}.pdf" ]; then
    echo "✓ ${name}.pdf (exists)"
  else
    echo "↓ ${name}.pdf  <- arxiv.org/pdf/${id}"
    curl -sL -o "${name}.pdf" "https://arxiv.org/pdf/${id}"
    sleep 1
  fi
done
echo "Done. $(ls -1 *.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs in papers/."
