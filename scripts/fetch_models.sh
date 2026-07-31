#!/usr/bin/env bash
# Download the pretrained detector weights (not stored in git).
#
#   KataCR dual detectors (MIT, wty-yy/KataCR) -- the accurate one
#   ClashRoyaleBuildABot ONNX (MIT, Pbatch/...) -- older, kept for comparison
#
# Usage: bash scripts/fetch_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/models vendor/KataCR/runs

gdrive () {  # $1 = file id, $2 = output path
  [ -f "$2" ] && { echo "have $2"; return; }
  curl -sL "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t" -o "$2"
  echo "downloaded $2"
}

gdrive 1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_ data/models/kcr_d1.pt
gdrive 1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD data/models/kcr_d2.pt
cp -n data/models/kcr_d1.pt vendor/KataCR/runs/detector1_v0.7.13.pt 2>/dev/null || true
cp -n data/models/kcr_d2.pt vendor/KataCR/runs/detector2_v0.7.13.pt 2>/dev/null || true

base="https://raw.githubusercontent.com/Pbatch/ClashRoyaleBuildABot/main/clashroyalebuildabot/models"
for f in side.onnx units_M_480x352.onnx; do
  [ -f "data/models/$f" ] || curl -sL "$base/$f" -o "data/models/$f"
done
echo "models ready in data/models/"
