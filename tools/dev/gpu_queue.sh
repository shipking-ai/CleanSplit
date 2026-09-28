#!/usr/bin/env bash
# Serialise the remaining GPU work: 8 GB of VRAM cannot hold two of these at once, and running them concurrently
# dropped the overlap-8 arm from 0.13x to 0.06x realtime. One at a time finishes both sooner.
set -u
cd /c/AI/CleanSplit
GPUTEST="$1"   # task output file of the running GPU test suite

echo "[queue] waiting for the GPU test suite to finish"
until grep -qE "passed|failed|error" "$GPUTEST" 2>/dev/null; do sleep 20; done
echo "[queue] gpu tests: $(grep -oE '[0-9]+ (passed|failed)[^,]*' "$GPUTEST" | tail -2 | tr '\n' ' ')"

echo "[queue] overlap 8: separating then scoring all three arms"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/overlap_experiment.py --limit 20 --overlaps 4 8 \
  > outputs/overlap_experiment.log 2>&1
echo "[queue] overlap done: $(grep -c '^sw_tta_ov8' outputs/overlap_experiment.log) ov8 songs, $(tail -1 outputs/overlap_experiment.log)"

echo "[queue] vocal recipe: is ep317 without TTA costing us the flagship stem?"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/vocal_best_recipe.py --limit 20 --overlap 4 \
  > outputs/vocal_recipe.log 2>&1
echo "[queue] vocal recipe done: $(tail -1 outputs/vocal_recipe.log)"
echo "[queue] ALL DONE"
