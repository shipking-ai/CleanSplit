#!/usr/bin/env bash
# Runs after tools/dev/gpu_queue.sh: one GPU job at a time (8 GB VRAM, see HANDOFF).
set -u
cd /c/AI/CleanSplit
Q="$1"   # gpu_queue.sh task output file
echo "[scnet] waiting for the overlap + vocal-recipe queue to finish"
until grep -q "ALL DONE" "$Q" 2>/dev/null; do sleep 30; done
echo "[scnet] separating 20 MUSDB songs with SCNet XL IHF, then scoring every stem"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/stem_ensemble_experiment.py --limit 20 --overlap 4 \
  > outputs/stem_ensemble.log 2>&1
sed -n '/MUSDB18-HQ, overlap/,$p' outputs/stem_ensemble.log
echo "[scnet] ALL DONE"
