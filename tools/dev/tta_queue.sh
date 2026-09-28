#!/usr/bin/env bash
# Runs last in the GPU chain: one job at a time (8 GB VRAM).
set -u
cd /c/AI/CleanSplit
Q="$1"   # scnet_queue.sh task output file
echo "[tta] waiting for the SCNet experiment to finish"
until grep -q "ALL DONE" "$Q" 2>/dev/null; do sleep 30; done
echo "[tta] separating 20 songs with SW at overlap 4 WITHOUT tta (1 pass instead of 3)"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/cache_arm.py sw_ov4 --separator bs_roformer_sw \
  --overlap 4 --no-tta --limit 20 > outputs/tta_cache.log 2>&1
echo "[tta] $(tail -1 outputs/tta_cache.log)"
echo "[tta] is TTA redundant with overlap 4? baseline = no tta"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/fullband_check.py sw_ov4 sw_tta_ov4 \
  --baseline sw_ov4 --limit 20 --out outputs/_benchmarks/musdb18hq_tta_vs_overlap.json \
  > outputs/tta_vs_overlap.log 2>&1
sed -n '/MUSDB18-HQ full band/,$p' outputs/tta_vs_overlap.log
echo "[tta] ALL DONE"
