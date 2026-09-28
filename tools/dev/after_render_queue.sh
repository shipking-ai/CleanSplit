#!/usr/bin/env bash
# Runs the two outstanding GPU experiments once the user's own songs have finished rendering.
# Strictly one job at a time: 8 GB of VRAM cannot hold two, and running two concurrently dropped throughput
# from 0.13x to 0.06x realtime (HANDOFF, 2026-09-28).
set -u
cd /c/AI/CleanSplit
echo "[chain] waiting for the user-song render to finish"
until grep -q "ALL DONE" outputs/user_render.log 2>/dev/null; do sleep 30; done

echo "[chain] SCNet: does a 4th-stem ensemble partner help drums/bass/other?"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/stem_ensemble_experiment.py --limit 20 --overlap 4 \
  > outputs/stem_ensemble.log 2>&1
sed -n '/MUSDB18-HQ, overlap/,$p' outputs/stem_ensemble.log

echo "[chain] TTA redundancy: is TTA worth keeping for the SW member at overlap 4?"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/cache_arm.py sw_ov4 --separator bs_roformer_sw \
  --overlap 4 --no-tta --limit 20 > outputs/tta_cache.log 2>&1
echo "[chain] $(tail -1 outputs/tta_cache.log)"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/fullband_check.py sw_ov4 sw_tta_ov4 \
  --baseline sw_ov4 --limit 20 --out outputs/_benchmarks/musdb18hq_tta_vs_overlap.json \
  > outputs/tta_vs_overlap.log 2>&1
sed -n '/MUSDB18-HQ full band/,$p' outputs/tta_vs_overlap.log

# The 8-unit tier (ensemble at overlap 4, no TTA) becomes measurable the moment sw_ov4 exists: docs/04 section 14.12.
echo "[chain] 8-unit tier now measurable from the new sw_ov4 arm"
echo "[chain] ALL DONE"
