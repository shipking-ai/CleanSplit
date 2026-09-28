#!/usr/bin/env bash
# The running vocal_best_recipe.py compiled its scoring code BEFORE the paired-statistics fix, so its own printout
# would be the unpaired form. Wait for its cache to complete, then re-run scoring from cache (CPU only, no GPU) so the
# verdict is computed by the corrected, paired, floored rule.
set -u
cd /c/AI/CleanSplit
until [ "$(ls data/musdb_cache/ep317_tta_ov4 2>/dev/null | wc -l)" -ge 20 ]; do sleep 20; done
echo "[verdict] ep317_tta_ov4 complete; re-scoring paired"
sleep 30   # let the GPU process finish writing the last file and release the model
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/vocal_best_recipe.py --limit 20 --overlap 4 2>&1 | grep -vE ": done$|: cached$"
echo "[verdict] DONE"
