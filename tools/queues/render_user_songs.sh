#!/usr/bin/env bash
# Re-separate the user's own three songs at the current best-measured settings and write listenable stems.
#
# Source audio is not in the repo, but every earlier run saved its exact input as outputs/<recipe>/<song>/original.wav,
# so these are the same signals the 2026-09-17 renders used -- only the recipe changes.
#
# Extra flags for the recipe under test are passed through as "$@" (e.g. --ep317-tta once that lands).
set -u
cd /c/AI/CleanSplit
OUT=outputs/best_render
mkdir -p "$OUT"
EXTRA=("$@")   # captured here: inside render(), "$@" would be render's own args, not the script's
render () {
  local src="$1" name="$2"
  if [ ! -f "$src" ]; then echo "[render] MISSING $src"; return; fi
  if [ -d "$OUT/$name/stems" ]; then echo "[render] $name already done, skipping"; return; fi
  echo "[render] $name  <- $src"
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m cleansplit.cli.main separate "$src" \
      --out "$OUT/$name" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tail -4
}
render "outputs/ensemble/Concrete_Crown/original.wav"          "Concrete_Crown"
render "outputs/ensemble/After_2/original.wav"                 "After_2"
render "outputs/ensemble_demucs/Too_Much_On_My_Plate/original.wav" "Too_Much_On_My_Plate"
echo "[render] ALL DONE -> $OUT"
