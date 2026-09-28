#!/usr/bin/env bash
# Wait for the MUSDB18-HQ zip to be complete, verify it, extract test/, then score the first N songs.
set -u
cd "$(dirname "$0")/../.."
ZIP=data/musdb18hq/musdb18hq.zip
SIZE=22656664047
MD5=12d4f2ecd55245a4688754dd76363103
N=${1:-20}
until [ "$(stat -c %s "$ZIP" 2>/dev/null || echo 0)" -ge "$SIZE" ]; do sleep 30; done
sleep 10  # let curl close the file
echo "download complete: $(stat -c %s "$ZIP") bytes"
got=$(md5sum "$ZIP" | cut -d' ' -f1)
if [ "$got" != "$MD5" ]; then echo "MD5 MISMATCH: got $got expected $MD5 -- stopping"; exit 1; fi
echo "md5 ok"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/eval/musdb_eval.py extract || { echo "EXTRACT FAILED"; exit 1; }
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/eval/musdb_eval.py run --limit "$N" || { echo "RUN FAILED"; exit 1; }
echo "ALL DONE"
