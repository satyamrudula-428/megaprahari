#!/bin/sh
# Create the git-ignored data/ layout described in documentation/plan/DATA_SETUP.md (task B6). Safe to re-run.
# Usage: sh tools/make_data_dirs.sh [root]      (default root: ./data)
set -eu
ROOT=${1:-data}
for d in inputs inputs/tir landing/hem landing/met ledger atlas work models outbox; do
  mkdir -p "$ROOT/$d"
done
echo "data layout ready under $ROOT/:"
(cd "$ROOT" && find . -type d | sort)
