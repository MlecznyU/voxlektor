#!/usr/bin/env bash
# Stop: run the fast gate when source or tests changed; feed failures back to Claude.
set -euo pipefail

cd "$(dirname "$0")/../.."

active="$(python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("stop_hook_active", False)).lower())')"
if [ "$active" = "true" ]; then
  exit 0
fi

if [ -z "$(git status --porcelain -- src tests)" ]; then
  exit 0
fi

if ! out="$(uv run poe check-fast 2>&1)"; then
  printf '%s\n' "$out" | tail -n 40 >&2
  exit 2
fi
exit 0
