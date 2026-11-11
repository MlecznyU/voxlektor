#!/usr/bin/env bash
# Stop: run the fast gate when source or tests changed; feed failures back to Claude.
# A failure is reported once: the same failing output again stays silent.
# Only the failing step's section is shown, capped at 15 lines.
set -euo pipefail

cd "$(dirname "$0")/../.."

active="$(python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("stop_hook_active", False)).lower())')"
if [ "$active" = "true" ]; then
  exit 0
fi

if [ -z "$(git status --porcelain -- src tests)" ]; then
  exit 0
fi

last_file="$(git rev-parse --git-dir)/stop_check.last"

if out="$(uv run poe check-fast 2>&1)"; then
  rm -f "$last_file"
  exit 0
fi

# Keep only the section of the step that failed (from the last "Poe =>" line).
failed="$(printf '%s\n' "$out" | awk '/^Poe => /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}')"
# Digits are dropped before hashing: timings and file counts change between runs.
digest="$(printf '%s' "$failed" | tr -d '0-9' | shasum | cut -d' ' -f1)"

if [ -f "$last_file" ] && [ "$(cat "$last_file")" = "$digest" ]; then
  exit 0
fi
printf '%s\n' "$digest" > "$last_file"

printf '%s\n' "$failed" | head -n 15 >&2
exit 2
