#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit): lint, format and type-check the edited Python file.
set -euo pipefail

cd "$(dirname "$0")/../.."

f="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')"

case "$f" in
  *.py) ;;
  *) exit 0 ;;
esac
case "$f" in
  */src/*|src/*|*/tests/*|tests/*|*/scripts/*|scripts/*) ;;
  *) exit 0 ;;
esac
[ -f "$f" ] || exit 0

if ! out="$(uv run ruff check --fix "$f" 2>&1 && uv run ruff format "$f" 2>&1 && uv run pyright "$f" 2>&1)"; then
  printf '%s\n' "$out" >&2
  exit 2
fi
exit 0
