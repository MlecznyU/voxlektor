#!/usr/bin/env bash
# PreToolUse (Edit|Write|MultiEdit):
#  - blocks edits of the owner-approved API artefacts,
#  - reminds about the architecture skill when a NEW module is created (never blocks).
set -euo pipefail

cd "$(dirname "$0")/../.."

f="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')"

case "$f" in
  *tests/architecture/api_snapshot.txt|*docs/protocol/v1.schema.json)
    echo "API changes need the owner's approval: file an API change request instead" >&2
    exit 2
    ;;
esac

case "$f" in
  *src/voxlektor/*.py)
    if [ ! -e "$f" ]; then
      printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"New module: follow skill voxlektor-architecture and add it to docs/architecture.md §2 in this change."}}'
    fi
    ;;
esac
exit 0
