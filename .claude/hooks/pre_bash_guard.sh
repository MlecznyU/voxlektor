#!/usr/bin/env bash
# PreToolUse (Bash): the owner commits, and only the owner approves API changes.
set -euo pipefail

cmd="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')"

git_re='(^|[;&|(`]|[[:space:]])git([[:space:]]+[^;&|[:space:]]+)*[[:space:]]+(commit|push|reset|stash|rebase|checkout[[:space:]]+--)([[:space:]]|$)'
if [[ "$cmd" =~ $git_re ]]; then
  echo "the owner commits — report instead" >&2
  exit 2
fi

msg="API changes need the owner's approval: file an API change request instead"
if [[ "$cmd" =~ api_snapshot\.py.*--approve ]] || [[ "$cmd" =~ poe[[:space:]]+api-approve ]]; then
  echo "$msg" >&2
  exit 2
fi
if [[ "$cmd" == *api_snapshot.txt* ]]; then
  write_re='(>|(^|[^[:alnum:]_])(tee|cp|mv|sed|perl|truncate|rm|dd|install|python3?|node)([^[:alnum:]_]|$))'
  if [[ "$cmd" =~ $write_re ]]; then
    echo "$msg" >&2
    exit 2
  fi
fi
exit 0
