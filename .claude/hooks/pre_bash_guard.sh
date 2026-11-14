#!/usr/bin/env bash
# PreToolUse (Bash): the owner commits, and only the owner approves API changes.
#
# This guard is a process rail, not a security boundary: it catches the usual spellings of
# state-changing git commands and snapshot approvals so an agent stops and reports instead. A
# determined caller can always get around a regex (aliases, scripts, other interpreters); the
# real protection is review by the owner, who commits.
set -euo pipefail

cmd="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')"

# Blocked: commit, push, reset, stash, rebase, restore, clean; `checkout ... --`; and
# `checkout`/`switch` with -f / --force / --discard-changes (also in a short-flag cluster: -fb).
git_re='(^|[^[:alnum:]_-])git([[:space:]]+[^;&|[:space:]]+)*[[:space:]]+((commit|push|reset|stash|rebase|restore|clean)([^[:alnum:]_-]|$)|(checkout|switch)([[:space:]]+[^;&|[:space:]]+)*[[:space:]]+(--|--force|--discard-changes|-[[:alpha:]]*f[[:alpha:]]*)([^[:alnum:]_-]|$))'
if [[ "$cmd" =~ $git_re ]]; then
  echo "the owner commits — report instead" >&2
  exit 2
fi

msg="API changes need the owner's approval: file an API change request instead"
if [[ "$cmd" =~ api_snapshot.*--ap ]] || [[ "$cmd" == *api-approve* ]]; then
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
