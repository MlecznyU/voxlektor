"""Regression table for the Claude hook guards (owner commits, owner approves API changes)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"
SCRIPT = ROOT / "scripts" / "api_snapshot.py"

BLOCKED = [
    "git commit -m x",
    "git -C . commit -m x",
    "git -c x=y commit -m x",
    "git -C . push",
    "git --no-pager commit",
    "git push origin main",
    "git reset --hard",
    "git stash",
    "git rebase main",
    "git checkout -- f",
    "git checkout HEAD -- f",
    'bash -c "git commit -m x"',
    "bash -c 'git push'",
    "/usr/bin/git commit",
    "git commit;echo done",
    "cd a && git stash",
    "git restore f",
    "git restore --staged f",
    "git -C . restore --source=HEAD f",
    "git clean -fd",
    "git clean -n",
    "git checkout -f main",
    "git checkout --force main",
    "git checkout -fb topic",
    "git switch -f main",
    "git switch --force main",
    "git switch --discard-changes main",
    "git -C . switch -f main",
    "uv run poe api-approve",
    'uv run poe "api-approve"',
    "uv run poe -C . api-approve",
    "uv run python scripts/api_snapshot.py --approve",
    "uv run python scripts/api_snapshot.py --ap",
    "uv run python scripts/api_snapshot.py --appr",
    "python -m scripts.api_snapshot --approve",
    "echo x > tests/architecture/api_snapshot.txt",
    "cp x tests/architecture/api_snapshot.txt",
    "sed -i s/a/b/ tests/architecture/api_snapshot.txt",
]

ALLOWED = [
    "git status",
    "git log --oneline",
    "git diff",
    "git checkout main",
    "git checkout -b topic",
    "git switch main",
    "git switch -c topic",
    "git log --format=%H",
    "git diff --stat",
    "git config core.hooksPath .githooks",
    "echo cleanup",
    "git -C . status",
    "uv run poe check",
    "uv run poe api-check",
    "echo commit",
    "python scripts/api_snapshot.py --check",
    "python scripts/api_snapshot.py --add",
    "cat tests/architecture/api_snapshot.txt",
]


def _run(command: str) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(  # noqa: S603
        [str(HOOKS / "pre_bash_guard.sh")],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("command", BLOCKED)
def test_guard_blocks(command: str) -> None:
    result = _run(command)
    assert result.returncode == 2, command
    assert result.stderr.strip()


@pytest.mark.parametrize("command", ALLOWED)
def test_guard_allows(command: str) -> None:
    assert _run(command).returncode == 0, command


@pytest.mark.parametrize("flag", ["--ap", "--appr", "--appro"])
def test_snapshot_tool_rejects_abbreviated_flags(flag: str, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.txt"
    snapshot.write_text("keep\n")
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), flag, "--snapshot", str(snapshot), "--module", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert snapshot.read_text() == "keep\n"
