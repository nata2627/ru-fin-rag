"""Рабочие заметки (журнал решений, план, задания этапов) в репозиторий не попадают."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = re.compile(r"(^|/)(DECISIONS\.md|00-plan\.md|zadanie-etap-[^/]*\.md)$|(^|/)\.claude/")


def test_rabochih_zametok_net_v_git():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert [p for p in tracked.splitlines() if NOTES.search(p)] == []
