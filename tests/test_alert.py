"""Алерт по деградации метрики: откуда берется порог и что считается провалом.

Сам прогон учебной деградации идет отдельной командой и стоит три минуты
(`scripts/check_alert.py`). Здесь проверяется то, что ломается тихо: порог,
взятый не из замера, и правила лестницы, по которым исход считается сошедшимся.
Последнее важнее всего: если правила написаны так, что сходится всегда, то и
прогон ничего не доказывает.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rufinrag.pipeline import alert_threshold

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("check_alert", ROOT / "scripts/check_alert.py")
check_alert = importlib.util.module_from_spec(spec)
sys.modules["check_alert"] = check_alert
spec.loader.exec_module(check_alert)

THRESHOLD = 0.02


def write_threshold(root: Path, value: float, passed: bool = True) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results/alert-threshold.json").write_text(
        json.dumps({"threshold": value, "passed": passed}), encoding="utf-8"
    )


def rung(share: float, drop: float, fired: bool, text: str | None = None) -> dict:
    return {
        "share": share,
        "drop": drop,
        "drop_exact": drop,
        "relevant_removed": round(share * 1180),
        "relevant_total": 1180,
        "exit_code": 1 if fired else 0,
        "alert_fired": fired,
        "queries_scored": 991,
        "alert_text": text
        if text is not None
        else f"NDCG@10 упал на {drop:.4f} против эталона от 2026-10-23 "
        f"при пороге {THRESHOLD} на 991 размеченных вопросах",
    }


def test_threshold_comes_from_the_measurement(tmp_path):
    write_threshold(tmp_path, 0.035)
    assert alert_threshold(tmp_path) == 0.035


def test_failed_measurement_is_not_a_threshold(tmp_path):
    """Замер, уткнувшийся в потолок, молча заменять числом по умолчанию нельзя."""
    write_threshold(tmp_path, 0.05, passed=False)
    with pytest.raises(ValueError):
        alert_threshold(tmp_path)


def test_ladder_without_a_single_alert_is_a_failure():
    rungs = [rung(0.0, 0.0, False), rung(0.01, 0.0074, False)]
    assert check_alert.problems(rungs, THRESHOLD)


def test_empty_ladder_is_a_failure_with_a_message():
    """Пустая лестница раньше падала трейсбеком, а не своим сообщением."""
    assert check_alert.problems([], THRESHOLD) == ["лестница пустая: проверять нечего"]


def test_ladder_where_every_loss_fires_is_a_failure():
    """Половина проверки про ложную тревогу держится на молчащей ступени с потерей.

    Ступень без потери этого не показывает: там и падать нечему.
    """
    rungs = [rung(0.0, 0.0, False), rung(0.04, 0.0216, True), rung(0.08, 0.0446, True)]
    problems = check_alert.problems(rungs, THRESHOLD)
    assert any("не промолчала" in line for line in problems)


def test_rounded_drop_on_the_threshold_is_not_a_false_red():
    """Падение 0,02004 при пороге 0,02: алерт прав, и лестница обязана это принять."""
    edge = rung(0.04, 0.02, True)
    edge["drop_exact"] = 0.02004
    rungs = [rung(0.0, 0.0, False), rung(0.02, 0.0108, False), edge]
    assert check_alert.problems(rungs, THRESHOLD) == []


def test_silent_alert_above_the_threshold_is_a_failure():
    rungs = [rung(0.0, 0.0, False), rung(0.04, 0.0216, False)]
    problems = check_alert.problems(rungs, THRESHOLD)
    assert any("а алерта нет" in line for line in problems)


def test_alert_inside_the_threshold_is_a_failure():
    """Алерт, который кричит на шуме, бесполезен так же, как мертвый."""
    rungs = [rung(0.0, 0.0, False), rung(0.01, 0.0074, True)]
    problems = check_alert.problems(rungs, THRESHOLD)
    assert any("а алерт сработал" in line for line in problems)


def test_alert_on_the_whole_corpus_is_a_failure():
    rungs = [rung(0.0, 0.0, True), rung(0.04, 0.0216, True)]
    problems = check_alert.problems(rungs, THRESHOLD)
    assert any("на целом корпусе" in line for line in problems)


def test_alert_text_without_the_number_is_a_failure():
    rungs = [rung(0.0, 0.0, False), rung(0.04, 0.0216, True, text="качество упало")]
    problems = check_alert.problems(rungs, THRESHOLD)
    assert any("в тексте алерта нет" in line for line in problems)


def test_ladder_that_fires_exactly_above_the_threshold_passes():
    rungs = [
        rung(0.0, 0.0, False),
        rung(0.01, 0.0074, False),
        rung(0.02, 0.0108, False),
        rung(0.04, 0.0216, True),
        rung(0.08, 0.0446, True),
    ]
    assert check_alert.problems(rungs, THRESHOLD) == []


def test_measured_ladder_is_the_one_that_passed():
    """Отчет прогона и правила проверки сверяются друг с другом, а не живут рядом."""
    report = json.loads((ROOT / "results/alert-drill.json").read_text(encoding="utf-8"))
    assert report["passed"]
    assert check_alert.problems(report["rungs"], report["threshold"]) == []
