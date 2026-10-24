"""Сравнение двух конфигураций: что в нем ломается тихо.

Сам прогон идет отдельной командой и стоит пять секунд
(`scripts/compare_configs.py`). Здесь проверяется то, чего в выводе не видно.

Первое: правило берется из закоммиченного файла задачи 6.1, а не из этого
скрипта. Подмени файл правила на тот, где не стоит отметки «записано до
данных», и сравнение обязано отказаться считать.

Второе: знак перевеса означает то, что написано. Знак появился только в задаче
6.2, и перепутанная сторона тихо объявила бы победителем проигравшего.

Третье: пустая арка. Интерливинг команд на одинаковых выдачах ничьей не дает, и
без замера этого шума доля информативных запросов читается как доля
запросов, несущих сигнал.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.interleaving import margin  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "compare_configs", ROOT / "scripts/compare_configs.py"
)
compare = importlib.util.module_from_spec(spec)
sys.modules["compare_configs"] = compare
spec.loader.exec_module(compare)

GOOD = [f"good{i}" for i in range(10)]
PLAIN = [f"plain{i}" for i in range(10)]
REL = {f"good{i}": 2 for i in range(10)}


def report() -> dict:
    return json.loads((ROOT / "results/compare-configs.json").read_text(encoding="utf-8"))


def test_rule_must_be_written_before_data(tmp_path, monkeypatch):
    broken = tmp_path / "sample-size.json"
    broken.write_text(json.dumps({"rule": {"written_before_data": False}}), encoding="utf-8")
    monkeypatch.setattr(compare, "RULE", broken)
    with pytest.raises(ValueError):
        compare.load_rule()


def test_rule_is_read_from_the_committed_file():
    rule = compare.load_rule()
    published = json.loads((ROOT / "results/sample-size.json").read_text(encoding="utf-8"))
    assert rule == published["rule"]
    assert report()["rule_written_at"] == published["written_at"]


def test_margin_sign_points_at_the_right_side():
    for seed in range(20):
        assert margin(PLAIN, GOOD, REL, random.Random(seed)) > 0
        assert margin(GOOD, PLAIN, REL, random.Random(seed)) < 0


def test_identical_runs_are_not_a_tie():
    """Свойство интерливинга команд, из-за которого нужна пустая арка.

    Общий фрагмент достается той стороне, которая его вытянула, поэтому на двух
    копиях одной связки перевес бывает не нулевым. Если это когда-нибудь
    перестанет быть правдой, пустая арка потеряет смысл и ее надо убрать, а не
    оставить как украшение.
    """
    редкая = {"good0": 2, "good3": 1}
    margins = [margin(GOOD, list(GOOD), редкая, random.Random(seed)) for seed in range(20)]
    assert any(value != 0 for value in margins), "шума отбора нет, пустая арка не нужна"
    assert any(value > 0 for value in margins) and any(
        value < 0 for value in margins
    ), "шум отбора вышел односторонним, то есть слияние предпочитает одну сторону"
    # Когда размечено все, что попало в десятку, перевеса нет вовсе: позиции
    # делятся ровно, и шум берется именно из редкой разметки.
    assert all(margin(GOOD, list(GOOD), REL, random.Random(seed)) == 0 for seed in range(20))


def test_null_arm_covers_a_coin_and_does_not_fire():
    null = report()["null_arm"]
    assert null["covers_half"], null["win_share_ci_wilson"]
    # Уровень ложной тревоги замерен: правило объявлено на 0,05, и на пустой
    # арке оно обязано срабатывать редко, а не на половине порядков.
    assert null["stopped_runs"] <= null["runs"] * 0.2, null["stopped_runs"]
    assert null["informative_mean"] > 0, "шум отбора не замерен, арка ничего не показывает"


def test_wilson_interval_holds_the_share():
    lo, hi = compare.wilson(438, 768)
    assert lo < 438 / 768 < hi
    assert compare.wilson(0, 0) == (0.0, 0.0)
    узкий = compare.wilson(500, 1000)
    широкий = compare.wilson(15, 30)
    assert широкий[1] - широкий[0] > узкий[1] - узкий[0]


def test_small_looks_are_skipped_not_counted():
    rule = {"looks": 14, "min_queries_per_look": 30, "alpha": 0.05, "seed": 1}
    margins = {f"q{i}": (1 if i % 2 else -1) for i in range(140)}
    out = compare.sequential(margins, rule, 1, 2000)
    skipped = [row for row in out["table"] if row["skipped"]]
    assert skipped, "ни один взгляд не пропущен, хотя первые взгляды меньше тридцати"
    assert all(row["informative"] < 30 for row in skipped)
    assert all(not row["crossed"] for row in skipped), "пропущенный взгляд не может решать"


def test_subsamples_are_not_averaged():
    data = report()
    assert data["живые"]["queries"] + data["синтетика"]["queries"] < data["все"]["queries"]
    # Решение по сервису берется с живых вопросов, а не со среднего по подвыборкам.
    assert "живы" in data["decision"]["why"]


def test_published_verdicts_follow_from_the_published_numbers():
    data = report()
    for item in data["decision"]["per_subsample"]:
        part = data[item["subsample"]]
        ждем = (
            part["ndcg"]["above_target"]
            and part["ndcg"]["p_permutation"] < 0.05
            and part["fixed_look"]["significant"]
        )
        assert item["established"] == ждем, item["subsample"]
        if not item["established"]:
            assert item["winner"] is None


def test_russian_plural_for_the_decision_line():
    assert compare.questions(1) == "1 вопрос"
    assert compare.questions(142) == "142 вопроса"
    assert compare.questions(991) == "991 вопрос"
    assert compare.questions(11) == "11 вопросов"
    assert compare.questions(846) == "846 вопросов"
