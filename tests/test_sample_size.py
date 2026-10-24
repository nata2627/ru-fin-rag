"""Правило остановки и размер выборки: что в нем ломается тихо.

Сам расчет идет отдельной командой и стоит три секунды
(`scripts/plan_sample_size.py`). Здесь проверяется то, чего в выводе не видно.

Первое: правило в опубликованном файле это то же правило, что в коде. Константы
объявлены до данных, и расхождение между кодом и файлом означало бы, что
правило поменяли после замера.

Второе: интерливинг не возвращает знак. Если бы он его возвращал, задача 6.1
уже знала бы победителя, и правило остановки писалось бы с подсмотренным
ответом.

Третье: поправка на подглядывание действительно поправка. Граница, равная 1,96
при четырнадцати взглядах, означала бы, что поправки нет, а в отчете она
заявлена.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "plan_sample_size", ROOT / "scripts/plan_sample_size.py"
)
plan = importlib.util.module_from_spec(spec)
sys.modules["plan_sample_size"] = plan
spec.loader.exec_module(plan)


def report() -> dict:
    return json.loads((ROOT / "results/sample-size.json").read_text(encoding="utf-8"))


def test_published_rule_matches_the_code() -> None:
    rule = report()["rule"]
    assert rule["alpha"] == plan.ALPHA
    assert rule["power"] == plan.POWER
    assert rule["looks"] == plan.LOOKS
    assert rule["min_queries_per_look"] == plan.MIN_LOOK
    assert rule["offline_target_effect"] == plan.OFFLINE_TARGET
    assert rule["online_target_excess"] == plan.ONLINE_TARGET
    assert rule["offline_grid"] == list(plan.OFFLINE_GRID)
    assert rule["online_grid"] == list(plan.ONLINE_GRID)
    assert rule["seed"] == plan.SEED


def test_merged_list_is_a_fair_draft() -> None:
    left = [f"a{i}" for i in range(20)]
    right = [f"b{i}" for i in range(20)]
    for seed in range(50):
        merged, owners = plan.merge_team_draft(left, right, random.Random(seed))
        assert len(merged) == plan.DEPTH
        assert len(set(merged)) == len(merged), "один и тот же фрагмент попал в список дважды"
        assert abs(owners.count(0) - owners.count(1)) <= 1, "одна команда набрала позиций больше"


def test_shared_documents_are_not_duplicated() -> None:
    left = ["x", "y", "z"]
    right = ["y", "x", "w"]
    merged, _ = plan.merge_team_draft(left, right, random.Random(0))
    assert sorted(merged) == ["w", "x", "y", "z"]


def test_margin_carries_no_sign() -> None:
    """Перевес одинаков при любом порядке конфигураций: знака в нем нет.

    Пара здесь нарочно односторонняя: правая выдача держит все размеченное, у
    левой нет ничего. Знаковая мера дала бы здесь +N и -N, модуль дает N и N.
    """
    left = [f"a{i}" for i in range(10)]
    right = [f"good{i}" for i in range(10)]
    rel = {f"good{i}": 2 for i in range(10)}
    for seed in range(20):
        forward = plan.interleaving_margin(left, right, rel, random.Random(seed))
        backward = plan.interleaving_margin(right, left, rel, random.Random(seed))
        assert forward == backward > 0


def test_identical_runs_give_no_information() -> None:
    same = [f"a{i}" for i in range(10)]
    rel = {"a0": 2, "a3": 1}
    assert plan.interleaving_margin(same, list(same), rel, random.Random(0)) == 0


def test_peeking_correction_raises_the_boundary() -> None:
    once = plan.pocock_boundary([1000], 20000, plan.SEED)
    fourteen = plan.pocock_boundary(plan.look_sizes(1000, plan.LOOKS), 20000, plan.SEED)
    # Один взгляд это обычный тест, и граница обязана сойтись с 1,96. Точного
    # совпадения не бывает: статистика на 1000 запросах шагает на 2/sqrt(1000),
    # граница встает на ближайшее наблюденное значение, у которого хвост уже не
    # толще 0,05, и промахивается вверх не больше чем на два таких шага.
    step = 2 / math.sqrt(1000)
    assert 1.96 <= once <= 1.96 + 2 * step
    assert fourteen > 2.4, "на четырнадцати взглядах граница обязана подняться"


def test_small_looks_are_skipped_not_corrected() -> None:
    assert plan.pocock_boundary([5, 10, 20], 1000, plan.SEED) == float("inf")


def test_required_sample_follows_the_square_of_the_effect() -> None:
    # Ровно вчетверо не выходит из-за округления вверх до целого запроса.
    assert abs(plan.offline_n(0.23, 0.01) / plan.offline_n(0.23, 0.02) - 4) < 0.01
    assert plan.online_fixed_n(0.05) > plan.online_fixed_n(0.10)


def test_published_plan_holds_the_declared_power() -> None:
    for entry in report()["online"]["plans"]:
        assert entry["power_simulated"] >= plan.POWER
        assert entry["false_positive_simulated"] <= plan.ALPHA + 0.01
        assert entry["boundary_z"] > entry["boundary_z_without_correction"]
        assert entry["n_max"] >= entry["fixed_sample_n"]
