"""Витрина над журналом: одни запросы, два режима хранения, один результат.

Главная проверка здесь `test_both_modes_return_the_same_table`: она гоняет
запросы витрины по файлам JSONL через chdb и по тому же набору, залитому в
ClickHouse сервером, и сравнивает выходы как текст. Эта проверка требует
докера, поэтому помечена маркером `clickhouse` и пропускается, когда сервера
нет. Остальное проверяется в бесплатном режиме, который работает везде.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rufinrag import warehouse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_warehouses", ROOT / "scripts/compare_warehouses.py"
)
compare_warehouses = importlib.util.module_from_spec(spec)
sys.modules["compare_warehouses"] = compare_warehouses
spec.loader.exec_module(compare_warehouses)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    pytest.importorskip("chdb", reason="нет chdb, поставить: pip install -e '.[analytics]'")
    directory = tmp_path_factory.mktemp("journal")
    info = compare_warehouses.write_sample(directory, compare_warehouses.sample_events())
    return directory, info


@pytest.fixture(scope="module")
def free(sample):
    directory, _ = sample
    return warehouse.FilesWarehouse(directory)


def rows(text: str) -> list[dict[str, str]]:
    header, body = warehouse.parse_table(text)
    return [dict(zip(header, row, strict=True)) for row in body]


def test_mart_query_text_is_the_same_for_both_modes():
    """Разный только источник событий, сам запрос витрины один текст."""
    files = warehouse.FilesWarehouse(Path("/tmp/events")).source()
    server = warehouse.ServerWarehouse().source()
    assert files != server
    for name, body in warehouse.MART.items():
        assert body.strip() in warehouse.compose(name, files)
        assert body.strip() in warehouse.compose(name, server)


def test_traffic_by_day_splits_days_and_outcomes(free, sample):
    _, info = sample
    table = rows(free.run("traffic_by_day"))
    assert [row["day"] for row in table] == ["2026-10-21", "2026-10-22"]
    assert sum(int(row["queries"]) for row in table) == info["query_events"]
    for row in table:
        assert int(row["queries"]) == int(row["answered"]) + int(row["rejected"]) + int(
            row["unavailable"]
        )
        assert int(row["rejected"]) > 0 and int(row["unavailable"]) > 0


def test_stage_latency_shows_the_stages_that_actually_ran(free):
    table = {row["stage"]: row for row in rows(free.run("stage_latency"))}
    # Плотная ступень есть не у каждой связки, поэтому ее прогонов меньше, чем
    # лексических: ступени приходят из события, а не из колонок витрины.
    assert set(table) == {"bm25", "dense", "fusion", "rerank", "total"}
    assert int(table["rerank"]["runs"]) < int(table["dense"]["runs"]) < int(table["bm25"]["runs"])
    for row in table.values():
        assert 0 < float(row["p50_ms"]) <= float(row["p95_ms"]) <= float(row["max_ms"])


def test_rated_share_keeps_the_denominator_next_to_the_numerator(free):
    table = rows(free.run("rated_share"))
    for row in table:
        assert 0 < int(row["rated"]) < int(row["answered"])
    assert {"answered", "rated"} <= set(table[0])


def test_orphan_feedback_is_counted_not_dropped(free, sample):
    _, info = sample
    row = rows(free.run("feedback_match"))[0]
    assert int(row["feedback_events"]) == info["feedback_events"]
    assert int(row["orphans"]) == info["orphan_feedback"]
    assert int(row["matched"]) + int(row["orphans"]) == int(row["feedback_events"])
    assert int(row["up"]) + int(row["down"]) == int(row["feedback_events"])


def test_quantiles_are_exact_so_two_runs_agree(free):
    """Приближенный квантиль ClickHouse считает по выборке из потока.

    На нем два режима расходились бы по причине, которая не имеет отношения к
    хранению, поэтому витрина берет `quantileExact`.
    """
    assert "quantileExact" in warehouse.MART["stage_latency"]
    assert free.run("stage_latency") == free.run("stage_latency")


@pytest.mark.clickhouse
def test_both_modes_return_the_same_table(free, sample):
    directory, info = sample
    server = warehouse.ServerWarehouse()
    if not server.available():
        pytest.skip(f"нет ClickHouse на {server.url}: docker compose --profile logs up -d")

    assert server.load(directory) == info["lines"]

    for name in warehouse.MART:
        assert server.run(name) == free.run(name), f"режимы разошлись на {name}"
