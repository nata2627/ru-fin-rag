"""Страница дашборда: числа витрины, пустой журнал и чужой текст в разметке.

Три вещи, которые должны работать без рук. Страница собирается по собранному
набору событий и печатает то же, что считает витрина. Страница собирается на
пустом журнале и честно говорит, что данных нет, вместо падения или выдуманных
чисел: до первой выгрузки с хостинга журнал пуст всегда, а красный прогон
остановил бы часы приемки автономности. И значение из журнала попадает в
разметку как данные: текст запроса в открытый сервис присылает кто угодно.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from rufinrag import dashboard

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_dashboard = _load("build_dashboard")
compare_warehouses = _load("compare_warehouses")


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    pytest.importorskip("chdb", reason="нет chdb, поставить: pip install -e '.[analytics]'")
    journal = tmp_path_factory.mktemp("journal")
    compare_warehouses.write_sample(journal, compare_warehouses.sample_events())
    out = tmp_path_factory.mktemp("site")
    report = build_dashboard.build(journal, out, live=False, source=build_dashboard.SAMPLE_SOURCE)
    return report, (out / "index.html").read_text(encoding="utf-8"), out


@pytest.fixture
def empty(tmp_path):
    return build_dashboard.build(
        tmp_path / "journal", tmp_path / "site", live=True, source=build_dashboard.JOURNAL_SOURCE
    )


def test_page_prints_what_the_mart_counted(sample):
    """Каждое число страницы это ячейка витрины, а не счет рядом с ней."""
    report, page, out = sample
    mart = json.loads((out / "mart.json").read_text(encoding="utf-8"))
    cells = {
        dashboard.number(value)
        for model in mart.values()
        for row in model["rows"]
        for value in row.values()
    }
    from_mart = [value for value in report["values"] if value["source"] in mart]
    assert from_mart, "ни одно число страницы не пришло из витрины"
    for value in from_mart:
        # Доля печатается со знаменателем («24 из 96»), и обе половины это
        # ячейки той же модели.
        parts = value["text"].split(" из ") if " из " in value["text"] else [value["text"]]
        for part in parts:
            assert part in cells, f"{value['label']}: {part} нет в выводе витрины"


def test_rated_share_keeps_the_denominator_on_the_page(sample):
    report, page, _ = sample
    shares = [value for value in report["values"] if value["label"].startswith("rated_share.share")]
    assert shares
    for value in shares:
        assert " из " in value["text"]
        assert value["text"] in page


def test_page_names_every_mart_it_shows(sample):
    report, page, _ = sample
    for model in report["mart_models"]:
        assert f"<code>{model}</code>" in page


def test_page_has_charts_and_no_outside_requests(sample):
    report, page, _ = sample
    assert report["charts"] == 4
    assert page.count("<svg") == report["charts"]
    # Внешний ресурс в разметке означает, что страница живет ровно до того дня,
    # когда чужая машина ответит иначе.
    assert not re.search(r"<(script|link|img|iframe)\b", page)


def test_empty_journal_builds_a_page_that_says_so(empty):
    assert empty["empty"] is True
    assert empty["journal"]["lines"] == 0
    assert empty["charts"] == 0
    page = Path(empty["page"]).read_text(encoding="utf-8")
    assert "Данных пока нет" in page
    assert "<svg" not in page


def test_empty_journal_invents_nothing(empty):
    """На пустом журнале на странице нет ни одного числа витрины."""
    assert [value for value in empty["values"] if value["source"] != "journal"] == []
    assert empty["mart_models"] == []


def test_query_text_from_the_journal_is_data_not_code():
    """Чужой текст в ячейке витрины остается текстом.

    Выдачи текстов запросов на странице нет, но значение из журнала в разметку
    попадает (сутки, имя ступени), и путь у всех значений один: `safe`.
    """
    hostile = "<script>alert(1)</script>\x00\x1b&\"'"
    mart = {
        "stage_latency": {
            "columns": ["stage", "runs", "p50_ms", "p95_ms", "max_ms"],
            "rows": [
                {"stage": hostile, "runs": "1", "p50_ms": "1.0", "p95_ms": "1.0", "max_ms": "1.0"}
            ],
        }
    }
    snapshot = dashboard.Snapshot(
        mart=mart,
        journal={
            "files": ["queries-2026-10-21.jsonl"],
            "days": 1,
            "lines": 1,
            "last_event_at": hostile,
        },
        corpus_version=hostile,
        corpus_source="results/regulation.json",
        config_version=hostile,
        source=hostile,
        built_at="2026-10-23 00:00:00Z",
        live=True,
    )
    page = dashboard.render(snapshot).page
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "\x00" not in page and "\x1b" not in page


def test_escaping_covers_quotes_and_control_characters():
    assert dashboard.safe('a"b') == "a&quot;b"
    assert dashboard.safe("a\x00b") == "a�b"
    assert dashboard.safe("a&b<c") == "a&amp;b&lt;c"


def test_null_from_clickhouse_is_not_printed_as_zero():
    """У первых суток предыдущих нет, и ноль здесь означал бы «не изменилось»."""
    assert dashboard.number("\\N") == dashboard.NO_VALUE
    assert dashboard.number("0") == "0"
    assert dashboard.number("104.03") == "104,03"
    assert dashboard.number("62594") == "62 594"


def test_page_shows_the_last_days_and_says_how_many_are_hidden(tmp_path):
    """Длинный журнал режется по суткам, и страница называет, сколько скрыто."""
    days = [f"2026-09-{day:02d}" for day in range(1, 21)]
    mart = {
        "traffic_by_day": {
            "columns": ["day", "queries"],
            "rows": [{"day": day, "queries": "10"} for day in days],
        }
    }
    snapshot = dashboard.Snapshot(
        mart=mart,
        journal={"files": [], "days": len(days), "lines": 200, "last_event_at": days[-1]},
        corpus_version="chunks1-acts1",
        corpus_source="results/corpus.json",
        config_version="bm25",
        source="набор проверки",
        built_at="2026-10-23 00:00:00Z",
        live=True,
    )
    page = dashboard.render(snapshot).page
    assert f"Показаны последние {dashboard.DAYS_SHOWN} суток из {len(days)}" in page
    assert days[0] not in page
    assert days[-1] in page


def test_journal_summary_reads_the_moment_of_the_last_event(tmp_path):
    (tmp_path / "queries-2026-10-21.jsonl").write_text(
        '{"kind":"query","request_id":"a","received_at":"2026-10-21T08:00:00.000Z"}\n'
        '{"kind":"query","request_id":"b","received_at":"2026-10-21T09:30:00.000Z"}\n',
        encoding="utf-8",
    )
    summary = dashboard.journal_summary(tmp_path)
    assert summary == {
        "files": ["queries-2026-10-21.jsonl"],
        "days": 1,
        "lines": 2,
        "last_event_at": "2026-10-21T09:30:00.000Z",
    }


def test_missing_journal_directory_is_not_a_failure(tmp_path):
    assert dashboard.journal_summary(tmp_path / "nothing")["lines"] == 0
