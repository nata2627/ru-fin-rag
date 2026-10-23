"""Витрина над журналом: одни модели, два режима хранения, один результат.

Главная проверка здесь `test_both_modes_return_the_same_table`: она гоняет
модели витрины по файлам JSONL через chdb и по тому же набору, залитому в
ClickHouse сервером, и сравнивает выходы как текст. Эта проверка требует
докера, поэтому помечена маркером `clickhouse` и пропускается, когда сервера
нет. Остальное проверяется в бесплатном режиме, который работает везде.

Сами модели лежат в проекте dbt, и сюда приходит их текст. Поэтому тут же
проверяется связность проекта: что у каждой модели есть описание и тесты в
`schema.yml`, что марты читают события через слой `staging`, и что после
разворота модели в один запрос в тексте не осталось разметки jinja. Собирает
проект и сверяет его вывод с бесплатным режимом `scripts/build_mart.py`:
для этого нужен и сервер, и сам dbt, которого в прогоне тестов нет.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

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


@pytest.fixture(scope="module")
def models():
    return warehouse.load_models()


def test_mart_query_text_is_the_same_for_both_modes():
    """Разный только источник событий, остальной текст модели один."""
    files = warehouse.FilesWarehouse(Path("/tmp/events")).source()
    server = warehouse.ServerWarehouse().source()
    assert files != server
    for name in warehouse.MART:
        by_files = warehouse.compose(name, files)
        assert by_files.replace(files, server) == warehouse.compose(name, server)


def test_composed_model_keeps_no_jinja(models):
    """Разметку dbt исполняет dbt, а бесплатный режим обязан развернуть ее всю."""
    source = warehouse.FilesWarehouse(Path("/tmp/events")).source()
    for name in models:
        composed = warehouse.compose(name, source)
        assert "{{" not in composed and warehouse.SOURCE_PLACEHOLDER not in composed


def test_marts_read_the_journal_through_staging(models):
    """Март читает события только через `staging`.

    Иначе чистка событий (сутки из `received_at`, ступень `total` из карты)
    разъехалась бы между моделями, и трафик в одной модели считался бы по
    другому набору строк, чем качество в соседней.
    """
    for name, model in models.items():
        reads_journal = warehouse.SOURCE_PLACEHOLDER in model.body
        assert reads_journal == (model.layer == warehouse.STAGING), name
        if model.layer == warehouse.MARTS:
            assert model.refs, f"март {name} ни на что не ссылается"


def test_every_model_is_described_and_tested(models):
    """У каждой модели есть описание и хотя бы один тест в своем `schema.yml`."""
    described: dict[str, dict] = {}
    for path in (warehouse.DBT_PROJECT / "models").glob("*/schema.yml"):
        for entry in yaml.safe_load(path.read_text(encoding="utf-8"))["models"]:
            described[entry["name"]] = entry
    assert set(described) == set(models)
    for name, entry in described.items():
        assert entry.get("description", "").strip(), name
        tests = [test for column in entry["columns"] for test in column.get("tests", ())]
        assert tests, name


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


@pytest.fixture(scope="module")
def gapped(tmp_path_factory):
    """Журнал с дырой в ряду суток: 1, 2 и 10 октября, между ними сервис молчал.

    Дыра тут обычное состояние бесплатного хостинга, а не редкость, и именно
    на ней видно, по чему идет недельное окно.
    """
    pytest.importorskip("chdb", reason="нет chdb, поставить: pip install -e '.[analytics]'")
    directory = tmp_path_factory.mktemp("gapped")
    batch = compare_warehouses.sample_events(
        days=("2026-10-01", "2026-10-02", "2026-10-10"), per_day=10
    )
    compare_warehouses.write_sample(directory, batch)
    return warehouse.FilesWarehouse(directory)


def test_week_window_counts_calendar_days_not_rows(gapped):
    """Недельное окно идет по номеру суток, а не по строкам.

    По строкам сутки с дырой попали бы в одно окно с теми, что были неделей
    раньше, и «за последние семь суток» означало бы «за последние три записи».
    """
    table = {row["day"]: row for row in rows(gapped.run("traffic_by_day"))}
    assert list(table) == ["2026-10-01", "2026-10-02", "2026-10-10"]
    assert [table[day]["days_in_7d"] for day in table] == ["1", "2", "1"]
    assert table["2026-10-02"]["queries_7d"] == "20"
    # Восьмое число суток назад в окно не попадает, поэтому недельное число
    # последних суток это только они сами.
    assert table["2026-10-10"]["queries_7d"] == "10"
    quality = {row["day"]: row for row in rows(gapped.run("quality_by_day"))}
    assert [quality[day]["days_in_7d"] for day in quality] == ["1", "2", "1"]


def test_running_total_and_change_come_from_the_window(gapped):
    table = rows(gapped.run("traffic_by_day"))
    running = 0
    previous = None
    for row in table:
        running += int(row["queries"])
        assert int(row["queries_total"]) == running
        # У первых суток предыдущих нет, значит разницы нет: ноль означал бы,
        # что трафик не изменился.
        expected = r"\N" if previous is None else str(int(row["queries"]) - previous)
        assert row["queries_change"] == expected
        previous = int(row["queries"])


def test_repeat_window_tells_the_first_ask_from_a_repeat(gapped):
    """Место запроса в ряду повторов своего текста считается окном.

    Группировкой это не считается: номер повтора живет на строке журнала, а не
    на группе, поэтому первые обращения видны только в первых сутках, хотя
    текстов в каждых сутках одни и те же пять.
    """
    table = rows(gapped.run("repeat_queries"))
    assert [row["first_time"] for row in table] == ["5", "0", "0"]
    assert [row["distinct_texts"] for row in table] == ["5", "5", "5"]
    for row in table:
        assert int(row["first_time"]) + int(row["repeated"]) == int(row["answered"])
    assert int(table[-1]["median_gap_seconds"]) > 0


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


@pytest.mark.parametrize(
    ("code", "answer", "expected"),
    [(200, b"1\n", True), (401, b"Unauthorized", False)],
)
def test_available_asks_the_question_it_claims_to_ask(code, answer, expected):
    """Доступность это «пустят ли нас внутрь», а не «отвечает ли порт».

    Проверка стояла на `/ping`, который отвечает «Ok.» без всякой авторизации.
    Из-за этого в прогоне CI сервер, закрывший сетевой доступ пользователю
    `default`, выглядел доступным, а 401 вылетал уже посреди загрузки журнала
    (прогон 39976224616).
    """
    import http.server
    import threading

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Ok.\n")

        def do_POST(self):
            self.send_response(code)
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address[:2]
        assert warehouse.ServerWarehouse(f"http://{host}:{port}").available() is expected
    finally:
        server.shutdown()
        server.server_close()
