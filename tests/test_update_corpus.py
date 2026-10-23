"""Проверки докачки корпуса, которые можно сделать без сети.

Сам сбор с cbr.ru тестом не трогается: это чужой сайт, и дергать его из тестов
нельзя. Проверяется то, что ломается тихо: ключ акта, разница с корпусом и поля
фрагмента, которые должны совпасть с теми, что уже лежат в корпусе.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update_corpus", ROOT / "scripts/update_corpus.py")
update_corpus = importlib.util.module_from_spec(spec)
sys.modules["update_corpus"] = update_corpus
spec.loader.exec_module(update_corpus)

from rufinrag.corpus import FIELDS  # noqa: E402

ACT = {
    "act_id": "9999-П_01102026",
    "number": "9999-П",
    "date": "01.10.2026",
    "type": "П",
    "kind": "положение",
    "title": "О порядке проверки ночного прогона",
    "issue": "Вестник Банка России № 1 (2026)",
    "issue_url": "https://www.cbr.ru/Queries/XsltBlock/File/59540/0000",
    "text": "Глава 1. Общие положения\n1.1. Настоящее Положение проверяет нарезку. " * 40,
}


def test_act_key_keeps_number_and_date():
    """Номер без даты не уникален: у методических рекомендаций нумерация годовая."""
    assert update_corpus.act_key("9999-П", "01.10.2026") == "9999-П_01102026"


def test_known_acts_reads_compressed_corpus(tmp_path):
    corpus = tmp_path / "chunks.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as f:
        for act_id in ("590-П_28062017", "3054-У_04092013"):
            f.write(json.dumps({"act_id": act_id, "chunk_id": f"{act_id}#0000"}) + "\n")
    assert update_corpus.known_acts(corpus) == {"590-П_28062017", "3054-У_04092013"}


def test_chunk_rows_match_corpus_fields():
    rufin_chunking = pytest.importorskip("rufin.chunking")
    rows = update_corpus.chunk_rows(ACT, rufin_chunking.TokenRuler(None))
    assert rows
    for row in rows:
        assert tuple(row) == FIELDS
        assert row["act_id"] == ACT["act_id"]
        assert row["url"] == ACT["issue_url"]
        assert row["text"]
    assert [row["chunk_id"] for row in rows] == [
        f"{ACT['act_id']}#{i:04d}" for i in range(len(rows))
    ]


def test_skipped_acts_survive_between_runs(tmp_path):
    """Иначе прогон каждую ночь качает одни и те же 80 выпусков ради нуля актов."""
    path = tmp_path / "skipped.json"
    assert update_corpus.load_skipped(path) == {}
    update_corpus.save_skipped(path, {"590-П_28062017": {"reason": "мало кириллицы", "tries": 1}})
    assert update_corpus.load_skipped(path)["590-П_28062017"]["reason"] == "мало кириллицы"


def test_append_keeps_the_corpus_readable_on_a_broken_run(tmp_path, monkeypatch):
    """Обрыв посреди дописывания не должен оставлять нечитаемый корпус."""
    corpus = tmp_path / "chunks.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"act_id": "590-П_28062017", "chunk_id": "590-П_28062017#0000"}) + "\n")
    before = corpus.read_bytes()

    def explode(*args, **kwargs):
        raise KeyboardInterrupt("процесс снят посреди записи")

    monkeypatch.setattr(update_corpus.os, "replace", explode)
    with pytest.raises(KeyboardInterrupt):
        update_corpus.append_rows(corpus, [{"act_id": "new", "chunk_id": "new#0000"}])

    assert corpus.read_bytes() == before
    assert update_corpus.known_acts(corpus) == {"590-П_28062017"}
    assert not list(tmp_path.glob("*.new"))


def test_append_adds_rows_that_read_back(tmp_path):
    corpus = tmp_path / "chunks.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"act_id": "old", "chunk_id": "old#0000"}) + "\n")
    update_corpus.append_rows(corpus, [{"act_id": "new", "chunk_id": "new#0000"}])
    assert update_corpus.known_acts(corpus) == {"old", "new"}


def test_source_refusal_asks_the_run_to_try_later(tmp_path, monkeypatch):
    """Отказ cbr.ru это не поломка: корпус цел, и прогон обязан идти дальше.

    Прогон 39918625392 упал на этом месте целиком, и вместе с докачкой отменил
    переиндексацию, регрессию и отметку живости. Код 75 это контракт, на
    который смотрит шаг `Fetch new acts` в ночном прогоне.
    """
    vestnik = pytest.importorskip("rufin.sources.cbr_vestnik")

    def refuse(client):
        raise RuntimeError(
            "не удалось получить https://www.cbr.ru/about_br/publ/vestnik-search/: "
            "403 Client Error: Forbidden"
        )

    monkeypatch.setattr(vestnik, "fetch_acts_index", refuse)
    out = tmp_path / "corpus-update.json"
    monkeypatch.setattr(sys, "argv", ["update_corpus.py", "--out", str(out)])

    assert update_corpus.main() == update_corpus.EXIT_SOURCE_UNAVAILABLE
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "source_unavailable"
    assert "403" in report["source_error"]
    assert report["new_acts_added"] == 0


def test_empty_index_is_a_breakage_and_not_a_quiet_night(tmp_path, monkeypatch):
    """Сайт ответил, а под состав корпуса не подошло ничего.

    В норме это 1656 актов. Ноль означает, что разбор перечня больше не
    понимает верстку, и промолчать тут нельзя: «новых актов нет» стало бы
    ответом навсегда.
    """
    vestnik = pytest.importorskip("rufin.sources.cbr_vestnik")
    monkeypatch.setattr(vestnik, "fetch_acts_index", lambda client: [{"number": "1-У"}])
    monkeypatch.setattr(vestnik, "select_corpus", lambda index, since: [])
    out = tmp_path / "corpus-update.json"
    monkeypatch.setattr(sys, "argv", ["update_corpus.py", "--out", str(out)])

    assert update_corpus.main() == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "index_not_parsed"


def test_http_code_tells_a_missing_issue_from_a_refusing_site():
    """Два настоящих сообщения из прогонов 23.10.2026, по ним и разводятся пути.

    403 на поиске это отказ сайта, и докачка откладывается целиком. 404 на одном
    выпуске это свойство выпуска: его акты уходят в `skipped.json`, а прогон
    идет дальше. Сообщение без кода ответа считается отказом сайта, то есть
    более осторожным из двух толкований.
    """
    refusal = (
        "не удалось получить https://www.cbr.ru/about_br/publ/vestnik-search/: "
        "403 Client Error: Forbidden for url: https://www.cbr.ru/about_br/publ/vestnik-search/"
    )
    missing = (
        "не удалось получить https://www.cbr.ru/Queries/UniDbQuery/File/85920/-1/2623: "
        "404 Client Error: Not Found for url: "
        "https://www.cbr.ru/Queries/UniDbQuery/File/85920/-1/2623"
    )
    broken_pipe = "не удалось получить https://www.cbr.ru/: Connection aborted"

    assert update_corpus.http_code(RuntimeError(refusal)) == 403
    assert update_corpus.http_code(RuntimeError(missing)) == 404
    assert update_corpus.http_code(RuntimeError(broken_pipe)) is None


def test_repeat_run_asks_for_no_act_twice():
    """Второй прогон на том же перечне обязан вернуть пустой разбор.

    Это и есть идемпотентность докачки: ключ «номер плюс дата» сверяется с
    корпусом, и акт, который уже лежит в корпусе, в разбор не попадает.
    """
    wanted = [
        {"number": "9999-П", "date": "01.10.2026", "title": "О порядке"},
        {"number": "1234-У", "date": "02.10.2026", "title": "О сроках"},
    ]
    first = update_corpus.pending_acts(wanted, set(), set(), lambda number: number)
    assert sorted(first) == ["1234-У_02102026", "9999-П_01102026"]

    added_to_corpus = set(first)
    assert update_corpus.pending_acts(wanted, added_to_corpus, set(), lambda n: n) == {}


def test_acts_that_did_not_give_in_are_not_asked_again():
    """Память о неудачах тоже останавливает повтор: иначе 97 актов каждую ночь."""
    wanted = [{"number": "9999-П", "date": "01.10.2026", "title": "О порядке"}]
    blocked = {"9999-П_01102026"}
    assert update_corpus.pending_acts(wanted, set(), blocked, lambda n: n) == {}


def test_one_act_published_in_two_issues_is_taken_once():
    """В перечне у cbr.ru такие позиции есть: 97 недостающих актов на 102 позиции."""
    wanted = [
        {"number": "9999-П", "date": "01.10.2026", "title": "О порядке", "issue": "1"},
        {"number": "9999-П", "date": "01.10.2026", "title": "О порядке", "issue": "2"},
    ]
    assert list(update_corpus.pending_acts(wanted, set(), set(), lambda n: n)) == [
        "9999-П_01102026"
    ]
