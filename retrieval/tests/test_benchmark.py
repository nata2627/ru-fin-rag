"""Запись и чтение набора запросов в формате MTEB (BeIR).

Формат нужен, чтобы бенчмарк можно было отдать наружу и прогнать чужим
кодом. Значит, он должен переживать круг «записали — прочитали» без потерь.
"""
from __future__ import annotations

from rufin.benchmark import read_jsonl, read_qrels, write_corpus, write_qrels, write_queries


def test_разметка_переживает_запись_и_чтение(tmp_path):
    qrels = {"q1": {"c1": 2, "c2": 1}, "q2": {"c3": 2}}
    path = tmp_path / "qrels" / "test.tsv"
    write_qrels(str(path), qrels)
    assert read_qrels(str(path)) == qrels


def test_нулевая_оценка_не_попадает_в_файл(tmp_path):
    path = tmp_path / "qrels.tsv"
    write_qrels(str(path), {"q1": {"c1": 2, "c2": 0}})
    assert read_qrels(str(path)) == {"q1": {"c1": 2}}


def test_у_разметки_есть_шапка(tmp_path):
    path = tmp_path / "qrels.tsv"
    write_qrels(str(path), {"q1": {"c1": 2}})
    assert path.read_text(encoding="utf-8").splitlines()[0] == "query-id\tcorpus-id\tscore"


def test_файл_без_шапки_тоже_читается(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("q1\tc1\t2\n", encoding="utf-8")
    assert read_qrels(str(path)) == {"q1": {"c1": 2}}


def test_битая_строка_пропускается(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\tc1\t2\nмусор\n", encoding="utf-8")
    assert read_qrels(str(path)) == {"q1": {"c1": 2}}


def test_корпус_несет_опознавательные_данные_акта(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_corpus(str(path), [{
        "chunk_id": "590-П_28062017#0001", "number": "590-П", "date": "28.06.2017",
        "title": "О резервах", "section": "Глава 1", "text": "текст фрагмента",
    }])
    row = read_jsonl(str(path))[0]
    assert row["_id"] == "590-П_28062017#0001"
    assert row["title"] == "590-П от 28.06.2017. О резервах // Глава 1"
    assert row["text"] == "текст фрагмента"


def test_фрагмент_без_раздела_не_получает_пустой_хвост(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_corpus(str(path), [{
        "chunk_id": "c1", "number": "1-У", "date": "01.01.2020",
        "title": "О чем-то", "section": "", "text": "текст",
    }])
    assert "//" not in read_jsonl(str(path))[0]["title"]


def test_запросы_пишутся_в_формате_beir(tmp_path):
    path = tmp_path / "queries.jsonl"
    write_queries(str(path), [{"query_id": "syn0000", "text": "Какой порядок?", "origin": "x"}])
    row = read_jsonl(str(path))[0]
    assert row == {"_id": "syn0000", "text": "Какой порядок?"}


def test_кириллица_не_экранируется(tmp_path):
    path = tmp_path / "queries.jsonl"
    write_queries(str(path), [{"query_id": "q1", "text": "резерв"}])
    assert "резерв" in path.read_text(encoding="utf-8")


def test_разметка_с_четвёртым_столбцом_читается(tmp_path):
    """У разметки по названным пунктам есть столбец с основанием.

    Требование «ровно три столбца» прочитало бы такой файл как пустой,
    и разметка исчезла бы молча.
    """
    path = tmp_path / "qrels_by_clause.tsv"
    path.write_text("query-id\tcorpus-id\tscore\tosnovanie\n"
                    "exp0000\tакт#0001\t2\tпункты 3.1, 3.2 акта 590-П\n",
                    encoding="utf-8")
    assert read_qrels(path) == {"exp0000": {"акт#0001": 2}}


def test_pisma_po_porogu():
    from rufin.benchmark import LETTER_CHARS, letter_subsets

    texts = {"a": "x" * LETTER_CHARS, "b": "x" * (LETTER_CHARS + 1), "c": "коротко"}
    out = letter_subsets(texts, ["a", "b", "c"])
    assert out["письма"] == ["b"]
    assert out["живые без писем"] == ["a", "c"]


def test_pisma_v_podvyborkah_i_dokumente():
    """Письма это ровно живые длиннее порога, «живые» не изменились, а блок
    документа 18 совпадает с pisma.json."""
    import json
    import pathlib
    import sys

    from rufin.benchmark import LETTER_CHARS

    root = pathlib.Path(__file__).resolve().parents[1]
    subsets = json.loads((root / "data" / "queries" / "podvyborki.json").read_text(encoding="utf-8"))
    texts = {}
    with (root / "data" / "queries" / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            texts[q["query_id"]] = q["text"]
    live = subsets["живые"]
    assert sorted(subsets["письма"]) == sorted(q for q in live if len(texts[q]) > LETTER_CHARS)
    assert sorted(subsets["письма"] + subsets["живые без писем"]) == sorted(live)

    sys.path.insert(0, str(root / "scripts"))
    import letters_subset as L

    result = json.loads((root / "data" / "results" / "pisma.json").read_text(encoding="utf-8"))
    doc = (root / "docs" / "18-nabor-do-i-posle.md").read_text(encoding="utf-8")
    assert doc.split(L.BLOCK_START, 1)[1].split(L.BLOCK_END, 1)[0].strip() == L.doc_block(result)
    assert result["письма теста"] == subsets["письма"]
