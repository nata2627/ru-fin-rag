"""Граф ссылок между актами и расширение выдачи по цитированию.

Главная проверка здесь не про метрику, а про разбор: номер акта читается тем же
выражением, что в части А. Если выражения разойдутся, числа графа и числа
фильтра по номеру (EXPERIMENTS, S8) перестанут быть сравнимыми, и заметить это
по выдаче будет нельзя.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

from rufinrag import graph as graph_module
from rufinrag import references
from rufinrag.config import GRAPH_DIRECTIONS, ConfigError, load_catalog
from rufinrag.corpus import Act, Chunk, Corpus, TextBuilder, corpus_path
from rufinrag.graph import ActGraph, GraphError, build_graph, cited_candidates, reserve_merge
from rufinrag.references import act_numbers, count_act_numbers
from rufinrag.retrieval.bm25 import BM25Index
from rufinrag.retrieval.fusion import rrf
from rufinrag.search import SearchService

PART_A = Path(os.environ.get("RUFIN_PART_A", Path(__file__).resolve().parents[1] / "retrieval"))

SAMPLES = (
    "В соответствии с Положением Банка России № 590-П и пунктом 3.12 указанного положения",
    "Федеральный закон от 10 июля 2002 года № 86-ФЗ о Центральном банке",
    "Указание 3054-У, положения 590-п, письмо ИН-08-23/42",
    "Процентная ставка № 1 и ставок № 2 к величине собственных средств",
    "Информационное письмо от 27.03.2020 № ИН-03-41/32 и 4927-У",
    "1-я группа активов, коэффициент 0,01 процента",
    "Инструкция № 180-И утратила силу, применяется 199-И",
    "",
)


def part_a_references():
    path = PART_A / "src/rufin/references.py"
    if not path.exists():
        pytest.skip(f"части А нет на диске: {path}")
    spec = importlib.util.spec_from_file_location("part_a_references", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def act(number: str) -> Act:
    return Act(
        act_id=f"{number}_01012020",
        number=number,
        date="01.01.2020",
        type=number.split("-")[-1],
        title=f"Акт {number}",
        url="",
    )


def corpus_of(texts: dict[str, list[str]]) -> Corpus:
    """Корпус из нескольких актов: ключ это номер акта, значение это его фрагменты."""
    chunks: list[Chunk] = []
    builder = TextBuilder()
    for number, pieces in texts.items():
        owner = act(number)
        for position, text in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{owner.act_id}#{position:04d}",
                    act=owner,
                    section="",
                    position=position,
                    char_start=0,
                    char_end=len(text),
                )
            )
            builder.add(text)
    return Corpus(chunks=chunks, texts=builder.build(), source=Path("память"))


def test_port_reads_the_same_numbers_as_part_a():
    original = part_a_references()
    for text in SAMPLES:
        assert act_numbers(text) == original.act_numbers(text), text


def test_port_keeps_the_expression_of_part_a():
    original = part_a_references()
    assert references.LETTERED == original.LETTERED


def test_numbers_are_counted_with_repetitions():
    assert count_act_numbers("590-П, 590-п и 86-ФЗ") == {"590-П": 2, "86-ФЗ": 1}


def test_own_number_in_every_chunk_is_not_a_reference():
    """Шапка акта стоит в тексте каждого его фрагмента, и своим номером акт не ссылается."""
    corpus = corpus_of({"590-П": ["Положение 590-П. Глава 1", "Положение 590-П. Глава 2"]})
    built = build_graph(corpus)
    assert built.forward == {}
    assert built.stats.mentions == 0


def test_reference_to_an_act_of_the_corpus_becomes_an_edge():
    corpus = corpus_of({"590-П": ["Положение 590-П применяется с 254-П"], "254-П": ["Текст"]})
    built = build_graph(corpus)
    assert built.forward == {"590-П_01012020": ("254-П_01012020",)}
    assert built.backward == {"254-П_01012020": ("590-П_01012020",)}
    assert built.stats.edges == 1
    assert built.stats.acts_with_references == 1


def test_reference_outside_the_corpus_is_counted_and_gives_no_edge():
    corpus = corpus_of({"590-П": ["Положение 590-П на основании 86-ФЗ и 86-ФЗ"]})
    built = build_graph(corpus)
    assert built.forward == {}
    assert built.stats.references_outside == 1
    assert built.stats.mentions_outside == 2
    assert built.stats.mentions_outside_act_like == 2


def test_the_same_number_of_several_acts_links_to_all_of_them():
    """Методические рекомендации нумеруются заново каждый год: «10-МР» носят несколько."""
    second = act("10-МР")
    older = Act(
        act_id="10-МР_01012015",
        number="10-МР",
        date="01.01.2015",
        type="МР",
        title="Акт 10-МР",
        url="",
    )
    assert second.act_id != older.act_id
    builder = TextBuilder()
    chunks = []
    for owner, text in (
        (act("590-П"), "Положение 590-П и рекомендации 10-МР"),
        (second, "Рекомендации 10-МР, редакция 2020"),
        (older, "Рекомендации 10-МР, редакция 2015"),
    ):
        chunks.append(
            Chunk(
                chunk_id=f"{owner.act_id}#0000",
                act=owner,
                section="",
                position=0,
                char_start=0,
                char_end=len(text),
            )
        )
        builder.add(text)
    built = build_graph(Corpus(chunks=chunks, texts=builder.build(), source=Path("память")))
    assert set(built.forward["590-П_01012020"]) == {"10-МР_01012020", "10-МР_01012015"}
    assert built.stats.ambiguous_references == 1


def test_neighbors_skip_the_seed_acts_themselves():
    built = ActGraph(
        forward={"a": ("b", "c"), "b": ("a",)},
        backward={"b": ("a",), "c": ("a",), "a": ("b",)},
        act_chunks={},
        stats=build_graph(corpus_of({"590-П": ["Положение 590-П"]})).stats,
    )
    assert built.neighbors(["a"], "forward") == ["b", "c"]
    assert built.neighbors(["a", "b"], "forward") == ["c"]
    assert built.neighbors(["b"], "backward") == ["a"]
    assert built.neighbors(["a"], "both") == ["b", "c"]
    with pytest.raises(GraphError, match="направление"):
        built.neighbors(["a"], "вперед")


@pytest.fixture
def toy():
    corpus = corpus_of(
        {
            "590-П": ["Положение 590-П о резервах на возможные потери, порядок из 254-П"],
            "254-П": ["Положение 254-П. Титульный лист", "Резерв по ссуде определяется так"],
            "180-И": ["Инструкция 180-И об обязательных нормативах"],
        }
    )
    bm25 = BM25Index.build(corpus.ids, list(corpus.iter_texts()))
    return corpus, bm25, build_graph(corpus)


def test_candidates_come_only_from_neighbor_acts(toy):
    corpus, bm25, built = toy
    base = [("590-П_01012020#0000", 1.0)]
    extra, info = cited_candidates(
        "резерв по ссуде", base, corpus, built, bm25, direction="forward", seeds=1, depth=5
    )
    assert info.neighbor_acts == 1
    assert [chunk_id for chunk_id, _ in extra] == ["254-П_01012020#0001"]


def test_candidates_are_empty_without_neighbors(toy):
    corpus, bm25, built = toy
    base = [("180-И_01012020#0000", 1.0)]
    extra, info = cited_candidates(
        "норматив", base, corpus, built, bm25, direction="forward", seeds=1, depth=5
    )
    assert extra == []
    assert info.added == 0


def test_reserve_keeps_the_base_order_and_gives_away_the_last_places():
    base = [(f"c{i}", 1.0 / (i + 1)) for i in range(10)]
    extra = [("x0", 9.0), ("x1", 8.0)]
    merged = [chunk_id for chunk_id, _ in reserve_merge(base, extra, reserve=2, top=10)]
    assert merged[:8] == [f"c{i}" for i in range(8)]
    assert merged[8:] == ["x0", "x1"]


def test_reserve_does_not_duplicate_what_base_already_has():
    base = [(f"c{i}", 1.0) for i in range(10)]
    merged = [chunk_id for chunk_id, _ in reserve_merge(base, [("c3", 5.0)], reserve=1, top=10)]
    assert merged == [f"c{i}" for i in range(9)] + ["c9"]
    assert len(set(merged)) == 10


def test_zero_reserve_changes_nothing():
    base = [(f"c{i}", 1.0) for i in range(10)]
    assert reserve_merge(base, [("x", 5.0)], reserve=0, top=10) == base


def test_fusion_weight_is_applied_to_the_second_list():
    base = [("a", 1.0), ("b", 1.0)]
    extra = [("c", 1.0)]
    assert [doc for doc, _ in rrf([base, extra], k=1, top=3, weights=(1.0, 0.1))] == ["a", "b", "c"]
    assert [doc for doc, _ in rrf([base, extra], k=1, top=3, weights=(0.1, 1.0))] == ["c", "a", "b"]
    with pytest.raises(ValueError, match="весов"):
        rrf([base, extra], weights=(1.0,))


def test_config_directions_match_the_graph():
    assert GRAPH_DIRECTIONS == graph_module.DIRECTIONS


def test_project_bundle_with_the_graph_is_parsed():
    spec = load_catalog().get("hybrid-graph").graph
    assert spec is not None
    assert (spec.mode, spec.direction, spec.seeds, spec.reserve) == ("reserve", "forward", 3, 1)
    assert "graphreserveforward3" in load_catalog().get("hybrid-graph").version


def test_unknown_mode_and_direction_fail_loudly(tmp_path):
    for broken, message in (
        ({"mode": "магия", "direction": "forward", "seeds": 3, "reserve": 1}, "режим"),
        ({"mode": "reserve", "direction": "вперед", "seeds": 3, "reserve": 1}, "направление"),
        ({"mode": "reserve", "direction": "forward", "seeds": 3}, "reserve"),
        ({"mode": "fusion", "direction": "forward", "seeds": 3}, "depth"),
        ({"mode": "reserve", "direction": "forward", "seeds": 0, "reserve": 1}, "seeds"),
    ):
        path = tmp_path / "retrieval.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "default": "a",
                    "configs": {"a": {"first_stage": "bm25", "candidates": 10, "graph": broken}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=message):
            load_catalog(path)


@pytest.mark.slow
def test_service_expansion_repeats_the_offline_measurement(tmp_path):
    """Связка сервиса обязана давать то же, что посчитал замер на том же корпусе.

    Иначе числа S20 описывают скрипт, а не сервис. Сравнение идет на лексической
    связке: плотная ступень на маке и в части А дает перестановки близких по
    оценке фрагментов, и сверка превратилась бы в сверку железа.
    """
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default": "bm25-graph",
                "configs": {
                    "bm25-graph": {
                        "first_stage": "bm25",
                        "candidates": 50,
                        "graph": {
                            "mode": "reserve",
                            "direction": "forward",
                            "seeds": 3,
                            "reserve": 1,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    service = SearchService.build(catalog=load_catalog(path), with_dense=False)
    built = service.act_graph()
    for query in ("резерв по ссуде", "норматив достаточности капитала"):
        base = service.bm25.search(query, k=50)
        extra, _ = cited_candidates(
            query, base, service.corpus, built, service.bm25, direction="forward", seeds=3, depth=1
        )
        expected = [chunk_id for chunk_id, _ in reserve_merge(base, extra, 1, top=10)]
        answer = service.search(query, top=10)
        assert [hit.chunk_id for hit in answer.hits] == expected
        assert "graph" in answer.timings_ms
