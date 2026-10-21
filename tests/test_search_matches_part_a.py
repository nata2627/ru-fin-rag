"""Поиск сервиса обязан воспроизводить выдачу бенчмарка части А.

Код BM25, токенизация и слияние обратными рангами перенесены из части А, и если
они разойдутся, числа README перестанут соответствовать коду. Тесты держат это
свойство: та же связка на тех же запросах дает тот же порядок фрагментов.

Плотная ступень и гибрид сверяются мягче, и это свойство замера, а не поблажка.
В части А векторы корпуса считались на видеокарте в половинной точности и поиск
шел через FAISS, здесь умножение идет в numpy в одинарной точности на
процессоре, поэтому близкие по оценке фрагменты меняются местами. Замер на 200
запросах (`results/reproducibility.json`, пересчитывается
`scripts/check_reproducibility.py`): BM25 совпадает полностью, 200 из 200 по
порядку; гибрид дает тот же верхний результат в 199 случаях из 200, тот же
состав первой десятки в 195, тот же порядок в 181; плотная ступень 197, 189 и
164 соответственно. Пороги тестов ниже взяты из этих чисел.
"""

import pytest

from rufinrag.benchmark import load_queries, load_run
from rufinrag.corpus import corpus_path
from rufinrag.search import SearchError, SearchService

SAMPLE = 20


@pytest.fixture(scope="module")
def lexical():
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    return SearchService.build(with_dense=False)


@pytest.fixture(scope="module")
def full():
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    service = SearchService.build()
    if service.dense is None:
        pytest.skip(f"плотный поиск недоступен: {service.degraded}")
    return service


def sample_ids(reference, queries):
    ids = sorted(set(reference) & set(queries))[:SAMPLE]
    assert len(ids) == SAMPLE
    return ids


def test_bm25_reproduces_reference_run(lexical):
    queries = load_queries()
    reference = load_run("bm25")
    mismatched = [
        query_id
        for query_id in sample_ids(reference, queries)
        if [h.chunk_id for h in lexical.search(queries[query_id].text, 10, "bm25").hits]
        != reference[query_id][:10]
    ]
    assert not mismatched, f"выдача bm25 разошлась с частью А: {mismatched}"


def test_hybrid_reproduces_reference_run(full):
    queries = load_queries()
    reference = load_run("hybrid")
    ids = sample_ids(reference, queries)
    top1 = 0
    composition = 0
    for query_id in ids:
        mine = [h.chunk_id for h in full.search(queries[query_id].text, 10, "hybrid").hits]
        theirs = reference[query_id][:10]
        top1 += mine[0] == theirs[0]
        composition += set(mine) == set(theirs)
    assert top1 >= len(ids) - 1, f"верхний результат гибрида разошелся {len(ids) - top1} раз"
    assert (
        composition >= len(ids) - 2
    ), f"состав первой десятки разошелся {len(ids) - composition} раз"


def test_dense_stage_matches_reference_at_the_top(full):
    """Верхний результат плотной ступени совпадает, хвост может переставляться."""
    queries = load_queries()
    reference = load_run("dense")
    ids = sample_ids(reference, queries)
    top1 = 0
    composition = 0
    for query_id in ids:
        vector = full.encoder.encode_one(queries[query_id].text)
        mine = [chunk_id for chunk_id, _ in full.dense.search(vector, k=10)]
        theirs = reference[query_id][:10]
        top1 += mine[0] == theirs[0]
        composition += set(mine) == set(theirs)
    assert top1 >= len(ids) - 1, f"верхний результат разошелся на {len(ids) - top1} запросах"
    assert composition >= len(ids) - 3, f"состав разошелся на {len(ids) - composition} запросах"


def test_result_carries_versions(lexical):
    result = lexical.search("требования к расчету норматива H1", config="bm25")
    assert result.corpus_version == "chunks62594-acts1554"
    assert result.config_version == "bm25+bm25+cand50"
    assert result.timings_ms["total"] > 0
    assert len(result.hits) == 10


def test_hits_point_at_the_source(lexical):
    hit = lexical.search("порядок формирования резервов по ссудам", config="bm25").hits[0]
    assert hit.url.startswith("https://www.cbr.ru/")
    assert hit.act.startswith(("Указание", "Положение", "Инструкция", "Приказ", "Акт"))
    assert hit.text


def test_empty_query_is_an_error(lexical):
    with pytest.raises(SearchError, match="пустой запрос"):
        lexical.search("   ")


def test_unknown_config_is_an_error(lexical):
    from rufinrag.config import ConfigError

    with pytest.raises(ConfigError, match="нет связки"):
        lexical.search("норматив H1", config="hybrid-rerank-v3")


def test_degrades_to_lexical_without_dense(lexical):
    """Векторного хранилища нет, значит выдача лексическая и об этом сказано."""
    result = lexical.search("норматив достаточности капитала", config="hybrid")
    assert result.hits
    assert result.config_version.endswith("+degraded")
    assert any("плотный поиск" in note for note in result.degraded)
