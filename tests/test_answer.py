"""Слой ответа: цитаты по корпусу, отказ без опоры, оба режима.

Платных вызовов здесь нет и быть не может: генеративный режим получает
подставную модель функцией, а настоящий клиент собирается только тогда, когда
в окружении есть все три переменные (`rufinrag.generation`).
"""

import json
import os
from pathlib import Path

import pytest

from rufinrag import events
from rufinrag.answer import (
    EXCERPTS,
    GENERATE,
    IDF_COVERAGE,
    SUPPORT_FEATURES,
    AnswerError,
    AnswerService,
    Citation,
    clauses_in,
    coverage_of,
    header_length,
    idf_weights,
    load_answer_settings,
)
from rufinrag.config import ConfigError
from rufinrag.corpus import corpus_path
from rufinrag.generation import (
    BASE_URL_VAR,
    KEY_VAR,
    MODEL_VAR,
    GenerationError,
    GenerationSettings,
    parse_reply,
    settings_from_env,
)
from rufinrag.monitoring import SUPPORT_FEATURES as MONITOR_FEATURES
from rufinrag.search import SearchService

pytestmark = pytest.mark.answer

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT = ROOT / "results/answer.json"
PROBE = ROOT / "data/benchmark/out-of-corpus.jsonl"

IN_CORPUS = "порядок формирования резервов на возможные потери по ссудам"
# Вопрос из набора вне корпуса, группа «вне предметной области».
OUT_OF_CORPUS = "Сколько минут варить куриные яйца вкрутую?"


@pytest.fixture(scope="module")
def service():
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    search = SearchService.build(with_dense=False)
    return AnswerService.build(search)


@pytest.fixture
def measurement():
    if not MEASUREMENT.exists():
        pytest.skip("нет замера results/answer.json: make answer-threshold")
    return json.loads(MEASUREMENT.read_text(encoding="utf-8"))


def probe_queries(group: str | None = None) -> list[str]:
    rows = [json.loads(line) for line in PROBE.read_text(encoding="utf-8").splitlines() if line]
    return [row["text"] for row in rows if group is None or row["group"] == group]


def stub(reply: str, calls: list[str] | None = None):
    def complete(system: str, user: str) -> str:
        if calls is not None:
            calls.append(user)
        return reply

    return complete


def with_model(service: AnswerService, completer) -> AnswerService:
    """Тот же сервис с подставной моделью: настоящий клиент не собирается."""
    return AnswerService(
        search=service.search,
        settings=service.settings,
        generation=GenerationSettings(
            base_url="http://stub",
            model="stub-7b",
            key="stub",
            max_tokens=256,
            temperature=0.0,
            timeout_seconds=1.0,
        ),
        generation_off="",
        completer=completer,
    )


def test_threshold_in_config_equals_the_measured_one(measurement):
    """Порог в конфигурации и порог в замере это одно число, а не два похожих."""
    settings = load_answer_settings()
    assert settings.feature == measurement["support_feature"]
    assert settings.threshold == measurement["support_threshold"]
    assert settings.measured_in == "results/answer.json"


def test_support_feature_is_the_one_the_monitor_counts():
    """Признак опоры у сервиса и у метрики доли запросов без ответа один."""
    settings = load_answer_settings()
    assert settings.feature in SUPPORT_FEATURES
    assert set(SUPPORT_FEATURES) == {*MONITOR_FEATURES, IDF_COVERAGE}


def test_question_outside_the_corpus_is_refused_without_citations(service):
    """Выход задачи 7.1: вопрос вне корпуса дает отказ, а не выдумку."""
    answer = service.answer(OUT_OF_CORPUS, top=10)
    assert answer.answered is False
    assert answer.citations == ()
    assert answer.text == ""
    assert "опоры в корпусе нет" in answer.refusal
    # В отказе стоят числа, по которым он случился, а не одно слово «нет».
    assert str(answer.support.query_tokens) in answer.refusal
    assert answer.support.passed is False


def test_refusals_on_the_probe_set_match_the_measurement(service, measurement):
    """Доля отказов вне корпуса та же, что напечатана в замере."""
    refused = sum(1 for text in probe_queries() if not service.answer(text, top=10).answered)
    measured = (
        measurement["chosen"]["out_of_corpus_choose"]["refused"]
        + measurement["chosen"]["out_of_corpus_verify"]["refused"]
    )
    assert refused == measured, f"сервис отказал {refused} раз, в замере {measured}"


def test_in_corpus_question_is_answered_with_citations(service):
    answer = service.answer(IN_CORPUS, top=5)
    assert answer.answered is True
    assert answer.citations
    assert answer.text
    assert answer.text_source == "выдержки из найденных фрагментов"
    assert answer.refusal == ""


def test_citations_point_at_what_the_corpus_really_has(service):
    """Каждое поле ссылки сверяется с корпусом, а цитата с текстом фрагмента."""
    corpus = service.search.corpus
    for citation in service.answer(IN_CORPUS, top=5).citations:
        chunk = corpus.get(citation.chunk_id)
        text = corpus.text(citation.chunk_id)
        assert citation.number == chunk.act.number
        assert citation.date == chunk.act.date
        assert citation.url == chunk.act.url
        assert citation.section == chunk.section
        assert (citation.char_start, citation.char_end) == (chunk.char_start, chunk.char_end)
        assert citation.url.startswith("https://www.cbr.ru/")
        # Цитата это окно текста фрагмента, и его границы напечатаны рядом.
        window = " ".join(text[citation.quote_from : citation.quote_to].split())
        assert citation.quote.strip(". ") in window
        for clause in citation.clauses:
            assert clause in text


def citation_with(section: str, clauses: tuple[str, ...]) -> Citation:
    return Citation(
        chunk_id="590-П_28062017#0001",
        act="Положение Банка России № 590-П от 28.06.2017",
        act_id="590-П_28062017",
        number="590-П",
        date="28.06.2017",
        title="О порядке формирования резервов",
        section=section,
        clauses=clauses,
        url="https://www.cbr.ru/x",
        char_start=0,
        char_end=10,
        quote="текст",
        quote_from=0,
        quote_to=5,
        quote_terms=1,
        rank=1,
        score=1.0,
        idf_coverage=1.0,
    )


def test_reference_names_the_act_the_section_and_the_clauses():
    full = citation_with("Глава 1. Общие положения", ("1.1", "1.2"))
    assert full.reference == (
        "Положение Банка России № 590-П от 28.06.2017, Глава 1. Общие положения, пункты 1.1, 1.2"
    )
    assert citation_with("", ("1.1",)).reference.endswith("пункт 1.1")
    # Раздела и пунктов у фрагмента может не быть: в корпусе раздел есть у
    # 53 490 фрагментов из 62 606, номер пункта нашелся у 30 919. Пустое поле
    # в ссылку не печатается, выдумывать его нечем.
    assert citation_with("", ()).reference == "Положение Банка России № 590-П от 28.06.2017"


def test_clauses_are_read_from_the_beginning_of_a_line():
    text = (
        "1.1. Кредитная организация оценивает риск.\n"
        "Ссылка на пункт 3.14.3 внутри строки отдельным номером не становится.\n"
        "2.5 Резерв формируется."
    )
    assert clauses_in(text) == ("1.1", "2.5")


def test_quote_skips_the_header_of_the_fragment(service):
    """Шапку фрагмента цитировать незачем: она одна у всех фрагментов акта."""
    corpus = service.search.corpus
    for citation in service.answer(IN_CORPUS, top=5).citations:
        text = corpus.text(citation.chunk_id)
        chunk = corpus.get(citation.chunk_id)
        assert citation.quote_from >= header_length(chunk, text)
        assert not citation.quote.startswith(chunk.act.label)


def test_coverage_counts_a_word_missing_from_the_corpus_as_the_rarest(service):
    """Слово вне словаря обязано тянуть покрытие вниз, иначе борщ находится."""
    bm25 = service.search.bm25
    weights = idf_weights(bm25, "порядок приготовления борща")
    known = {term for term in weights if term in bm25.vocab}
    assert known != set(weights), "в запросе должно быть слово вне словаря"
    assert coverage_of(weights, known) < 1.0


def test_generation_is_unavailable_without_the_key(service):
    assert service.generation_ready is False
    with pytest.raises(AnswerError, match="генеративный режим недоступен"):
        service.answer(IN_CORPUS, mode=GENERATE)


def test_half_configured_generation_is_loud(monkeypatch):
    """Задана половина переменных, значит это опечатка, а не выключенный режим."""
    monkeypatch.setenv(KEY_VAR, "x")
    monkeypatch.delenv(BASE_URL_VAR, raising=False)
    monkeypatch.delenv(MODEL_VAR, raising=False)
    with pytest.raises(ConfigError, match="наполовину"):
        settings_from_env(max_tokens=1, temperature=0.0, timeout_seconds=1.0)


def test_generation_off_reason_names_the_variables():
    for name in (KEY_VAR, BASE_URL_VAR, MODEL_VAR):
        os.environ.pop(name, None)
    settings, reason = settings_from_env(max_tokens=1, temperature=0.0, timeout_seconds=1.0)
    assert settings is None
    assert KEY_VAR in reason


def test_generated_answer_keeps_only_the_cited_excerpts(service):
    calls: list[str] = []
    reply = json.dumps({"answer": "Резерв формируется по правилам Положения.", "used": [2]})
    answer = with_model(service, stub(reply, calls)).answer(IN_CORPUS, mode=GENERATE)
    assert answer.answered is True
    assert answer.text == "Резерв формируется по правилам Положения."
    assert answer.text_source == "модель stub-7b по найденным фрагментам"
    assert len(answer.citations) == 1
    assert answer.citations[0].rank == 2
    # В подсказку уехали выдержки со ссылками, а не весь корпус.
    assert "Выдержки:" in calls[0]
    assert "[1]" in calls[0]


def test_generation_is_not_called_at_all_when_there_is_no_support(service):
    """Отказ случается до модели: платный вызов на вопрос вне корпуса не нужен."""
    calls: list[str] = []
    reply = json.dumps({"answer": "Варить девять минут.", "used": [1]})
    answer = with_model(service, stub(reply, calls)).answer(OUT_OF_CORPUS, mode=GENERATE)
    assert answer.answered is False
    assert calls == []
    assert answer.mode == GENERATE


def test_generated_answer_citing_an_act_outside_the_hits_is_refused(service):
    reply = json.dumps({"answer": "Смотрите Указание Банка России № 9999-У.", "used": [1]})
    answer = with_model(service, stub(reply)).answer(IN_CORPUS, mode=GENERATE)
    assert answer.answered is False
    assert "вне выдачи" in answer.refusal
    assert "9999-у" in answer.refusal.lower()


def test_unparseable_model_reply_is_a_refusal(service):
    answer = with_model(service, stub("конечно, вот ответ")).answer(IN_CORPUS, mode=GENERATE)
    assert answer.answered is False
    assert "не JSON" in answer.refusal


def test_model_refusal_is_passed_through(service):
    reply = json.dumps({"refusal": "в выдержках нет порядка расчета"})
    answer = with_model(service, stub(reply)).answer(IN_CORPUS, mode=GENERATE)
    assert answer.answered is False
    assert "модель отказалась" in answer.refusal


def test_broken_model_does_not_break_the_service(service):
    def fails(system: str, user: str) -> str:
        raise GenerationError("модель не ответила: TimeoutError")

    answer = with_model(service, fails).answer(IN_CORPUS, mode=GENERATE)
    assert answer.answered is False
    assert "не ответила" in answer.refusal


def test_parse_reply_drops_numbers_outside_the_excerpts():
    text, used, refusal = parse_reply(json.dumps({"answer": "да", "used": [1, 7]}), excerpts=3)
    assert (text, used, refusal) == ("да", (1,), "")
    _, _, refusal = parse_reply(json.dumps({"answer": "да", "used": [9]}), excerpts=3)
    assert "вне выдачи" in refusal


def test_parse_reply_takes_a_fenced_object():
    reply = '```json\n{"answer": "да", "used": [1]}\n```'
    assert parse_reply(reply, excerpts=1)[0] == "да"


def test_unknown_mode_is_an_error(service):
    with pytest.raises(AnswerError, match="нет режима ответа"):
        service.answer(IN_CORPUS, mode="whatever")


def test_journal_line_carries_the_refusal_and_no_quotes(service):
    """В журнал едут числа и идентификаторы, а куски актов остаются в корпусе."""
    answer = service.answer(OUT_OF_CORPUS, top=5)
    event = events.answer_event(
        events.new_request_id(), events.utc_now(), 5, answer, service_version="0.1.0"
    )
    assert event["answer_outcome"] == "refusal"
    assert event["mode"] == EXCERPTS
    assert event["support"]["threshold"] == service.settings.threshold
    assert event["reason"] == answer.refusal
    assert set(event) <= {
        *events.QUERY_FIELDS,
        *events.ANSWERED_FIELDS,
        *events.ANSWER_FIELDS,
        "reason",
        "query_truncated_to",
    }

    answered = service.answer(IN_CORPUS, top=5)
    line = json.dumps(
        events.answer_event(events.new_request_id(), events.utc_now(), 5, answered, "0.1.0"),
        ensure_ascii=False,
    )
    assert '"answer_outcome":"answer"' in line.replace(", ", ",").replace('": ', '":')
    for citation in answered.citations:
        assert citation.quote[:40] not in line
