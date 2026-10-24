"""Ответ по выдаче поиска: выдержки с цитатами, отказ при отсутствии опоры.

Два режима. Основной это выдержки: сервис отдает куски найденных фрагментов со
ссылкой на акт, раздел, пункты и адрес источника. Второй режим включается
ключом модели из окружения, и тогда ответ пишет модель строго по тем же
фрагментам. Ключа у проекта нет и может не появиться вовсе, поэтому первый
режим это рабочее состояние сервиса, а не обрубок второго (PLAN, этап 7).

Отказ счетный и объяснимый одним числом. Признаки опоры те же, что у монитора
доли запросов без хорошего ответа (`rufinrag.monitoring`, задача 6.4), плюс
третий, `idf_coverage`: двух разных «отсутствий опоры» в проекте быть не
должно, иначе метрика считает одно, а сервис делает другое. Замер выбрал третий
признак, потому что оценка BM25 растет с длиной запроса (корреляция Спирмена
0,86), и порог, выведенный на длинных вопросах эталона, отказывал короткому
запросу по длине, а не по отсутствию опоры.

Порог свой, а не взятый у монитора: у монитора он стоит в точке равенства
ошибок, а здесь выведен из объявленного бюджета ложных отказов, потому что
отказ на живом вопросе практика дороже слабого ответа, рядом с которым
напечатаны номер акта, пункт и ссылка. Правило и бюджет записаны до чисел
(`orchestration/EXPERIMENTS.md`, S21), числа лежат в `results/answer.json`.
Главное число там отрицательное, и его видно: отказ ловит вопросы с чужим
словарем, а не отсутствием ответа: вопрос из соседней отрасли права, где слова
те же, он не отклонит. Измерены только ложные отказы (9 из 991): набора
вопросов из чужой области в проекте нет.

Ссылка собирается только из того, что в корпусе есть. Отдельного поля с
номером пункта у фрагмента нет вовсе, поэтому номера пунктов вынимаются из
текста фрагмента и поле называется тем, чем является: номера, напечатанные в
этом фрагменте. Границы цитаты внутри текста фрагмента печатаются рядом с
цитатой, чтобы ее можно было проверить по корпусу, не веря сервису на слово.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ConfigError, env, project_root
from .corpus import Chunk
from .generation import (
    SYSTEM,
    Completer,
    GenerationError,
    GenerationSettings,
    build_prompt,
    http_completer,
    parse_reply,
    settings_from_env,
)
from .monitoring import SUPPORT_FEATURES as MONITOR_FEATURES
from .monitoring import QueryShape, answerless, shape_of_text
from .retrieval.bm25 import BM25Index
from .retrieval.text import _ACT_NUMBER, stem, token_spans
from .search import Hit, SearchResult, SearchService

ANSWER_CONFIG = Path("configs") / "answer.yaml"

EXCERPTS = "excerpts"
GENERATE = "generate"
MODES = (EXCERPTS, GENERATE)

IDF_COVERAGE = "idf_coverage"
SUPPORT_FEATURES = (*MONITOR_FEATURES, IDF_COVERAGE)

# Номер пункта по началу строки: «3.10.» или «3.14.3». Одноуровневые номера
# («3.») не берутся: ими в актах нумеруются и пункты, и позиции перечислений, и
# строки таблиц, и отличить их по тексту нельзя.
_CLAUSE = re.compile(r"(?m)^[ \t]*(\d+(?:\.\d+)+)\.?(?=[\s)])")
MAX_CLAUSES = 8

# Номер акта в тексте ответа модели. Выражение то же, что у токенизации индекса:
# двух описаний одного формата в проекте быть не должно.
_ACT_IN_TEXT = re.compile(_ACT_NUMBER, re.I)

# Сколько символов назад и вперед цитате разрешено доехать до границы
# предложения. Нужно затем, что окно открывается на слове запроса и без этого
# начинается с середины слова.
SNAP_BACK = 160
SNAP_FORWARD = 80
_SENTENCE_END = re.compile(r"[.;:!?]\s")


class AnswerError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class AnswerSettings:
    feature: str
    threshold: float
    depth: int
    citations: int
    quote_chars: int
    generation_excerpts: int
    max_tokens: int
    temperature: float
    timeout_seconds: float
    measured_in: str

    @property
    def version(self) -> str:
        """Версия слоя ответа для ответа API и для строки журнала."""
        return f"{self.feature}>={self.threshold:g}+depth{self.depth}+quote{self.quote_chars}"


def answer_config_path() -> Path:
    override = env("RUFINRAG_ANSWER_CONFIG")
    return Path(override).expanduser() if override else project_root() / ANSWER_CONFIG


def load_answer_settings(path: Path | None = None) -> AnswerSettings:
    path = path or answer_config_path()
    if not path.exists():
        raise ConfigError(f"нет файла конфигурации ответа {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: ожидался объект настроек")
    support = _section(raw, "support", path)
    excerpts = _section(raw, "excerpts", path)
    generation = _section(raw, "generation", path)
    feature = str(_required(support, "feature", path))
    if feature not in SUPPORT_FEATURES:
        raise ConfigError(
            f"{path}: признак опоры {feature!r} неизвестен, есть {', '.join(SUPPORT_FEATURES)}"
        )
    return AnswerSettings(
        feature=feature,
        threshold=float(_required(support, "threshold", path)),
        depth=int(_required(support, "depth", path)),
        citations=int(_required(excerpts, "citations", path)),
        quote_chars=int(_required(excerpts, "quote_chars", path)),
        generation_excerpts=int(_required(generation, "excerpts", path)),
        max_tokens=int(_required(generation, "max_tokens", path)),
        temperature=float(_required(generation, "temperature", path)),
        timeout_seconds=float(_required(generation, "timeout_seconds", path)),
        measured_in=str(_required(support, "measured_in", path)),
    )


def _section(raw: dict, name: str, path: Path) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: нет раздела {name}")
    return section


def _required(section: dict, key: str, path: Path):
    if section.get(key) in (None, ""):
        raise ConfigError(f"{path}: нет обязательного поля {key}")
    return section[key]


@dataclass(frozen=True, slots=True)
class Citation:
    """Ссылка на фрагмент: только поля, которые в корпусе есть.

    `clauses` это номера пунктов, напечатанные в тексте этого фрагмента, а не
    поле корпуса: такого поля у фрагмента нет. `quote_from` и `quote_to` это
    границы цитаты в символах внутри текста фрагмента, `char_start` и
    `char_end` это границы самого фрагмента внутри текста акта.
    """

    chunk_id: str
    act: str
    act_id: str
    number: str
    date: str
    title: str
    section: str
    clauses: tuple[str, ...]
    url: str
    char_start: int
    char_end: int
    quote: str
    quote_from: int
    quote_to: int
    quote_terms: int
    rank: int
    score: float
    idf_coverage: float

    @property
    def reference(self) -> str:
        parts = [self.act]
        if self.section:
            parts.append(self.section)
        if self.clauses:
            head = "пункт" if len(self.clauses) == 1 else "пункты"
            parts.append(f"{head} {', '.join(self.clauses)}")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "reference": self.reference,
            "act": self.act,
            "act_id": self.act_id,
            "number": self.number,
            "date": self.date,
            "title": self.title,
            "section": self.section,
            "clauses": list(self.clauses),
            "url": self.url,
            "chunk_chars": [self.char_start, self.char_end],
            "quote": self.quote,
            "quote_chars": [self.quote_from, self.quote_to],
            "quote_terms": self.quote_terms,
            "rank": self.rank,
            "score": round(self.score, 6),
            "idf_coverage": round(self.idf_coverage, 4),
        }


@dataclass(frozen=True, slots=True)
class Support:
    """Опора под вопросом одним числом и всем, из чего это число собрано."""

    feature: str
    value: float
    threshold: float
    passed: bool
    hits: int
    depth: int
    query_tokens: int
    tokens_in_corpus: int
    top_score: float
    best_chunk_id: str
    best_rank: int

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "hits": self.hits,
            "depth": self.depth,
            "query_tokens": self.query_tokens,
            "tokens_in_corpus": self.tokens_in_corpus,
            "top_score": round(self.top_score, 6),
            "best_chunk_id": self.best_chunk_id,
            "best_rank": self.best_rank,
        }

    @property
    def reason(self) -> str:
        if not self.hits:
            return "поиск не нашел ни одного фрагмента по словам этого запроса"
        value = f"{self.value:.2f}".replace(".", ",")
        threshold = f"{self.threshold:g}".replace(".", ",")
        return (
            f"опоры в корпусе нет: признак {self.feature} у лучшего из {self.hits} "
            f"найденных фрагментов {value} при пороге {threshold}. "
            f"Слов в запросе {self.query_tokens}, из них в словаре корпуса "
            f"{self.tokens_in_corpus}"
        )


@dataclass(frozen=True, slots=True)
class Answer:
    query: str
    mode: str
    answered: bool
    text: str
    text_source: str
    refusal: str
    citations: tuple[Citation, ...]
    support: Support
    search: SearchResult
    answer_version: str
    timings_ms: dict[str, float]


def clauses_in(text: str, limit: int = MAX_CLAUSES) -> tuple[str, ...]:
    """Номера пунктов, напечатанные в тексте фрагмента, в порядке появления."""
    out: list[str] = []
    for match in _CLAUSE.finditer(text):
        number = match.group(1)
        if number not in out:
            out.append(number)
            if len(out) == limit:
                break
    return tuple(out)


def header_length(chunk: Chunk, text: str) -> int:
    """Длина шапки фрагмента, если она на месте.

    Шапку («вид акта, номер, дата, название» и пустая строка) приписывает
    нарезка части А: так лексическая ступень находит фрагмент по номеру акта из
    вопроса (S9). В цитату шапка не идет, иначе каждая выдержка начиналась бы с
    одного и того же заголовка вместо текста нормы.

    Шапка опознается по номеру акта в первом абзаце, а не сравнением с
    `label` и `title`: название в шапке напечатано иначе, чем лежит в поле
    (кавычки, длина), и точное сравнение нашло шапку только у 6094 фрагментов
    из 62 606, а это правило у всех 62 606.
    """
    head, separator, _ = text.partition("\n\n")
    if separator and f"№ {chunk.act.number}" in head:
        return len(head) + len(separator)
    return 0


def idf_weights(bm25: BM25Index, query: str) -> dict[str, float]:
    """Вес каждого слова запроса: его idf в индексе.

    Слову, которого в словаре корпуса нет вовсе, дается наибольший idf индекса:
    такое слово самое редкое из возможных, и фрагмент, который его не содержит,
    запрос не покрывает. Без этого вопрос про борщ выглядел бы полностью
    покрытым любым фрагментом со словом «приготовление».
    """
    unseen = float(bm25.idf.max()) if bm25.idf.size else 0.0
    weights: dict[str, float] = {}
    for token, _, _ in token_spans(query):
        term = stem(token)
        if term in weights:
            continue
        index = bm25.vocab.get(term)
        weights[term] = float(bm25.idf[index]) if index is not None else unseen
    return weights


def coverage_of(weights: dict[str, float], terms: set[str]) -> float:
    total = sum(weights.values())
    if not total:
        return 0.0
    return sum(weight for term, weight in weights.items() if term in terms) / total


def pick_quote(text: str, body_start: int, terms: set[str], limit: int) -> tuple[int, int, int]:
    """Окно текста, в котором стоит больше всего разных слов запроса.

    Возвращает границы и число слов запроса внутри. Окно открывается на слове
    запроса и доводится до границы предложения: иначе цитата начинается с
    середины слова, и читать ее нельзя.
    """
    marks = [
        (start, end, stem(token))
        for token, start, end in token_spans(text)
        if start >= body_start and stem(token) in terms
    ]

    best_terms, best_start = 0, body_start
    right = 0
    for left in range(len(marks)):
        window_end = marks[left][0] + limit
        right = max(right, left)
        while right < len(marks) and marks[right][1] <= window_end:
            right += 1
        found = len({term for _, _, term in marks[left:right]})
        if found > best_terms:
            best_terms, best_start = found, marks[left][0]

    start = _snap_back(text, body_start, best_start)
    end = _snap_forward(text, start + limit)
    return start, end, best_terms


def _snap_back(text: str, body_start: int, start: int) -> int:
    """Начало цитаты сдвигается назад до конца предыдущего предложения."""
    left = max(body_start, start - SNAP_BACK)
    ends = list(_SENTENCE_END.finditer(text[left:start]))
    if ends:
        return left + ends[-1].end()
    return left if left == body_start else start


def _snap_forward(text: str, end: int) -> int:
    if end >= len(text):
        return len(text)
    window = text[end : end + SNAP_FORWARD]
    match = _SENTENCE_END.search(window)
    if match:
        return end + match.end()
    space = window.find(" ")
    return end + space if space >= 0 else end


def quote_text(text: str, start: int, end: int) -> str:
    """Цитата одной строкой: переносы строк в нормативном тексте идут по ширине
    колонки, и оставлять их значит рвать фразу посреди слова на экране."""
    return " ".join(text[start:end].split())


@dataclass
class AnswerService:
    """Слой ответа над поиском. Держит настройки и, если ключ есть, клиента модели."""

    search: SearchService
    settings: AnswerSettings
    generation: GenerationSettings | None = None
    generation_off: str = ""
    completer: Completer | None = None

    @classmethod
    def build(
        cls,
        search: SearchService,
        settings: AnswerSettings | None = None,
        completer: Completer | None = None,
    ):
        settings = settings or load_answer_settings()
        generation, off = settings_from_env(
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            timeout_seconds=settings.timeout_seconds,
        )
        if generation is not None and completer is None:
            completer = http_completer(generation)
        return cls(
            search=search,
            settings=settings,
            generation=generation,
            generation_off=off,
            completer=completer if generation is not None else None,
        )

    @property
    def generation_ready(self) -> bool:
        return self.generation is not None and self.completer is not None

    @property
    def generation_model(self) -> str:
        return self.generation.model if self.generation is not None else ""

    def support_of(self, result: SearchResult, query: str) -> tuple[Support, dict[str, float]]:
        """Опора под запросом и покрытие каждого фрагмента выдачи.

        Признаки монитора считаются его же кодом, по тем же токенам и тому же
        словарю, которыми пользуется метрика доли запросов без хорошего ответа.
        """
        return self._support(result, idf_weights(self.search.bm25, query), query)

    def _support(
        self, result: SearchResult, weights: dict[str, float], query: str
    ) -> tuple[Support, dict[str, float]]:
        bm25 = self.search.bm25
        depth = self.settings.depth
        covered: dict[str, float] = {}
        for hit in result.hits[:depth]:
            text = self.search.corpus.text(hit.chunk_id)
            terms = {stem(token) for token, _, _ in token_spans(text)}
            covered[hit.chunk_id] = coverage_of(weights, terms)

        shape = shape_of_text(
            query,
            bm25.vocab,
            hits=len(result.hits),
            top_score=result.hits[0].score if result.hits else None,
        )
        best_id, best_coverage = "", 0.0
        if covered:
            best_id, best_coverage = max(covered.items(), key=lambda item: item[1])
        value = _support_value(self.settings.feature, shape, best_coverage)
        best_rank = next((h.rank for h in result.hits if h.chunk_id == best_id), 0)
        return (
            Support(
                feature=self.settings.feature,
                value=value,
                threshold=self.settings.threshold,
                passed=not _refused(self.settings.feature, self.settings.threshold, shape, value),
                hits=len(result.hits),
                depth=min(depth, len(result.hits)),
                query_tokens=shape.tokens,
                tokens_in_corpus=shape.found,
                top_score=result.hits[0].score if result.hits else 0.0,
                best_chunk_id=best_id,
                best_rank=best_rank,
            ),
            covered,
        )

    def answer(
        self,
        query: str,
        top: int = 10,
        config: str | None = None,
        mode: str | None = None,
    ) -> Answer:
        mode = mode or EXCERPTS
        if mode not in MODES:
            raise AnswerError(f"нет режима ответа {mode!r}, есть {', '.join(MODES)}")
        if mode == GENERATE and not self.generation_ready:
            raise AnswerError(f"генеративный режим недоступен: {self.generation_off}")

        result = self.search.search(query, top=top, config=config)
        weights = idf_weights(self.search.bm25, query)
        t0 = time.monotonic()
        support, covered = self._support(result, weights, query)
        timings = dict(result.timings_ms)
        timings["support"] = round((time.monotonic() - t0) * 1000, 3)

        if not support.passed:
            return self._refusal(query, mode, support.reason, support, result, timings)

        t0 = time.monotonic()
        citations = self._citations(result, covered, set(weights), limit=self._limit(mode))
        timings["citations"] = round((time.monotonic() - t0) * 1000, 3)
        if mode == GENERATE:
            return self._generated(query, citations, support, result, timings)
        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 3)
        return Answer(
            query=query,
            mode=mode,
            answered=True,
            text=self._excerpt_text(citations),
            text_source="выдержки из найденных фрагментов",
            refusal="",
            citations=citations,
            support=support,
            search=result,
            answer_version=self.settings.version,
            timings_ms=timings,
        )

    def _limit(self, mode: str) -> int:
        return self.settings.citations if mode == EXCERPTS else self.settings.generation_excerpts

    def _refusal(
        self,
        query: str,
        mode: str,
        reason: str,
        support: Support,
        result: SearchResult,
        timings: dict[str, float],
    ) -> Answer:
        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 3)
        return Answer(
            query=query,
            mode=mode,
            answered=False,
            text="",
            text_source="",
            refusal=reason,
            citations=(),
            support=support,
            search=result,
            answer_version=self.settings.version,
            timings_ms=timings,
        )

    def _citations(
        self, result: SearchResult, covered: dict[str, float], terms: set[str], limit: int
    ) -> tuple[Citation, ...]:
        return tuple(
            self._citation(hit, terms, covered.get(hit.chunk_id, 0.0))
            for hit in result.hits[:limit]
        )

    def _citation(self, hit: Hit, terms: set[str], coverage: float) -> Citation:
        chunk = self.search.corpus.get(hit.chunk_id)
        text = self.search.corpus.text(hit.chunk_id)
        body_start = header_length(chunk, text)
        start, end, found = pick_quote(text, body_start, terms, self.settings.quote_chars)
        return Citation(
            chunk_id=chunk.chunk_id,
            act=chunk.act.label,
            act_id=chunk.act_id,
            number=chunk.act.number,
            date=chunk.act.date,
            title=chunk.act.title,
            section=chunk.section,
            clauses=clauses_in(text[body_start:]),
            url=chunk.act.url,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            quote=quote_text(text, start, end),
            quote_from=start,
            quote_to=end,
            quote_terms=found,
            rank=hit.rank,
            score=hit.score,
            idf_coverage=coverage,
        )

    def _excerpt_text(self, citations: tuple[Citation, ...]) -> str:
        head = (
            f"Ответ собран из выдержек найденных актов, своими словами он не формулируется: "
            f"{self.generation_off or 'генеративный режим выключен'}. "
            f"Выдержек {len(citations)}, ссылка на источник стоит у каждой."
        )
        blocks = [head, ""]
        for number, citation in enumerate(citations, start=1):
            blocks.append(f"{number}. {citation.reference}")
            blocks.append(f"«{citation.quote}»")
            blocks.append(citation.url)
            blocks.append("")
        return "\n".join(blocks).strip()

    def _generated(
        self,
        query: str,
        citations: tuple[Citation, ...],
        support: Support,
        result: SearchResult,
        timings: dict[str, float],
    ) -> Answer:
        if self.completer is None:
            raise AnswerError(f"генеративный режим недоступен: {self.generation_off}")
        excerpts = [(c.reference, c.quote) for c in citations]
        t0 = time.monotonic()
        try:
            reply = self.completer(SYSTEM, build_prompt(query, excerpts))
        except GenerationError as exc:
            timings["generation"] = round((time.monotonic() - t0) * 1000, 3)
            return self._refusal(query, GENERATE, str(exc), support, result, timings)
        timings["generation"] = round((time.monotonic() - t0) * 1000, 3)

        text, used, refusal = parse_reply(reply, excerpts=len(excerpts))
        if refusal:
            return self._refusal(query, GENERATE, refusal, support, result, timings)
        kept = tuple(citations[number - 1] for number in used)
        stray = _stray_numbers(text, kept)
        if stray:
            # Модель назвала акт, которого в выдержках нет. Такой ответ не
            # публикуется: проверить его по корпусу нечем, и это ровно та
            # выдумка, от которой отказ и защищает.
            return self._refusal(
                query,
                GENERATE,
                f"ответ модели ссылается на акты вне выдачи: {', '.join(stray)}",
                support,
                result,
                timings,
            )
        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 3)
        return Answer(
            query=query,
            mode=GENERATE,
            answered=True,
            text=text,
            text_source=f"модель {self.generation_model} по найденным фрагментам",
            refusal="",
            citations=kept,
            support=support,
            search=result,
            answer_version=self.settings.version,
            timings_ms=timings,
        )


def _support_value(feature: str, shape: QueryShape, coverage: float) -> float:
    if feature == IDF_COVERAGE:
        return coverage
    return shape.support(feature)


def _refused(feature: str, threshold: float, shape: QueryShape, value: float) -> bool:
    """То же правило, что у монитора, с третьим признаком.

    Признаки монитора считает он сам (`monitoring.answerless`), чтобы метрика
    доли запросов без хорошего ответа и поведение сервиса не разъехались.
    """
    if feature == IDF_COVERAGE:
        return shape.hits == 0 or value < threshold
    return answerless(shape, feature, threshold)


def _stray_numbers(text: str, citations: tuple[Citation, ...]) -> tuple[str, ...]:
    """Номера актов из текста модели, которых нет среди процитированных."""
    known = {c.number.lower() for c in citations}
    out: list[str] = []
    for match in _ACT_IN_TEXT.finditer(text):
        number = match.group(0)
        if number.lower() not in known and number not in out:
            out.append(number)
    return tuple(out)
