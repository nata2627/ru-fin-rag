"""Мониторинг трафика: дрейф распределения запросов и доля запросов без хорошего ответа.

Два измерения, оба считаются по журналу запросов, который уже едет в
репозиторий (`data/events`, DECISIONS, решение 12), и ничего дополнительно
хранить для них не надо. Третье измерение задачи, деградация на эталонном
наборе, живет отдельно: порог в `results/alert-threshold.json`, текст алерта
собирает `scripts/run_regression.py`. Здесь оно только печатается рядом.

Дрейф меряется PSI (индекс стабильности населения) по трем признакам текста
запроса, а не расхождением распределений слов: словарь запросов это десятки
тысяч бинов, и на окне в сотню запросов такая оценка состоит из шума целиком.
Признаки объяснимы по одному: сколько слов в запросе, какая доля слов отсутствует
в корпусе, назван ли номер акта. Правила и границы бинов записаны до чисел
(`orchestration/EXPERIMENTS.md`, S20), пороги выведены замером
(`scripts/measure_monitoring_thresholds.py`).

Доли эталона пересчитываются каждым прогоном на текущем словаре индекса, а не
читаются из файла замера: словарь растет вместе с корпусом, и иначе дрейфом
запросов считалась бы доливка актов. Границы бинов `tokens` при этом заморожены
в файле замера, иначе бины ездили бы вместе с окном.

«Хороший ответ» здесь счетный и названный прямо: это не суждение о тексте
ответа, которого в сервисе пока нет, а оценка лучшего кандидата поиска. Признак
слабый, и его ошибки в обе стороны лежат в файле замера рядом с порогом.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir
from .corpus import corpus_path, load_corpus
from .retrieval.bm25 import BM25Index
from .retrieval.text import _ACT_NUMBER, tokenize

THRESHOLDS_FILE = "results/monitoring-thresholds.json"
EVENT_GLOB = "queries-*.jsonl"

TOKENS = "tokens"
OOV = "oov_share"
ACT = "act_number"
FEATURES = (TOKENS, OOV, ACT)

# Границы бинов доли слов вне словаря. Назначены, а не выведены из данных, и это
# сознательно: ноль это отдельный случай («все слова запроса корпусу известны»),
# а дальше идут крупные шаги, в которых бин не пустеет на окне в сотню запросов.
OOV_EDGES = (0.0, 0.2, 0.5)
# Сглаживание Джеффриса. Без него пустой бин окна дает бесконечный PSI, то есть
# алерт срабатывает от одного необычного запроса.
SMOOTHING = 0.5

SUPPORT_FEATURES = ("top_score", "top_score_per_token")

# Номер акта опознается тем же выражением, которым его выделяет токенизация
# индекса: двух описаний одного формата в проекте быть не должно.
_ACT_TOKEN = re.compile(rf"^(?:{_ACT_NUMBER})$", re.I)


class MonitoringError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class QueryShape:
    """Признаки одного запроса: все считаются из строки журнала и словаря индекса."""

    tokens: int
    found: int
    act_number: bool
    hits: int
    top_score: float | None

    @property
    def oov_share(self) -> float:
        if not self.tokens:
            return 1.0
        return (self.tokens - self.found) / self.tokens

    @property
    def top_score_per_token(self) -> float:
        """Оценка на найденное слово запроса.

        Сырая оценка BM25 растет с числом слов запроса, а длина живого вопроса и
        длина синтетического различаются в разы, поэтому второй кандидат в
        признак опоры делится на число слов, которые вообще есть в словаре.
        """
        if self.top_score is None:
            return 0.0
        return self.top_score / self.found if self.found else 0.0

    def support(self, feature: str) -> float:
        if feature == "top_score":
            return 0.0 if self.top_score is None else self.top_score
        if feature == "top_score_per_token":
            return self.top_score_per_token
        raise MonitoringError(f"нет признака опоры {feature!r}, известны: {SUPPORT_FEATURES}")


def shape_of_tokens(
    tokens: list[str], vocab, hits: int = 0, top_score: float | None = None
) -> QueryShape:
    return QueryShape(
        tokens=len(tokens),
        found=sum(1 for token in tokens if token in vocab),
        act_number=any(_ACT_TOKEN.match(token) for token in tokens),
        hits=hits,
        top_score=top_score,
    )


def shape_of_text(text: str, vocab, hits: int = 0, top_score: float | None = None) -> QueryShape:
    return shape_of_tokens(tokenize(text), vocab, hits=hits, top_score=top_score)


def shape_of_event(event: dict, vocab) -> QueryShape:
    """Признаки по строке журнала.

    Токены берутся из поля `query_normalized`, то есть из того, что увидела
    лексическая ступень на самом деле, а не из повторной токенизации текста:
    версия стеммера могла с тех пор поменяться, и тогда мониторинг считал бы
    запрос иначе, чем его считал поиск.
    """
    normalized = event.get("query_normalized")
    tokens = normalized.split() if normalized is not None else tokenize(event.get("query", ""))
    results = event.get("results") or []
    return shape_of_tokens(
        tokens,
        vocab,
        hits=int(event.get("hits", len(results))),
        top_score=float(results[0]["score"]) if results else None,
    )


def token_edges(shapes: list[QueryShape]) -> list[int]:
    """Границы бинов длины запроса по квартилям эталона."""
    values = sorted(shape.tokens for shape in shapes)
    if not values:
        raise MonitoringError("пустой эталон: границы бинов не из чего считать")
    return [values[int(q * (len(values) - 1))] for q in (0.25, 0.5, 0.75)]


def bins_of(feature: str, edges: list[int]) -> int:
    if feature == TOKENS:
        return len(edges) + 1
    if feature == OOV:
        return len(OOV_EDGES) + 1
    if feature == ACT:
        return 2
    raise MonitoringError(f"нет признака {feature!r}, известны: {FEATURES}")


def bin_of(feature: str, shape: QueryShape, edges: list[int]) -> int:
    if feature == TOKENS:
        return sum(1 for edge in edges if shape.tokens > edge)
    if feature == OOV:
        share = shape.oov_share
        # Ноль это отдельный бин, поэтому сравнение строгое: доля 0,0 не попадает
        # в бин «до 0,2 включительно».
        return sum(1 for edge in OOV_EDGES if share > edge)
    if feature == ACT:
        return 1 if shape.act_number else 0
    raise MonitoringError(f"нет признака {feature!r}, известны: {FEATURES}")


def counts_of(shapes: list[QueryShape], edges: list[int]) -> dict[str, list[int]]:
    out = {feature: [0] * bins_of(feature, edges) for feature in FEATURES}
    for shape in shapes:
        for feature in FEATURES:
            out[feature][bin_of(feature, shape, edges)] += 1
    return out


def shares(counts: list[int]) -> list[float]:
    total = sum(counts) + SMOOTHING * len(counts)
    return [(count + SMOOTHING) / total for count in counts]


def psi(window: list[int], reference: list[int]) -> float:
    """PSI между окном и эталоном: сумма (p - q) * ln(p / q) по бинам."""
    if len(window) != len(reference):
        raise MonitoringError(f"бинов в окне {len(window)}, в эталоне {len(reference)}")
    p = shares(window)
    q = shares(reference)
    return sum((pi - qi) * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))


def drift(window: list[QueryShape], reference: list[QueryShape], edges: list[int]) -> dict:
    """Статистика дрейфа: PSI по каждому признаку и наибольший из них."""
    window_counts = counts_of(window, edges)
    reference_counts = counts_of(reference, edges)
    by_feature = {
        feature: round(psi(window_counts[feature], reference_counts[feature]), 4)
        for feature in FEATURES
    }
    worst = max(by_feature, key=lambda name: by_feature[name])
    return {
        "psi": by_feature[worst],
        "worst_feature": worst,
        "by_feature": by_feature,
        "window_counts": window_counts,
        "reference_counts": reference_counts,
        "window": len(window),
        "reference": len(reference),
    }


def answerless(shape: QueryShape, feature: str, floor: float) -> bool:
    """Запрос без хорошего ответа: кандидатов нет или опора лучшего ниже порога."""
    if shape.hits == 0:
        return True
    return shape.support(feature) < floor


def answerless_share(shapes: list[QueryShape], feature: str, floor: float) -> dict:
    bad = [shape for shape in shapes if answerless(shape, feature, floor)]
    return {
        "answerless": len(bad),
        "answered": len(shapes),
        "share": round(len(bad) / len(shapes), 4) if shapes else None,
        "no_hits": sum(1 for shape in shapes if shape.hits == 0),
    }


def youden(values: list[float], good: list[bool]) -> dict:
    """Точка максимума J Юдена: где признак опоры лучше всего делит набор.

    Положительный класс это «в первой десятке есть релевантный фрагмент».
    Предсказание «ответ есть» ставится при `значение >= порог`, порог берется из
    самих значений, при равенстве J берется меньший порог. Рядом считаются обе
    ошибки: сколько запросов с ответом признак назовет безответными и сколько
    безответных пропустит.
    """
    if len(values) != len(good):
        raise MonitoringError(f"значений {len(values)}, меток {len(good)}")
    positives = sum(good)
    negatives = len(good) - positives
    if not positives or not negatives:
        raise MonitoringError(f"один класс: с ответом {positives}, без ответа {negatives}")
    best: dict = {}
    for cut in sorted(set(values)):
        tp = sum(1 for value, flag in zip(values, good, strict=True) if flag and value >= cut)
        fp = sum(1 for value, flag in zip(values, good, strict=True) if not flag and value >= cut)
        j = tp / positives - fp / negatives
        if not best or j > best["j"]:
            best = {
                "floor": cut,
                "j": j,
                "tpr": tp / positives,
                "fpr": fp / negatives,
                "good_below_floor": positives - tp,
                "bad_above_floor": fp,
                "positives": positives,
                "negatives": negatives,
            }
    best["j"] = round(best["j"], 4)
    for name in ("floor", "tpr", "fpr"):
        best[name] = round(best[name], 4)
    return best


def load_index(base: Path | None = None) -> BM25Index:
    """Индекс, по которому считается словарь запроса.

    Собранный индекс читается с диска, а при его отсутствии собирается из
    корпуса: мониторингу нужен только словарь, но собирать его отдельной копией
    кода значило бы завести второй словарь, который разъедется с поиском.
    """
    base = base or data_dir()
    path = base / "index/bm25.npz"
    if path.exists():
        return BM25Index.load(path)
    corpus = load_corpus(corpus_path(base))
    return BM25Index.build(corpus.ids, list(corpus.iter_texts()))


def read_events(directory: Path) -> list[dict]:
    """Все события журнала из файлов суток, в порядке времени события."""
    rows: list[dict] = []
    for path in sorted(directory.glob(EVENT_GLOB)):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise MonitoringError(f"{path.name}, строка {number}: {exc}") from exc
    return sorted(rows, key=lambda row: row.get("received_at", ""))


def answered_window(events: list[dict], size: int) -> list[dict]:
    """Последние `size` запросов, дошедших до поиска.

    Окно считается по событиям, а не по суткам: трафика у сервиса из резюме
    единицы в день, и окно «за сутки» было бы пустым почти всегда.
    """
    answered = [row for row in events if row.get("kind") == "query" and row.get("outcome") == "ok"]
    return answered[-size:]


def event_totals(events: list[dict]) -> dict:
    queries = [row for row in events if row.get("kind") == "query"]
    answered = [row for row in queries if row.get("outcome") == "ok"]
    days = sorted({row.get("received_at", "")[:10] for row in queries if row.get("received_at")})
    return {
        "events": len(events),
        "queries": len(queries),
        "answered": len(answered),
        "rejected": sum(1 for row in queries if row.get("outcome") == "rejected"),
        "unavailable": sum(1 for row in queries if row.get("outcome") == "unavailable"),
        "feedback": sum(1 for row in events if row.get("kind") == "feedback"),
        "days": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
    }


def monitoring_rule(root: Path) -> dict:
    """Пороги мониторинга: числа из замера, а не из кода.

    Замер, не прошедший свою же проверку (порог дрейфа не отличает живые
    вопросы от эталона), пользоваться собой не дает: тогда мера на этом трафике
    ничего не ловит, и это ошибка, а не повод взять число по умолчанию. То же
    правило, что у порога деградации в `rufinrag.pipeline.alert_threshold`.
    """
    path = root / THRESHOLDS_FILE
    if not path.exists():
        raise MonitoringError(f"нет замера порогов {THRESHOLDS_FILE}: make monitoring-thresholds")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise MonitoringError(f"{THRESHOLDS_FILE}: замер порогов не прошел свою проверку")
    return report["rule_applied"]
