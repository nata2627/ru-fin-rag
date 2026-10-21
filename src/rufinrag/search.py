"""Поиск как сервис: связка собирается по конфигурации, а не по коду.

Ступени по порядку: лексический поиск BM25, плотный поиск по векторам, слияние
обратными рангами, необязательное переранжирование кросс-энкодером. Какие
ступени включены, решает файл `configs/retrieval.yaml` (SPEC, раздел 3).

Задержка возвращается по ступеням: без этого непонятно, что именно дорого, а
на бесплатном CPU дорого переранжирование (48,0 с при глубине 50).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Catalog, RetrievalConfig, data_dir, load_catalog
from .corpus import Chunk, Corpus, corpus_path, load_corpus
from .retrieval.bm25 import BM25Index
from .retrieval.dense import DenseIndex, QueryEncoder
from .retrieval.fusion import rrf
from .retrieval.rerank import Reranker

BM25_FILE = "index/bm25.npz"
TEXTS_FILE = "index/texts.blocks"
VECTORS_FILE = "vectors/rufin-384.npy"
VECTOR_IDS_FILE = "vectors/rufin-384.ids.txt"
DENSE_MODEL_DIR = "models/rufin"


class SearchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    score: float
    rank: int
    act_id: str
    act: str
    title: str
    section: str
    url: str
    text: str

    @classmethod
    def of(cls, chunk: Chunk, text: str, score: float, rank: int, snippet: int = 600):
        return cls(
            chunk_id=chunk.chunk_id,
            score=score,
            rank=rank,
            act_id=chunk.act_id,
            act=chunk.act.label,
            title=chunk.act.title,
            section=chunk.section,
            url=chunk.act.url,
            text=text[:snippet],
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    config: str
    config_version: str
    corpus_version: str
    hits: list[Hit]
    timings_ms: dict[str, float]
    degraded: list[str]


@dataclass
class SearchService:
    """Держит корпус и индексы в памяти и отвечает на запросы.

    Собирается методом `build`, потому что чтение корпуса и индекса это секунды
    и делается один раз на старте процесса, а не на каждый запрос.
    """

    corpus: Corpus
    bm25: BM25Index
    catalog: Catalog
    dense: DenseIndex | None = None
    encoder: QueryEncoder | None = None
    rerankers: dict[str, Reranker] = field(default_factory=dict)
    load_seconds: float = 0.0
    degraded: list[str] = field(default_factory=list)
    dense_off: str = ""

    @classmethod
    def build(
        cls,
        base: Path | None = None,
        catalog: Catalog | None = None,
        with_dense: bool = True,
    ):
        base = base or data_dir()
        catalog = catalog or load_catalog()
        t0 = time.monotonic()
        corpus = load_corpus(corpus_path(base), base / TEXTS_FILE)

        index_path = base / BM25_FILE
        if index_path.exists():
            bm25 = BM25Index.load(index_path)
            if bm25.ids != corpus.ids:
                raise SearchError(
                    f"{index_path} собран на другом корпусе: в индексе {len(bm25.ids)} "
                    f"фрагментов, в корпусе {len(corpus)}. Пересобрать: "
                    "python scripts/build_index.py"
                )
        else:
            bm25 = BM25Index.build(corpus.ids, list(corpus.iter_texts()))

        degraded: list[str] = []
        dense_off: str = ""
        dense = None
        encoder = None
        vectors = base / VECTORS_FILE
        model_dir = base / DENSE_MODEL_DIR
        if not with_dense:
            dense_off = "выключен переменной RUFINRAG_DISABLE_DENSE"
        elif not vectors.exists() or not model_dir.exists():
            # Базовая установка без плотной части это штатный режим, а не
            # поломка: иначе проверка здоровья на хостинге считает рабочий
            # сервис сломанным и перезапускает контейнер по кругу.
            dense_off = f"нет артефактов плотной части: {vectors.name}, {model_dir.name}"
        else:
            try:
                # Модель грузится до индекса: при бэкенде faiss обратный
                # порядок роняет процесс (докстринг retrieval/dense.py).
                encoder = QueryEncoder(model_dir)
                dense = DenseIndex.load(
                    vectors,
                    base / VECTOR_IDS_FILE,
                    backend=os.environ.get("RUFINRAG_DENSE_BACKEND") or "numpy",
                )
                if dense.ids != corpus.ids:
                    raise SearchError(
                        f"{vectors}: векторы собраны на другом корпусе "
                        f"({len(dense.ids)} против {len(corpus)})"
                    )
            except SearchError:
                raise
            except Exception as exc:
                # Артефакты на месте, а поиск не поднялся: это уже поломка, и
                # она попадает в `/health` как деградация (SPEC, раздел 6).
                degraded.append(f"плотный поиск недоступен: {exc}")
                dense = None
                encoder = None

        return cls(
            corpus=corpus,
            bm25=bm25,
            catalog=catalog,
            dense=dense,
            encoder=encoder,
            load_seconds=time.monotonic() - t0,
            degraded=degraded,
            dense_off=dense_off,
        )

    @property
    def corpus_version(self) -> str:
        return self.corpus.version

    def search(self, query: str, top: int = 10, config: str | None = None) -> SearchResult:
        if not query.strip():
            raise SearchError("пустой запрос")
        cfg = self.catalog.get(config)
        timings: dict[str, float] = {}
        degraded = list(self.degraded)

        candidates, stage = self._first_stage(query, cfg, timings, degraded)
        if cfg.rerank:
            reranked, seconds = self._reranker(cfg).rerank(
                query, candidates[: cfg.rerank.depth], dict(self._texts(candidates)), top=top
            )
            timings["rerank"] = seconds * 1000
            candidates = reranked
        candidates = candidates[:top]

        hits = [
            Hit.of(self.corpus.get(chunk_id), self.corpus.text(chunk_id), score, rank)
            for rank, (chunk_id, score) in enumerate(candidates, start=1)
        ]
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return SearchResult(
            query=query,
            config=cfg.name,
            config_version=cfg.version if stage == cfg.first_stage else f"{cfg.version}+degraded",
            corpus_version=self.corpus_version,
            hits=hits,
            timings_ms={k: round(v, 3) for k, v in timings.items()},
            degraded=degraded,
        )

    def _first_stage(
        self,
        query: str,
        cfg: RetrievalConfig,
        timings: dict[str, float],
        degraded: list[str],
    ) -> tuple[list[tuple[str, float]], str]:
        lexical: list[tuple[str, float]] = []
        if cfg.first_stage in ("bm25", "hybrid"):
            t0 = time.monotonic()
            lexical = self.bm25.search(query, k=cfg.candidates)
            timings["bm25"] = (time.monotonic() - t0) * 1000
            if cfg.first_stage == "bm25":
                return lexical, "bm25"

        if self.dense is None or self.encoder is None:
            if not lexical:
                t0 = time.monotonic()
                lexical = self.bm25.search(query, k=cfg.candidates)
                timings["bm25"] = (time.monotonic() - t0) * 1000
            reason = self.dense_off or "плотный поиск не поднялся"
            degraded.append(
                f"связка потребовала плотный поиск, отдана лексическая выдача: {reason}"
            )
            return lexical, "bm25"

        t0 = time.monotonic()
        vector = self.encoder.encode_one(query)
        timings["encode"] = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        dense_hits = self.dense.search(vector, k=cfg.candidates)
        timings["dense"] = (time.monotonic() - t0) * 1000
        if cfg.first_stage == "dense":
            return dense_hits, "dense"

        t0 = time.monotonic()
        fused = rrf([lexical, dense_hits], k=cfg.rrf_k, top=cfg.candidates)
        timings["fusion"] = (time.monotonic() - t0) * 1000
        return fused, "hybrid"

    def _reranker(self, cfg: RetrievalConfig) -> Reranker:
        spec = cfg.rerank
        if spec is None:
            raise SearchError(f"связка {cfg.name} без переранжирования")
        if spec.model not in self.rerankers:
            self.rerankers[spec.model] = Reranker(
                model_path=spec.model,
                batch_size=spec.batch_size,
                max_length=spec.max_length,
            )
        return self.rerankers[spec.model]

    def _texts(self, candidates: list[tuple[str, float]]):
        for chunk_id, _ in candidates:
            yield chunk_id, self.corpus.text(chunk_id)
