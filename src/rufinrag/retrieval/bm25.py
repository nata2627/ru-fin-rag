"""BM25 по фрагментам корпуса на разреженной матрице.

Перенесено из части А (`ru-fin-retrieval`, `src/rufin/retrieval/bm25.py`):
формула Okapi без прибавления единицы под логарифмом, k1 = 1.5, b = 0.75,
отрицательные idf заменяются долей среднего. В части А эта реализация сверена
с `rank_bm25` с расхождением 1,4e-06, а здесь тест сверяет выдачу сервиса с
выдачей бенчмарка на тех же запросах.

Почему разреженная матрица, а не `rank_bm25`: тот держит токенизированные
документы списками строк, и на 62 594 фрагментах это гигабайты. Замер на маке
21.10.2026: индекс 52,0 МБ, сборка 9,9 с, поиск при k = 50 медиана 1,24 мс.

Индекс сериализуется в `.npz`, потому что на бесплатном хостинге 10 секунд
сборки это 10 секунд недоступности сервиса на каждом холодном старте.
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .ranking import top_k
from .text import tokenize

K1 = 1.5
B = 0.75
# Доля среднего idf, которой заменяются отрицательные значения. У Okapi BM25
# idf слова, встречающегося более чем в половине документов, отрицателен, и без
# замены такое слово начинает работать против документов, где оно есть.
EPSILON = 0.25

FORMAT = 1


@dataclass
class BM25Index:
    ids: list[str]
    vocab: dict[str, int]
    matrix: csr_matrix  # слова на документы, значения это частоты
    idf: np.ndarray
    doc_norm: np.ndarray  # k1 * (1 - b + b * |d| / avgdl)
    build_seconds: float = 0.0
    k1: float = K1
    _scores: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str], k1: float = K1, b: float = B):
        t0 = time.monotonic()
        vocab: dict[str, int] = {}
        # Компактные массивы, а не списки питона: у корпуса около семи миллионов
        # ненулевых частот, и списки целых заняли бы под гигабайт только на
        # время сборки.
        rows = array("i")
        cols = array("i")
        vals = array("f")
        lengths = np.zeros(len(texts), dtype=np.float32)

        for col, text in enumerate(texts):
            counts: dict[int, int] = {}
            n = 0
            for tok in tokenize(text):
                idx = vocab.get(tok)
                if idx is None:
                    idx = vocab[tok] = len(vocab)
                counts[idx] = counts.get(idx, 0) + 1
                n += 1
            lengths[col] = n
            for idx, cnt in counts.items():
                rows.append(idx)
                cols.append(col)
                vals.append(cnt)

        matrix = csr_matrix(
            (
                np.frombuffer(vals, dtype=np.float32),
                (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32)),
            ),
            shape=(len(vocab), len(texts)),
        )
        df = np.diff(matrix.indptr).astype(np.float32)
        n_docs = len(texts)
        idf = (np.log(n_docs - df + 0.5) - np.log(df + 0.5)).astype(np.float32)
        if idf.size:
            idf = np.where(idf < 0, EPSILON * float(idf.mean()), idf).astype(np.float32)
        avgdl = float(lengths.mean()) if n_docs else 0.0
        doc_norm = (k1 * (1.0 - b + b * lengths / (avgdl or 1.0))).astype(np.float32)

        return cls(
            ids=list(chunk_ids),
            vocab=vocab,
            matrix=matrix,
            idf=idf,
            doc_norm=doc_norm,
            build_seconds=time.monotonic() - t0,
            k1=k1,
        )

    def scores(self, query: str) -> np.ndarray:
        """Оценки по всем документам. Буфер переиспользуется между запросами."""
        if self._scores is None or self._scores.shape[0] != len(self.ids):
            self._scores = np.zeros(len(self.ids), dtype=np.float32)
        out = self._scores
        out.fill(0.0)
        for tok in tokenize(query):
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            start, end = self.matrix.indptr[idx], self.matrix.indptr[idx + 1]
            docs = self.matrix.indices[start:end]
            tf = self.matrix.data[start:end]
            out[docs] += self.idf[idx] * tf * (self.k1 + 1.0) / (tf + self.doc_norm[docs])
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = top_k(s, k)
        return [(self.ids[i], float(s[i])) for i in order[:k]]

    @property
    def size_mb(self) -> float:
        return (
            self.matrix.data.nbytes
            + self.matrix.indices.nbytes
            + self.matrix.indptr.nbytes
            + self.idf.nbytes
            + self.doc_norm.nbytes
        ) / 1048576

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.vocab, key=self.vocab.get)
        np.savez_compressed(
            path,
            format=np.array([FORMAT]),
            ids=np.array(self.ids),
            terms=np.array(terms),
            data=self.matrix.data,
            indices=self.matrix.indices,
            indptr=self.matrix.indptr,
            shape=np.array(self.matrix.shape),
            idf=self.idf,
            doc_norm=self.doc_norm,
            k1=np.array([self.k1], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            if int(z["format"][0]) != FORMAT:
                raise ValueError(
                    f"{path}: формат индекса {int(z['format'][0])}, ожидается {FORMAT}. "
                    "Пересобрать: python scripts/build_index.py"
                )
            terms = [str(t) for t in z["terms"]]
            matrix = csr_matrix(
                (z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"])
            )
            return cls(
                ids=[str(i) for i in z["ids"]],
                vocab={t: i for i, t in enumerate(terms)},
                matrix=matrix,
                idf=z["idf"],
                doc_norm=z["doc_norm"],
                k1=float(z["k1"][0]),
            )
