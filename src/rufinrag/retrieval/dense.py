"""Плотный поиск: векторы фрагментов и кодирование запроса.

Поиск точный, матричным умножением по нормированным векторам: 62 594 вектора
по 384 измерения это 48 МБ в float16, и приближенный индекс здесь не нужен.
Сравнение с Qdrant и FAISS идет отдельной задачей и отдельным замером.

Модель запроса грузится лениво, потому что torch нужен не всем режимам: связка
по умолчанию в онлайне плотную часть использует, а тесты BM25 и сборка индекса
нет (DECISIONS, решение 7).

Префикс «query: » обязателен: модель дообучена из multilingual-e5-small, и без
префикса качество проседает на ровном месте. Корпус индексировался с «passage: »,
то есть обе стороны должны идти со своими пометками.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
MAX_SEQ_LENGTH = 512


class DenseError(Exception):
    pass


@dataclass
class DenseIndex:
    ids: list[str]
    vectors: np.ndarray  # (n, dim), нормированные

    @classmethod
    def load(cls, vectors_path: Path, ids_path: Path):
        if not vectors_path.exists():
            raise DenseError(
                f"нет векторов {vectors_path}. Собрать из части А: "
                "python scripts/import_corpus.py"
            )
        vectors = np.load(vectors_path)
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines()]
        if len(ids) != vectors.shape[0]:
            raise DenseError(
                f"{vectors_path}: векторов {vectors.shape[0]}, идентификаторов {len(ids)}"
            )
        return cls(ids=ids, vectors=vectors)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size_mb(self) -> float:
        return self.vectors.nbytes / 1048576

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        # float32 для умножения: в float16 накапливается ошибка, а выигрыша по
        # времени на таком размере нет.
        scores = self.vectors.astype(np.float32) @ query_vector.astype(np.float32)
        if k >= scores.shape[0]:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return [(self.ids[i], float(scores[i])) for i in order[:k]]


class QueryEncoder:
    """Кодирование запроса усреднением по токенам с маской и нормировкой.

    Через `transformers` напрямую, а не через `sentence-transformers`: в весах
    части А записаны имена классов той версии библиотеки, что стояла в сессии
    Kaggle, и локальная версия их не находит. Обертка все равно не делает
    ничего, кроме среднего по маске и нормировки.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_path = str(model_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path).to(device).eval()

    def encode(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        torch = self._torch
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        batch = self.tokenizer(
            [prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
