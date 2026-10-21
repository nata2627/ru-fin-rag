"""Переранжирование кросс-энкодером.

Кросс-энкодер читает запрос и документ вместе, поэтому качество выше
би-энкодера: в части А он дал плюс 0,207 NDCG@10 при p меньше 0,001, и это
единственный установленный там выигрыш. Цена замерена 21.10.2026 на маке:
48,0 с на запрос на CPU при глубине 50 и 12,5 с при глубине 10, на MPS 20,4 с и
3,4 с (`results/artifacts.md`). Поэтому в онлайне по умолчанию он выключен и
живет как конфигурация поиска (DECISIONS, решение 6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Reranker:
    model_path: str
    device: str = "cpu"
    batch_size: int = 8
    max_length: int = 512
    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_path, device=self.device, max_length=self.max_length
            )
        return self._model

    def rerank(
        self, query: str, candidates: list[tuple[str, float]], texts: dict[str, str], top: int = 10
    ) -> tuple[list[tuple[str, float]], float]:
        """Переупорядочить кандидатов. Возвращает выдачу и затраченное время."""
        if not candidates:
            return [], 0.0
        model = self._load()
        t0 = time.monotonic()
        pairs = [(query, texts[c]) for c, _ in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = sorted(
            ((c, float(s)) for (c, _), s in zip(candidates, scores, strict=True)),
            key=lambda x: -x[1],
        )
        return order[:top], time.monotonic() - t0
