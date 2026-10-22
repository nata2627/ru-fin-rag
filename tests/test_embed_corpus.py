"""Проверки прогона векторов, которые можно сделать без видеокарты и без модели.

Прогон идет на чужой машине, поэтому важны две вещи, которые ломаются молча:
что он продолжается с места обрыва и что порядок векторов совпадает с порядком
фрагментов в корпусе. Сам энкодер подменяется заглушкой: его поведение здесь
не проверяется, проверяется обвязка.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("embed_corpus", ROOT / "scripts/embed_corpus.py")
embed_corpus = importlib.util.module_from_spec(spec)
sys.modules["embed_corpus"] = embed_corpus
spec.loader.exec_module(embed_corpus)


class StubModel:
    """Отдает вектор, в котором записан номер фрагмента: по нему видно порядок."""

    calls: ClassVar[list[int]] = []

    def __init__(self, *args, **kwargs):
        self.max_seq_length = 0

    def encode(self, texts, **kwargs):
        StubModel.calls.append(len(texts))
        out = np.zeros((len(texts), embed_corpus.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, 0] = len(text)
        return out


@pytest.fixture
def stub_encoder(monkeypatch):
    StubModel.calls = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return StubModel


def run(out: Path, limit: int, argv_extra: list[str] | None = None) -> None:
    argv = ["embed_corpus.py", "--out", str(out), "--limit", str(limit), "--device", "cpu"]
    sys.argv = argv + (argv_extra or [])
    assert embed_corpus.main() == 0


def test_writes_vectors_and_ids(tmp_path, stub_encoder):
    out = tmp_path / "vectors.npy"
    run(out, 20)
    vectors = np.load(out)
    assert vectors.shape == (20, embed_corpus.DIM)
    assert vectors.dtype == np.float16
    ids = out.with_suffix(".ids.txt").read_text(encoding="utf-8").split()
    assert len(ids) == 20
    assert json.loads(embed_corpus.progress_path(out).read_text(encoding="utf-8"))["done"] == 20


def test_continues_after_a_break(tmp_path, stub_encoder):
    out = tmp_path / "vectors.npy"
    run(out, 20)
    first = np.load(out).copy()

    embed_corpus.write_progress(out, done=10, total=20, seconds=1.0)
    StubModel.calls = []
    run(out, 20)
    assert sum(StubModel.calls) == 10, "второй запуск посчитал больше, чем осталось"
    assert np.array_equal(np.load(out), first), "продолжение изменило уже посчитанное"


def test_progress_from_another_run_is_an_error(tmp_path, stub_encoder):
    out = tmp_path / "vectors.npy"
    run(out, 20)
    embed_corpus.write_progress(out, done=10, total=999, seconds=1.0)
    with pytest.raises(SystemExit, match="от другого прогона"):
        run(out, 20)
