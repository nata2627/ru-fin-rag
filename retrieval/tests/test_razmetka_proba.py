"""Проба разметки: разбор ответов разметчика и слепота партий."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import razmetka_proba as R  # noqa: E402


def test_razbor_otvetov(tmp_path):
    path = tmp_path / "batch_01.answers.tsv"
    path.write_text("1\tда\tда\tчасть\n2\tнет\tда\tобщая\n3\tнет\tнет\tиное\nмусор\n", encoding="utf-8")
    out = R.parse(str(path))
    # «да» на вопрос 1 гасит второй ответ и пометку, «нет, нет» гасит пометку
    assert out == {"1": ("да", "-", "-"), "2": ("нет", "да", "общая"), "3": ("нет", "нет", "-")}


def test_otvet_vne_shkaly_padaet(tmp_path):
    path = tmp_path / "batch_01.answers.tsv"
    path.write_text("1\tскорее да\tнет\t-\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        R.parse(str(path))


def test_partii_slepye(tmp_path):
    """В партиях нет ни одного поля ответа человека."""
    if not (R.os.path.exists(R.HUMAN) and R.os.path.exists(R.POOL)):
        pytest.skip("контрольной разметки или пула нет")
    R.prepare(type("A", (), {"dir": str(tmp_path), "batch": 25})())
    rows = R.read_tsv(R.HUMAN)
    text = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("batch_*.prompt.txt"))
    assert text.count("=== Пара ") == len(rows)
    assert "otvet1_celikom" not in text and "otvet2_vmeste" not in text
    assert set(R.BLIND).isdisjoint({"otvet1_celikom", "otvet2_vmeste", "tip"})


def test_kappa_peresschityvaetsya_iz_listov():
    """Каппы файла результата пересчитываются из листов разметки без модели,
    и блок документа 18 совпадает с файлом."""
    import json

    result = json.loads(pathlib.Path(R.RESULT).read_text(encoding="utf-8"))
    model, again, person = (R.read_marks(p) for p in (R.OUT_SHEET, R.REPEAT_SHEET, R.HUMAN))
    assert R.agreement(model, person) == result["с человеком"]
    assert R.agreement(again, model) == result["повтор"]["с первым прогоном"]
    assert R.agreement(again, person) == result["повтор"]["с человеком"]
    assert result["с человеком"]["каппа"] >= R.THRESHOLD
    assert result["повтор"]["с первым прогоном"]["каппа"] >= R.REPEAT_THRESHOLD
    doc = pathlib.Path(R.DOC).read_text(encoding="utf-8")
    assert doc.split(R.BLOCK_START, 1)[1].split(R.BLOCK_END, 1)[0].strip() == R.doc_block(result)
