"""Сверка чисел README: число должно стоять в тексте целиком.

Проверка искала ожидаемую запись подстрокой, и «0,27» находилось внутри
«0,274», а «174» внутри «1174». То есть ошибка в опубликованном числе
выглядела бы как сошедшаяся проверка, а весь смысл скрипта в обратном.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_numbers", ROOT / "scripts" / "verify_numbers.py"
)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


@pytest.mark.parametrize(
    "ожидается,текст,есть",
    [
        ("0,27", "доля отказов 0,27 на 991 запросе", True),
        ("0,27", "доля отказов 0,274 на 991 запросе", False),
        ("174", "37 из 174 запросов", True),
        ("174", "37 из 1174 запросов", False),
        ("62 594", "корпус 62 594 фрагмента", True),
        ("62 594", "корпус 62 5940 фрагментов", False),
        ("37 из 174", "названо 37 из 174", True),
        ("0,690", "NDCG@10 0,690 у связки", True),
        ("0,69", "NDCG@10 0,690 у связки", False),
    ],
)
def test_chislo_stoit_celikom(ожидается, текст, есть):
    assert V.стоит_в_тексте(ожидается, текст) is есть


def test_vse_proverki_readme_shodyatsya():
    """Тот же прогон, что в check: 192 проверки и ни одного расхождения."""
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    проверок = V.checks()
    assert len(проверок) > 100
    расхождения = [
        (что, где, ждем) for что, где, ждем in проверок if not V.стоит_в_тексте(ждем, readme)
    ]
    assert расхождения == []


def test_kartochka_modeli_sveryaetsya(tmp_path, monkeypatch):
    # A stale number in the HuggingFace model card must fail the check the same
    # way a stale README does: the card leaves the repository on publication.
    card = (ROOT / "retrieval/docs/MODEL_CARD_EN.md").read_text(encoding="utf-8")
    assert "0.6791" in card
    (tmp_path / "card.md").write_text(card.replace("0.6791", "0.6800"), encoding="utf-8")
    monkeypatch.setattr(V, "ROOT", tmp_path)
    monkeypatch.setattr(V, "ТЕКСТЫ_ЭТАПА_12", {"card.md": ("en", ("dev",))})
    assert any("0.6791" in p for p in V.etap12_texts())
    (tmp_path / "card.md").write_text(card, encoding="utf-8")
    assert V.etap12_texts() == []
