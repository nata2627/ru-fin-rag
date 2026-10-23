"""Подъем мелкого фрагмента к родительскому куску.

Проверяется то, от чего зависит честность сравнения нарезок: родитель
выбирается по наибольшему пересечению (а базовые куски идут с перекрытием,
поэтому кандидатов больше одного), повторы в выдаче снимаются, и глубина
считается по родителям, а не по мелким фрагментам.
"""
from __future__ import annotations

from rufin.parents import ParentIndex


def index() -> ParentIndex:
    # базовая нарезка одного акта: три куска по 1000 символов с перекрытием
    return ParentIndex.from_records([
        {"act_id": "A", "chunk_id": "A#0", "char_start": 0, "char_end": 1000},
        {"act_id": "A", "chunk_id": "A#1", "char_start": 850, "char_end": 1850},
        {"act_id": "A", "chunk_id": "A#2", "char_start": 1700, "char_end": 2700},
        {"act_id": "B", "chunk_id": "B#0", "char_start": 0, "char_end": 500},
    ])


def test_родитель_по_наибольшему_пересечению():
    idx = index()
    assert idx.parent_of("A", 0, 400) == "A#0"
    assert idx.parent_of("A", 1200, 1600) == "A#1"
    assert idx.parent_of("A", 2000, 2400) == "A#2"


def test_в_зоне_перекрытия_берется_тот_где_куска_больше():
    idx = index()
    # 860..1000 лежит в обоих, но в A#0 попадает 140 символов, в A#1 — 150
    assert idx.parent_of("A", 860, 1010) == "A#1"
    assert idx.parent_of("A", 840, 990) == "A#0"


def test_при_ничьей_берется_тот_что_в_акте_раньше():
    idx = ParentIndex.from_records([
        {"act_id": "A", "chunk_id": "A#0", "char_start": 0, "char_end": 100},
        {"act_id": "A", "chunk_id": "A#1", "char_start": 100, "char_end": 200},
    ])
    assert idx.parent_of("A", 50, 150) == "A#0"


def test_кусок_вне_базовой_нарезки_уходит_к_ближайшему():
    # пробел в покрытии: позиция выдачи не должна исчезать, иначе ветвь
    # получит преимущество — у нее станет меньше промахов
    idx = ParentIndex.from_records([
        {"act_id": "A", "chunk_id": "A#0", "char_start": 0, "char_end": 100},
        {"act_id": "A", "chunk_id": "A#1", "char_start": 500, "char_end": 600},
    ])
    assert idx.parent_of("A", 200, 300) == "A#0"
    assert idx.parent_of("A", 400, 450) == "A#1"


def test_незнакомый_акт_родителя_не_имеет():
    assert index().parent_of("C", 0, 100) is None


def test_подъем_снимает_повторы_и_держит_порядок():
    idx = index()
    meta = {
        "f0": ("A", 1200, 1400),   # A#1
        "f1": ("A", 1400, 1600),   # A#1 снова
        "f2": ("A", 0, 200),       # A#0
        "f3": ("B", 0, 100),       # B#0
    }
    assert idx.lift(["f0", "f1", "f2", "f3"], meta, top=10) == ["A#1", "A#0", "B#0"]


def test_глубина_считается_по_родителям():
    idx = index()
    meta = {f"f{i}": ("A", 1200 + i, 1300 + i) for i in range(5)}
    meta["f9"] = ("A", 0, 100)
    # пять мелких фрагментов одного родителя — это одна позиция, и до второй
    # позиции список доходит только потому, что читается глубже
    assert idx.lift([f"f{i}" for i in range(5)] + ["f9"], meta, top=2) == ["A#1", "A#0"]
