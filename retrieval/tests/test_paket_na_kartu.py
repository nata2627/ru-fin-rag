"""Пакет для карты: ни один модуль пакета rufin не остается на маке.

Подготовка D0 упала 27.10 на `ModuleNotFoundError: rufin.training.batching`:
модуль появился 26.10, а список файлов пакета правили руками и его пропустили.
Пять минут квоты и перезапуск. Поэтому список сверяется с деревом `src/rufin`
кодом: новый модуль либо попадает в пакет, либо называется здесь лишним.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "make_kaggle_package", ROOT / "scripts" / "make_kaggle_package.py"
)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

# Модулям сбора и разбора источников на карте делать нечего: корпус уезжает
# туда готовым, а сеть к cbr.ru с Kaggle не трогается вовсе. Переранжирование
# на карте считает свой скрипт `kaggle/gpu_rerank.py`. Новый модуль попадает
# либо в пакет, либо сюда, с причиной.
НЕ_НУЖНЫ_НА_КАРТЕ: set[str] = {
    "src/rufin/sources/__init__.py",
    "src/rufin/sources/cbr_lawacts.py",
    "src/rufin/sources/cbr_vestnik.py",
    "src/rufin/http.py",
    "src/rufin/pdftext.py",
    "src/rufin/vestnik_split.py",
    "src/rufin/references.py",
    "src/rufin/parents.py",
    "src/rufin/explanations.py",
    "src/rufin/qrels_remap.py",
    "src/rufin/gate.py",
    "src/rufin/retrieval/rerank.py",
}


def модули_дерева() -> set[str]:
    корень = ROOT / "src" / "rufin"
    return {str(p.relative_to(ROOT)) for p in корень.rglob("*.py") if "__pycache__" not in p.parts}


def test_v_pakete_est_vse_moduli():
    в_пакете = {p for p in P.PACKAGE if p.startswith("src/rufin/")}
    пропущены = модули_дерева() - в_пакете - НЕ_НУЖНЫ_НА_КАРТЕ
    assert not пропущены, f"модули не уедут на карту: {sorted(пропущены)}"


def test_v_pakete_net_nesushchestvuyushchih():
    нет_на_диске = [p for p in P.PACKAGE if not (ROOT / p).exists()]
    assert not нет_на_диске, f"в списке пакета файлов нет на диске: {нет_на_диске}"
