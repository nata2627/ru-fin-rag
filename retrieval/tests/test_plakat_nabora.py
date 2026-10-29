"""Плакат набора и таблица в документе держат только числа из снимка.

Снимок `nabor-audit-do.json` пишет `scripts/nabor_audit.py`, плакат и таблицу
рисует `scripts/nabor_plakat.py`. Здесь три сверки: плакат на диске совпадает
с тем, что скрипт рисует из снимка; каждое число плаката и таблицы есть в
снимке; снимок заново считается из данных тем же скриптом.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nabor_plakat as P  # noqa: E402

SNAPSHOT = ROOT / "data" / "results" / "nabor-audit-do.json"
POSTER = ROOT / "docs" / "img" / "nabor-do.svg"
DOC = ROOT / "docs" / "18-nabor-do-i-posle.md"

# Число, не приклеенное к букве или дефису: «590-П», «95-я», «M1» и «S28» это
# имена, а не замеры.
NUMBER = re.compile(r"(?<![\w.,-])\d{1,3}(?: \d{3})*(?:,\d+)?%?(?![\w-])")


def allowed(snap: dict) -> set[str]:
    out: set[str] = set()

    def walk(x) -> None:
        if isinstance(x, dict):
            if "доля" in x and x["доля"] is not None and "числитель" in x:
                out.add(P.pct(x))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.update({P.num(x), P.num(x, 1), P.num(x, 2)})

    walk(snap)
    return out


def strangers(text: str, snap: dict) -> list[str]:
    ok = allowed(snap)
    return [n for n in NUMBER.findall(text) if n not in ok]


def poster_values(svg: str) -> str:
    return "\n".join(re.findall(r'class="[^"]*\bv\b[^"]*"[^>]*>([^<]*)<', svg))


def doc_table(doc: str) -> str:
    return doc.split(P.TABLE_START, 1)[1].split(P.TABLE_END, 1)[0]


@pytest.fixture(scope="module")
def snap() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_plakat_narisovan_iz_snimka(snap):
    assert POSTER.read_text(encoding="utf-8") == P.svg(snap, None)


def test_sem_blokov_i_stalo_ne_izmereno(snap):
    svg = POSTER.read_text(encoding="utf-8")
    for code in ("M1", "M2", "M3", "M4", "M5", "M6", "M7"):
        assert f">{code}. " in svg
    assert svg.count(P.NOT_MEASURED) == 7


def test_kazhdoe_chislo_plakata_est_v_snimke(snap):
    values = poster_values(POSTER.read_text(encoding="utf-8"))
    assert len(NUMBER.findall(values)) >= 30
    assert strangers(values, snap) == []


def test_tablica_dokumenta_iz_snimka(snap):
    table = doc_table(DOC.read_text(encoding="utf-8"))
    assert table.strip() == P.table(snap, None)
    assert strangers(table, snap) == []


def test_podmena_chisla_lovitsya(snap):
    values = poster_values(POSTER.read_text(encoding="utf-8"))
    assert "62 594" in values
    assert strangers(values.replace("62 594", "62 606"), snap) == ["62 606"]


def test_dolya_ryadom_so_znamenatelem(snap):
    for b in P.blocks(snap):
        if b["доля"]["знаменатель"] is not None:
            assert " из " in P.headline(b)


@pytest.mark.data
@pytest.mark.parametrize("snapshot", sorted((ROOT / "data" / "results").glob("nabor-audit-*.json")),
                         ids=lambda p: p.stem)
def test_snimok_sozdan_skriptom(snapshot, tmp_path):
    if not (ROOT / "data" / "chunks" / "base.jsonl").exists():
        pytest.skip("нарезки нет: `make chunks`")
    snap = json.loads(snapshot.read_text(encoding="utf-8"))
    out = tmp_path / "audit.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "nabor_audit.py"), "--tag", snap["тег"], "--out", str(out)]
    for key, value in snap["аргументы"].items():
        cmd += [key, str(ROOT / value) if isinstance(value, str) else str(value)]
    subprocess.run(cmd, check=True, capture_output=True)
    assert json.loads(out.read_text(encoding="utf-8")) == snap
