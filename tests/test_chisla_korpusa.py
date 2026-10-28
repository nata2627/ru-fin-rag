"""Корпусные числа README и ночной прогон.

Ночная докачка меняет корпус и словарь индекса. Если README за ними не
успевает, сверка чисел краснеет, а красный плановый прогон рвет цепочку
приемки автономности (так было 23.10). Тесты держат три вещи: замена трогает
только свои числа, после нее сверка сходится, а ночной прогон коммитит все
файлы, которые перезаписывают его шаги.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load("corpus_numbers")
V = _load("verify_numbers")

# Числа другой природы, которые стоят в README рядом и меняться не должны.
ЧУЖИЕ = ("62 594", "51 555", "85 399", "991")


@pytest.fixture
def площадка(tmp_path, monkeypatch):
    """Копии README, результатов и состояния: настоящие файлы не трогаются."""
    results = tmp_path / "results"
    results.mkdir()
    for name in ("corpus.json", "index.json", "corpus-published.json"):
        shutil.copy(ROOT / "results" / name, results / name)
    readme = tmp_path / "README.md"
    shutil.copy(ROOT / "README.md", readme)
    monkeypatch.setattr(C, "RESULTS", results)
    return readme, results


def _вырасти(results: Path) -> None:
    corpus = json.loads((results / "corpus.json").read_text(encoding="utf-8"))
    corpus.update(chunks=62_640, acts=1559, file_mib=23.1, measured_at="2026-10-29")
    (results / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    index = json.loads((results / "index.json").read_text(encoding="utf-8"))
    index.update(vocab=49_700, chunks=62_640)
    (results / "index.json").write_text(json.dumps(index), encoding="utf-8")


def test_zapis_bez_izmeneniy_nichego_ne_menyaet(площадка):
    """Ночь без докачки не должна давать правок README: иначе коммит каждую ночь."""
    readme, results = площадка
    before = readme.read_text(encoding="utf-8")
    state = (results / "corpus-published.json").read_text(encoding="utf-8")
    C.run(readme, results / "corpus-published.json", write=True)
    assert readme.read_text(encoding="utf-8") == before
    assert (results / "corpus-published.json").read_text(encoding="utf-8") == state


def test_vyrosshiy_korpus_perepisan_i_chuzhie_chisla_celi(площадка):
    """Замена берет только свои числа: 62 594, 51 555, 85 399 и 991 стоят рядом."""
    readme, results = площадка
    before = readme.read_text(encoding="utf-8")
    _вырасти(results)
    C.run(readme, results / "corpus-published.json", write=True)
    after = readme.read_text(encoding="utf-8")
    for токен in ЧУЖИЕ:
        assert len(re.findall(rf"(?<![\d]){re.escape(токен)}(?!\d)", after)) == len(
            re.findall(rf"(?<![\d]){re.escape(токен)}(?!\d)", before)
        ), токен
    assert "62 640 фрагмента" in after and "1559 акта" in after
    assert "словарь индекса 49 700" in after and "23,1 МиБ" in after
    assert "На 29.10.2026 в корпусе" in after
    assert "1555 акта" not in after and "62 606 фрагмента" not in after
    assert after.count("22,7 МиБ") == 0 and after.count("23,1 МиБ") == before.count("22,7 МиБ")
    # знаменатели чужих замеров остаются на составе корпуса, на котором сняты
    assert before.count("62 606 фрагментов") == after.count("62 606 фрагментов")


def test_granica_po_cifram_ne_cepliaet_1_555_vnutri_51_555(площадка):
    """«1 555» внутри «51 555» не токен: граница по цифрам, форма записи сохраняется."""
    readme, results = площадка
    old = json.loads((results / "corpus-published.json").read_text(encoding="utf-8"))
    old["acts"] = 1555
    text = "словарь 51 555 до стеммера, в корпусе 1 555 акта, еще 1555 акта.\n"
    new = dict(old, acts=1560)
    updated, counts = C.rewrite(
        text + "62 606 фрагмента, словарь индекса 49 643, 22,7 МиБ, На 23.10.2026 в корпусе.\n",
        old,
        new,
    )
    assert "51 555" in updated
    assert "1 560 акта" in updated and "1560 акта" in updated
    assert counts["acts"] == 2


def test_posle_zapisi_sverka_shoditsya(площадка, monkeypatch):
    """Фразы, которые ищет verify_numbers про корпус и индекс, находятся в новом README."""
    readme, results = площадка
    _вырасти(results)
    C.run(readme, results / "corpus-published.json", write=True)
    текст = " ".join(readme.read_text(encoding="utf-8").split())
    real_load = V.load
    monkeypatch.setattr(
        V,
        "load",
        lambda name: json.loads((results / name).read_text(encoding="utf-8"))
        if name in ("corpus.json", "index.json")
        else real_load(name),
    )
    проверки = [c for c in V.checks() if c[1] in ("corpus.json", "index.json")]
    assert len(проверки) >= 4
    for что, _, ожидается in проверки:
        assert V.стоит_в_тексте(ожидается, текст), что


def test_net_starogo_tokena_padaet(площадка):
    """Текст переписали руками: молча пропустить нельзя, число устареет при зеленой сверке."""
    readme, results = площадка
    text = readme.read_text(encoding="utf-8").replace("словарь индекса 49 643", "словарь 49 643")
    readme.write_text(text, encoding="utf-8")
    _вырасти(results)
    with pytest.raises(SystemExit) as err:
        C.run(readme, results / "corpus-published.json", write=True)
    assert "vocab" in str(err.value) and "переписали руками" in str(err.value)
    # упавшая замена не оставляет состояние, которое уже не про текст
    state = json.loads((results / "corpus-published.json").read_text(encoding="utf-8"))
    assert state["vocab"] == 49_643


def test_sostoyanie_sovpadaet_s_readme_repozitoriya():
    """Состояние в репозитории говорит правду о README: иначе ночной прогон упадет в первую ночь."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    state = json.loads((ROOT / "results/corpus-published.json").read_text(encoding="utf-8"))
    C.rewrite(text, state, state)


def _git_add(текст: str) -> set[str]:
    блок = re.search(r"git add ((?:.+\\\n)*.+)", текст).group(1)
    return set(блок.replace("\\\n", " ").split())


def test_git_add_nochnogo_progona_soderzhit_vse_perezapisannye_fajly():
    """Исходная мина: индекс и снимок корпуса перезаписывались, но в коммит не шли."""
    текст = (ROOT / ".github/workflows/nightly.yml").read_text(encoding="utf-8")
    в_коммите = _git_add(текст)
    нужно = set(C.WRITES) | {"results/corpus.json", "results/index.json"}
    assert нужно <= в_коммите, sorted(нужно - в_коммите)


def test_poryadok_shagov_pereschet_perezapis_sverka_kommit():
    """Сначала пересчет файлов, потом перезапись README, потом сверка, потом коммит."""
    текст = (ROOT / ".github/workflows/nightly.yml").read_text(encoding="utf-8")

    def позиция(фраза: str) -> int:
        место = текст.find(фраза)
        assert место >= 0, фраза
        return место

    assert позиция("scripts/build_index.py") < позиция("scripts/describe_corpus.py")
    assert позиция("scripts/describe_corpus.py") < позиция("scripts/corpus_numbers.py --write")
    assert позиция("scripts/skills_table.py --write") < позиция("scripts/verify_numbers.py")
    assert позиция("scripts/corpus_numbers.py --write") < позиция("scripts/verify_numbers.py")
    assert позиция("scripts/verify_numbers.py") < позиция("git add")
