"""Число из файла, который переписывает прогон, обязано переписываться и в README.

29.10.2026 плановый прогон и прогон приемки покраснели одной ночью, оба на
сверке чисел, и оба по одной причине. В README стояли два числа из файлов,
которые прогон перезаписывает сам: время сборки индекса из `index.json` (на
раннере 14,0 с против 11,3 с на маке) и ссылка на прогон, порвавший цепочку
приемки, из `acceptance.json`. Красный плановый прогон рвет цепочку приемки
автономности, то есть цена такой мелочи это отсчет заново.

Правило, которое держит этот файл: если сверка берет число из файла, который
переписывает прогон, то README в том же прогоне переписывает генератор, и
генератор назван здесь поименно. Новая проверка на такой файл без генератора
роняет этот тест, а не плановый прогон через сутки.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess

import pytest

КОРЕНЬ = pathlib.Path(__file__).resolve().parents[1]


def _модуль(имя: str):
    spec = importlib.util.spec_from_file_location(имя, КОРЕНЬ / "scripts" / f"{имя}.py")
    м = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(м)
    return м


V = _модуль("verify_numbers")
L = _модуль("acceptance_line")

# Файлы results, которые переписывает прогон по расписанию или прогон приемки.
ПЕРЕПИСЫВАЕТ_ПРОГОН = {
    "index.json": "nightly.yml, шаг Rebuild index",
    "corpus.json": "nightly.yml, шаг Snapshot the corpus",
    "corpus-update.json": "nightly.yml, шаг Fetch new acts",
    "regression.json": "nightly.yml, шаг Regression",
    "regulation.json": "nightly.yml, шаг Record the heartbeat",
    "monitoring.json": "nightly.yml, шаг Monitor",
    "registry.json": "nightly.yml, шаг Rebuild the version registry",
    "events.json": "nightly.yml, шаг Drain the query log",
    "acceptance.json": "acceptance.yml, шаг Record the autonomy acceptance",
}

# Проверки, которым это разрешено, потому что их место в README переписывает
# генератор в том же прогоне до сверки.
ДЕРЖИТ_ГЕНЕРАТОР = {
    "актов в корпусе": "scripts/corpus_numbers.py --write",
    "размер корпуса": "scripts/corpus_numbers.py --write",
    "фрагментов в корпусе": "scripts/corpus_numbers.py --write",
    "словарь индекса": "scripts/corpus_numbers.py --write",
    "строка состояния приемки": "scripts/acceptance_line.py --write",
    "дата старта приемки": "неизменно: первый плановый прогон уже случился",
    "ссылка на первый плановый прогон": "неизменно: первый плановый прогон уже случился",
    # Реестр прогон пересобирает каждую ночь, но эти три числа описывают
    # устройство, а не замер: состав связок и артефактов и длина отпечатка
    # меняются только правкой конфигурации, и тогда README правится той же
    # правкой. Отпечатки и версии из того же файла в README не печатаются.
    "связок в реестре": "структура: меняется правкой configs/retrieval.yaml",
    "артефактов в реестре": "структура: меняется правкой состава артефактов",
    "длина версии выпуска": "константа устройства реестра",
}


def test_chislo_iz_perepisyvaemogo_fayla_derzhit_generator() -> None:
    беда = [
        f"{имя}: берется из {файл} ({ПЕРЕПИСЫВАЕТ_ПРОГОН[файл]}), "
        f"а в README его никто не переписывает"
        for имя, файл, _ in V.checks()
        if файл in ПЕРЕПИСЫВАЕТ_ПРОГОН and имя not in ДЕРЖИТ_ГЕНЕРАТОР
    ]
    assert беда == [], "\n".join(беда)


def test_generatory_zovutsya_do_sverki() -> None:
    """Генератор после сверки бесполезен: прогон упадет раньше, чем перепишет."""
    for прогон, генераторы in (
        ("nightly.yml", ["skills_table.py --write", "corpus_numbers.py --write"]),
        ("acceptance.yml", ["skills_table.py --write", "acceptance_line.py --write"]),
    ):
        текст = (КОРЕНЬ / ".github" / "workflows" / прогон).read_text(encoding="utf-8")
        сверка = текст.index("verify_numbers.py")
        for г in генераторы:
            assert г in текст, f"{прогон}: нет вызова {г}"
            assert текст.index(г) < сверка, f"{прогон}: {г} стоит после сверки чисел"


def test_stroka_sostoyaniya_v_readme_sovpadaet_s_zamerom() -> None:
    readme = (КОРЕНЬ / "README.md").read_text(encoding="utf-8")
    assert L.строка() in readme


def test_povtornaya_zapis_nichego_ne_menyaet() -> None:
    было = (КОРЕНЬ / "README.md").read_text(encoding="utf-8")
    вывод = subprocess.run(
        [".venv/bin/python", "scripts/acceptance_line.py", "--write"],
        cwd=КОРЕНЬ,
        capture_output=True,
        text=True,
        check=True,
    )
    стало = (КОРЕНЬ / "README.md").read_text(encoding="utf-8")
    assert стало == было, "запись при неизменившемся замере тронула README"
    assert "уже совпадает" in вывод.stdout


def test_bez_yakorya_skript_padaet(tmp_path: pathlib.Path) -> None:
    """Текст переписали руками, значит генератор больше не держит эти числа."""
    подмена = tmp_path / "README.md"
    подмена.write_text("README без строки состояния\n", encoding="utf-8")
    старый = L.README
    try:
        L.README = подмена
        with pytest.raises(SystemExit) as отказ:
            L.main_bez_argumentov()
    finally:
        L.README = старый
    assert "переписали руками" in str(отказ.value)


def test_data_zamera_lezhit_v_fayle() -> None:
    """Без даты замера строка состояния не знает, на какое она число."""
    d = json.loads((КОРЕНЬ / "results" / "acceptance.json").read_text(encoding="utf-8"))
    assert "checked_at" in d
    assert d["checked_at"].endswith("Z")
