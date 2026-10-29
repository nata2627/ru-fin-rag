"""Счет приемки автономности обязан пересчитываться сам, прогоном `acceptance`.

Приемка считается по истории плановых прогонов, то есть каждый ночной прогон
меняет ее числа. Пока пересчета в прогонах не было, файл показывал 5 прогонов,
когда их прошло 11, а строка README про приемку брала числа из этого файла:
публичный текст отставал тем сильнее, чем дольше никто не запускал скрипт
руками.

Считает отдельный прогон, а не шаг внутри `nightly`, и это главное, что держат
проверки ниже. Шагом внутри регламента любой отказ пересчета (кончились права
на чтение Actions, API ответил 502) красит плановый прогон, то есть рвет
цепочку, длину которой измеряет. Измеритель внутри измеряемого так и работает.
"""

from __future__ import annotations

from pathlib import Path

КОРЕНЬ = Path(__file__).resolve().parents[1]
РЕГЛАМЕНТ = (КОРЕНЬ / ".github/workflows/nightly.yml").read_text(encoding="utf-8")
ПРИЕМКА = (КОРЕНЬ / ".github/workflows/acceptance.yml").read_text(encoding="utf-8")


def test_priemka_peresschityvaetsya_progonom() -> None:
    assert "scripts/record_acceptance.py" in ПРИЕМКА


def test_priemka_ne_vnutri_reglamenta() -> None:
    """Отказ пересчета не имеет права красить плановый прогон."""
    assert "record_acceptance" not in РЕГЛАМЕНТ


def test_povod_eto_zavershenie_reglamenta() -> None:
    """Коммит бота чужих прогонов не запускает, поводом идет `workflow_run`.

    К этому моменту плановый прогон уже завершен, значит попадает в счет сразу.
    Шагом внутри себя он учесть себя не мог бы: исхода у идущего прогона нет.
    """
    assert "workflow_run" in ПРИЕМКА
    assert "workflows: [nightly]" in ПРИЕМКА


def test_prava_na_chtenie_istorii_progonov() -> None:
    """Без `actions: read` команда `gh run list` отвечает 403.

    Явный блок permissions ставит все неназванное в none, в том числе чтение
    истории прогонов, по которой приемка и считается.
    """
    assert "actions: read" in ПРИЕМКА


def test_zapis_priemki_edet_v_kommit() -> None:
    """Пересчет без коммита бесполезен: файл перезапишется и пропадет."""
    добавление = ПРИЕМКА[
        ПРИЕМКА.index("git add results/acceptance.json") : ПРИЕМКА.index("if git diff --cached")
    ]
    assert "results/acceptance.json" in добавление
    assert "README.md" in добавление


def test_proverka_chisel_idet_posle_perezapisey() -> None:
    """Проверка обязана читать тот README, который уйдет в коммит.

    В регламенте она стояла до перезаписи таблицы навыков, то есть проверяла
    состояние, которого в истории не будет, а расхождение уезжало в коммит
    незамеченным. В обоих прогонах порядок теперь один.
    """
    for прогон, первый in (
        (ПРИЕМКА, "scripts/record_acceptance.py"),
        (РЕГЛАМЕНТ, "scripts/record_heartbeat.py"),
    ):
        таблица = прогон.index("scripts/skills_table.py --write")
        проверка = прогон.index("scripts/verify_numbers.py")
        коммит = прогон.index("git add ")
        assert прогон.index(первый) < таблица < проверка < коммит


def test_tekushchii_schet_v_readme_ne_dublirutsya() -> None:
    """Число приемки стоит в README один раз, строкой таблицы навыков.

    Таблицу вписывает `skills_table.py --write`. Если тот же счет продублировать
    абзацем руками, расходиться они начнут в первую же ночь.
    """
    import json
    import re

    readme = (КОРЕНЬ / "README.md").read_text(encoding="utf-8")
    замер = json.loads((КОРЕНЬ / "results/acceptance.json").read_text(encoding="utf-8"))

    строки = [
        s
        for s in readme.splitlines()
        if s.startswith("| ") and "Приемка автономности по факту" in s
    ]
    assert len(строки) == 1
    assert f"прогонов {замер['scheduled_runs_total']}" in строки[0]

    # Счет цепочки стоит ровно в одном месте помимо таблицы навыков: в строке
    # состояния, которую 29.10.2026 стал переписывать прогон приемки
    # (`scripts/acceptance_line.py`). Руками это число не ставится нигде, иначе
    # оно разойдется с файлом в первую же ночь, как и случилось 29.10.
    строка_состояния = [s for s in readme.splitlines() if s.startswith("**Состояние приемки")]
    assert len(строка_состояния) == 1
    прочее = [
        s
        for s in readme.splitlines()
        if not s.startswith("**Состояние приемки")
        and (re.search(r"\bв цепочке\s+\d", s) or re.search(r"\d[,.]\d\s+дня из", s))
    ]
    assert прочее == []
