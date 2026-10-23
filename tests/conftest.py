"""Общие условия прогона тестов.

Журнал запросов сервиса пишется в `data/events` рабочего дерева и оттуда едет
в репозиторий (DECISIONS, решение 12). Тестовый прогон туда писать не должен, и
один раз уже написал: сверка задачи 4.1 нашла 77 событий тестов в рабочем
дереве. Переменная ставится здесь на весь прогон и наследуется подпроцессами,
поэтому новый тест, который поднимет сервис, закрыт этим же правилом и помнить
о нем не надо.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def event_dir_outside_the_repo(tmp_path_factory):
    directory = tmp_path_factory.mktemp("events")
    previous = os.environ.get("RUFINRAG_EVENT_DIR")
    os.environ["RUFINRAG_EVENT_DIR"] = str(directory)
    yield directory
    if previous is None:
        os.environ.pop("RUFINRAG_EVENT_DIR", None)
    else:
        os.environ["RUFINRAG_EVENT_DIR"] = previous


@pytest.fixture(scope="session", autouse=True)
def repo_event_dir_stays_as_it_was():
    """Проверка на выходе, а не доверие переменной окружения.

    Переменную выше может снять любой тест, который ставит свою и потом
    убирает ее из окружения целиком: так и утекли события прошлого прогона.
    Эта проверка ловит утечку сама, в каком бы тесте она ни завелась.
    """
    directory = ROOT / "data/events"

    def state() -> dict[str, int]:
        return {p.name: p.stat().st_size for p in directory.glob("*.jsonl")}

    before = state()
    yield
    after = state()
    assert after == before, f"прогон дописал события в {directory}: было {before}, стало {after}"
