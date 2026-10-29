"""Сплит корпуса по актам: чтение, проверки, назначение группы.

Сплит режется по актам, а не по фрагментам. Фрагменты одного акта нарезаны
с перекрытием и делят друг с другом текст, поэтому обучение на одних
фрагментах акта и проверка на других — это утечка: модель видела те же
предложения, просто с другой границей.

Группы и их назначение:

* `train` — фрагменты, по которым генерируются обучающие вопросы;
* `dev` — отложенные целиком акты, на них принимается или отклоняется
  каждый этап рецепта обучения;
* `test_zhivye` — акты, про которые спрашивают в «Разъяснениях»; здесь
  живут живые вопросы, главная цифра проекта;
* `test_zhanr` — отложенный целиком класс документов (Инструкции): проверка
  переноса не на новый акт, а на другой язык и другую структуру;
* `test_nevidannye` — акты, целиком не вошедшие в обучение: обобщение
  против запоминания.

Группы не пересекаются, и каждый акт корпуса состоит ровно в одной.
"""
from __future__ import annotations

import json

TRAIN = "train"
DEV = "dev"
TEST_LIVE = "test_zhivye"
TEST_KIND = "test_zhanr"
TEST_UNSEEN = "test_nevidannye"

GROUPS = (TRAIN, DEV, TEST_LIVE, TEST_KIND, TEST_UNSEEN)
HELD_OUT = (DEV, TEST_LIVE, TEST_KIND, TEST_UNSEEN)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_act(split: dict) -> dict[str, str]:
    """act_id -> название группы."""
    out: dict[str, str] = {}
    for group, acts in split["gruppy"].items():
        for act_id in acts:
            out[act_id] = group
    return out


def overlaps(split: dict) -> list[tuple[str, str, str]]:
    """Акты, попавшие больше чем в одну группу: (act_id, группа, группа)."""
    seen: dict[str, str] = {}
    bad: list[tuple[str, str, str]] = []
    for group, acts in split["gruppy"].items():
        for act_id in acts:
            if act_id in seen:
                bad.append((act_id, seen[act_id], group))
            else:
                seen[act_id] = group
    return bad


def act_of_chunk(chunk_id: str) -> str:
    """Идентификатор акта по идентификатору фрагмента: «акт#номер»."""
    return chunk_id.split("#", 1)[0]


# Поля обучающей записи, где лежит фрагмент-источник: вопрос по одной норме,
# вопрос по двум нормам (E5) и пара фрагментов для генерации (pairs_e5).
SOURCE_FIELDS = ("gold_chunk_id", "gold_chunk_id_2", "chunk_id", "chunk_id_2")


def sources_of(row: dict) -> set[str]:
    out = {row[f] for f in SOURCE_FIELDS if isinstance(row.get(f), str) and row[f]}
    return out | set(row.get("gold_chunk_ids") or ())


def fragment_leaks(rows: list[dict], test_fragments: set[str]) -> list[tuple[str, str]]:
    """Обучающие записи, чей фрагмент-источник лежит в эталоне теста.

    Проверка по актам этого не ловит: подвыборка «синтетика» собрана до
    сплита, и 106 из 150 ее запросов про обучающие акты. Поэтому с задачи
    11.2 утечка проверяется по фрагментам: (номер записи или query_id, фрагмент).
    """
    out = []
    for i, row in enumerate(rows):
        for cid in sorted(sources_of(row) & test_fragments):
            out.append((row.get("query_id", str(i)), cid))
    return out


def drop_leaked(rows: list[dict], test_fragments: set[str]) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for row in rows:
        (dropped if sources_of(row) & test_fragments else kept).append(row)
    return kept, dropped
