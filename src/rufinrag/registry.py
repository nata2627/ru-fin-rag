"""Реестр версий: что именно отвечает на запросы и чем это проверяется.

Задача 6.3. Трекинг прогонов обучения и оценки идет в MLflow
(`scripts/track_runs.py`), а здесь лежит то, что нужно сервису и его проверке:
одна запись, которая называет версию каждого артефакта, и правило сравнения
записи с тем, что подняла машина.

Зачем запись отдельно от трекинга. MLflow в образ сервиса не ставится (это
66 МБ зависимостей на бесплатном тарифе с 512 МБ памяти), и сервис обязан знать
свою версию без него. Поэтому реестр это файл `results/registry.json`, который
собирает код из репозитория, а сервис его только читает и сравнивает.

Что считается расхождением. Версия корпуса, версия связки по умолчанию и версия
слоя ответа сравниваются дословно. Расхождение не роняет сервис: он продолжает
отвечать, но говорит об этом в `/health` и в ответе. Молча отвечать с индексом,
которого в реестре нет, нельзя: тогда числа в README не про то, что работает.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .answer import load_answer_settings
from .config import load_catalog

REGISTRY_FILE = "results/registry.json"
DIGEST_CHARS = 12
# Поля о самой сборке записи, а не о выпуске: дата сборки и машина не меняют
# того, что отвечает на запросы, и в отпечаток не идут. Иначе версия выпуска
# менялась бы от пересборки реестра на другой машине.
#
# Отпечатки артефактов в версию тоже не идут, и это не упрощение. Индекс,
# векторы и веса в git не едут (448,8 МБ весов и 45,9 МБ векторов), значит
# ночной прогон их не видит, а бесплатный тариф поднимается без плотной части
# вовсе. Считай версию по отпечаткам, и она менялась бы от машины к машине,
# то есть сервис не смог бы проверить свою. Индекс при этом детерминирован от
# корпуса, поэтому версию корпуса и числа индекса для выпуска достаточно, а
# отпечатки лежат рядом справкой и переносятся из прежней записи там, где
# файла на машине нет.
NOT_IN_VERSION = {"version", "built_at", "machine", "seconds", "artifacts"}
# Артефакты, версия которых не выводится из кода: их приходится считать с диска.
ARTIFACT_FILES = (
    "data/index/bm25.npz",
    "data/index/texts.blocks",
    "data/vectors/rufin-384.npy",
    "data/models/rufin/config.json",
    "data/models/rufin/model.safetensors",
)


@dataclass(frozen=True)
class Mismatch:
    field: str
    registered: str
    running: str


def file_digest(path: Path, blocks: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(blocks), b""):
            sha.update(block)
    return sha.hexdigest()[:DIGEST_CHARS]


def artifacts(root: Path, previous: dict | None = None) -> dict[str, dict[str, object] | None]:
    """Отпечаток и размер каждого артефакта, которого нет в коде.

    Файла может не быть: базовая установка идет без плотной части, а в git не
    едут ни индекс, ни векторы, ни веса. Тогда запись переносится из прежнего
    реестра с пометкой `from_previous`, и видно, что отпечаток снят не здесь.
    Так ночная пересборка не теряет отпечаток весов, который посчитать ей
    нечем. Прежней записи нет совсем, стоит `null`, то есть «не замерено», а
    не тихо пропущенная строка.
    """
    было = (previous or {}).get("artifacts") or {}
    out: dict[str, dict[str, object] | None] = {}
    for name in ARTIFACT_FILES:
        path = root / name
        if path.exists():
            out[name] = {
                "digest": file_digest(path),
                "mb": round(path.stat().st_size / 1048576, 1),
            }
        elif было.get(name):
            out[name] = {**было[name], "from_previous": True}
        else:
            out[name] = None
    return out


def build(
    root: Path, corpus_version: str | None = None, previous: dict | None = None
) -> dict[str, object]:
    """Запись реестра из того, что лежит в репозитории.

    `corpus_version` передается, если корпус уже поднят: считать его заново
    означает прочитать 62 606 фрагментов второй раз, а на старте сервиса это
    лишние секунды.
    """
    catalog = load_catalog()
    answer = load_answer_settings()
    index_report = root / "results/index.json"
    index = json.loads(index_report.read_text(encoding="utf-8")) if index_report.exists() else {}
    entry: dict[str, object] = {
        "service_version": __version__,
        "catalog_version": catalog.version,
        "default_config": catalog.default,
        "config_versions": {
            name: config.version for name, config in sorted(catalog.configs.items())
        },
        "answer_version": answer.version,
        "corpus_version": corpus_version,
        "index": {
            "chunks": index.get("chunks"),
            "vocab": index.get("vocab"),
            "measured_at": index.get("measured_at"),
        },
        "artifacts": artifacts(root, previous),
    }
    entry["version"] = version_of(entry)
    return entry


def version_of(entry: dict[str, object]) -> str:
    """Короткий отпечаток записи: он и есть версия выпуска.

    Считается по записи без полей о сборке, в устойчивом порядке ключей,
    поэтому две сборки на одних артефактах дают одно число, а любая смена
    связки, порога ответа или файла индекса дает другое.
    """
    payload = {key: value for key, value in entry.items() if key not in NOT_IN_VERSION}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def load(root: Path) -> dict[str, object] | None:
    path = root / REGISTRY_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare(registered: dict[str, object] | None, running: dict[str, str]) -> list[Mismatch]:
    """Чем поднятый сервис отличается от записи реестра.

    Сравниваются версии, а не отпечатки файлов: сервис на бесплатном тарифе
    поднимается без плотной части, и отпечаток ее весов там посчитать нечем.
    Версия корпуса, связки и слоя ответа при этом видны всегда.
    """
    if registered is None:
        return [Mismatch("registry", "нет файла реестра", "сервис поднят")]
    found: list[Mismatch] = []
    for field, value in running.items():
        expected = registered.get(field)
        if field == "config_version":
            versions = registered.get("config_versions") or {}
            expected = versions.get(registered.get("default_config"))
        if expected is None:
            found.append(Mismatch(field, "в реестре нет", str(value)))
        elif str(expected) != str(value):
            found.append(Mismatch(field, str(expected), str(value)))
    return found
