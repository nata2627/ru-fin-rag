"""Чтение конфигурации поиска и переменных окружения.

Падаем громко на пустой обязательной переменной и на неизвестном имени связки:
тихая подстановка значения по умолчанию прячет опечатку в имени переменной на
хостинге, и прогон молча отвечает не тем, чем думает автор (SPEC, раздел 3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "retrieval.yaml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class RerankSpec:
    model: str
    depth: int
    batch_size: int
    max_length: int


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    first_stage: str
    candidates: int
    dense_model: str | None
    rrf_k: int
    rerank: RerankSpec | None
    note: str

    @property
    def version(self) -> str:
        """Версия связки для ответа API и для строки лога.

        Имя плюс состав: по нему видно, тем же ли путем получены две выдачи.
        """
        parts = [self.name, self.first_stage, f"cand{self.candidates}"]
        if self.dense_model:
            parts.append(self.dense_model)
        if self.rerank:
            parts.append(f"rerank{self.rerank.depth}")
        return "+".join(parts)


@dataclass(frozen=True)
class Catalog:
    version: int
    default: str
    configs: dict[str, RetrievalConfig]

    def get(self, name: str | None = None) -> RetrievalConfig:
        key = name or self.default
        if key not in self.configs:
            known = ", ".join(sorted(self.configs))
            raise ConfigError(f"нет связки {key!r}, известны: {known}")
        return self.configs[key]


def load_catalog(path: Path | None = None) -> Catalog:
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"нет файла конфигурации {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "configs" not in raw:
        raise ConfigError(f"{path}: нет раздела configs")

    configs: dict[str, RetrievalConfig] = {}
    for name, item in raw["configs"].items():
        rerank = item.get("rerank")
        configs[name] = RetrievalConfig(
            name=name,
            first_stage=_required(item, "first_stage", name),
            candidates=int(_required(item, "candidates", name)),
            dense_model=item.get("dense_model"),
            rrf_k=int(item.get("rrf_k", 60)),
            rerank=RerankSpec(
                model=_required(rerank, "model", f"{name}.rerank"),
                depth=int(rerank.get("depth", 50)),
                batch_size=int(rerank.get("batch_size", 8)),
                max_length=int(rerank.get("max_length", 512)),
            )
            if rerank
            else None,
            note=str(item.get("note", "")).strip(),
        )

    default = raw.get("default")
    if default not in configs:
        raise ConfigError(f"{path}: связка по умолчанию {default!r} не описана")
    return Catalog(version=int(raw.get("version", 1)), default=default, configs=configs)


def _required(item: dict, key: str, where: str):
    if not isinstance(item, dict) or item.get(key) in (None, ""):
        raise ConfigError(f"{where}: нет обязательного поля {key}")
    return item[key]


def env(name: str, required: bool = False) -> str:
    """Переменная окружения. Пустая обязательная переменная это ошибка, а не пустая строка."""
    value = os.environ.get(name, "").strip()
    if required and not value:
        raise ConfigError(f"не задана переменная окружения {name}")
    return value


def live_mode() -> bool:
    return env("RUFINRAG_LIVE") == "1"


def data_dir() -> Path:
    override = env("RUFINRAG_DATA_DIR")
    return Path(override).expanduser() if override else ROOT / "data"
