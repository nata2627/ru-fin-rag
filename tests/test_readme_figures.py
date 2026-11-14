"""Картинки README: числа на них взяты из замеров и не устаревают молча.

Картинку рисует человек командой `make figures`, ночной прогон ее не трогает.
Отсюда два правила. Манифест чисел обязан совпадать с пересчетом по файлам
результатов, иначе замер поменялся, а картинка осталась прежней. И ни одно
число не берется из файла, который переписывает прогон по расписанию: такая
сверка покраснела бы ночью, а перерисовать картинку там некому.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _module(ROOT / "scripts" / "readme_figures.py")
REWRITTEN = _module(ROOT / "tests" / "test_perepisyvaemye_chisla.py").ПЕРЕПИСЫВАЕТ_ПРОГОН


def test_manifest_matches_results():
    assert F.stale() == []


def test_every_figure_is_on_disk():
    manifest = json.loads(F.MANIFEST.read_text(encoding="utf-8"))
    assert manifest, "манифест пуст"
    for name in manifest:
        assert (F.OUT / f"{name}.png").exists(), f"нет картинки {name}.png"


def test_sources_are_frozen():
    used = F.sources(F.figures_data())
    assert used, "у чисел картинок нет источников"
    rewritten = {f"results/{name}" for name in REWRITTEN}
    both = used & rewritten
    assert not both, f"число картинки взято из файла, который переписывает прогон: {both}"


def test_sources_are_tracked():
    """Источник вне git в чистом клоне не найдется, и сверка упадет у читателя."""
    used = sorted(F.sources(F.figures_data()))
    tracked = subprocess.run(
        ["git", "ls-files", "--", *used], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    assert [p for p in used if p not in tracked] == []


def test_stale_names_a_changed_number(tmp_path, monkeypatch):
    manifest = json.loads(F.MANIFEST.read_text(encoding="utf-8"))
    manifest["memory"]["limit"]["value"] += 1
    changed = tmp_path / "figures.json"
    changed.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(F, "MANIFEST", changed)
    assert any("memory.limit.value" in line for line in F.stale())
