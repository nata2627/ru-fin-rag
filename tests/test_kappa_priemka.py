"""Третье число приемки: стек вывода записан.

Проверка держится на одном краю. Температура судьи равна 0.0 по замыслу, и
пустым значением она не является, а `not meta.get(...)` считал ее пустой.
Из-за этого приемка объявляла стек незаписанным именно у детерминированного
прогона, то есть у правильного, и разметку нельзя было применить по причине,
которой нет.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
СТОЛБЦЫ = ["query_id", "chunk_id", "otsenka_0_1_2"]
# Три оценки и одно расхождение: каппа на одной категории не считается вовсе.
ОЦЕНКИ = [("q1", "c1", 2), ("q2", "c2", 1), ("q3", "c3", 0), ("q4", "c4", 2)]
ПОВТОР = [("q1", "c1", 2), ("q2", "c2", 1), ("q3", "c3", 0), ("q4", "c4", 1)]
МЕТА = {
    "модель": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "семя": 20261024,
    "температура": 0.0,
    "стек вывода": {"vllm": "0.32.0", "torch": "2.13.0+cu130"},
}


def лист(путь, строки):
    with open(путь, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=СТОЛБЦЫ, delimiter="\t")
        w.writeheader()
        for q, c, оценка in строки:
            w.writerow({"query_id": q, "chunk_id": c, "otsenka_0_1_2": оценка})


def приемка(tmp_path, мета):
    судья = tmp_path / "judge_live.tsv"
    повтор = tmp_path / "judge_live_repeat.tsv"
    лист(судья, ОЦЕНКИ)
    лист(повтор, ПОВТОР)
    (tmp_path / "judge_live_meta.json").write_text(
        json.dumps(мета, ensure_ascii=False), encoding="utf-8")
    готово = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "kappa.py"),
         "--judge", str(судья), "--repeat", str(повтор),
         "--out", str(tmp_path / "kappa.txt")],
        capture_output=True, text=True, cwd=ROOT)
    return готово.stdout


def test_temperatura_nol_eto_zapisannyy_stek(tmp_path):
    вывод = приемка(tmp_path, МЕТА)
    assert "в метаданных нет" not in вывод, вывод
    assert "стек вывода записан      1.000 при пороге 1.00: взято" in вывод, вывод


def test_propuschennaya_temperatura_eto_nezapisannyy_stek(tmp_path):
    """Обратная сторона: ослабить проверку до «ключ есть» было бы хуже молчания."""
    без = {k: v for k, v in МЕТА.items() if k != "температура"}
    вывод = приемка(tmp_path, без)
    assert "в метаданных нет: температура" in вывод, вывод
    assert "НЕ ВЗЯТО" in вывод, вывод


def test_pustaya_stroka_eto_tozhe_propusk(tmp_path):
    вывод = приемка(tmp_path, {**МЕТА, "семя": "  "})
    assert "в метаданных нет: семя" in вывод, вывод
