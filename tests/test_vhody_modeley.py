"""Поиск весов среди входов Kaggle.

Эта функция стоит между «прогон считает» и «прогон встал на двенадцать
часов»: веса берутся только из подключённой модели, потому что хаб без
токена не падает, а молча замирает. Раскладку входов Kaggle задаёт сама,
и она не повторяет имя модели в хабе.

Проверяется ровно тот случай, на котором прогон судьи и остановился:
`Qwen/Qwen2.5-14B-Instruct-AWQ` приезжает как
`qwen2.5/transformers/14b-instruct-awq/1`, то есть части имени
разъезжаются по каталогам.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))

import gpu_common as common  # noqa: E402


def положить(корень, относительный: str) -> str:
    папка = корень / относительный
    папка.mkdir(parents=True, exist_ok=True)
    (папка / "config.json").write_text("{}", encoding="utf-8")
    return str(папка)


def test_имя_целиком_в_одном_каталоге(tmp_path):
    ждём = положить(tmp_path, "intfloat-multilingual-e5-small/transformers/default/1")
    assert common.resolve_model("intfloat/multilingual-e5-small",
                                root=str(tmp_path)) == ждём


def test_части_имени_разъехались_по_каталогам(tmp_path):
    ждём = положить(tmp_path, "qwen2.5/transformers/14b-instruct-awq/1")
    assert common.resolve_model("Qwen/Qwen2.5-14B-Instruct-AWQ",
                                root=str(tmp_path)) == ждём


def test_чужая_модель_не_подходит_по_частям(tmp_path):
    положить(tmp_path, "intfloat-multilingual-e5-small/transformers/default/1")
    with pytest.raises(SystemExit) as ошибка:
        common.resolve_model("Qwen/Qwen2.5-14B-Instruct-AWQ", root=str(tmp_path))
    # В сообщении обязан быть список того, что видно: без него «подключите
    # модель» одинаково звучит и когда модель не подключена, и когда она
    # подключена, а имя не сошлось.
    assert "multilingual-e5-small" in str(ошибка.value)


def test_необязательная_модель_возвращает_имя(tmp_path):
    assert common.resolve_model("Qwen/Qwen2.5-14B-Instruct-AWQ",
                                root=str(tmp_path),
                                обязательно=False) == "Qwen/Qwen2.5-14B-Instruct-AWQ"


def test_берётся_корень_модели_а_не_вложенная_папка(tmp_path):
    корень = положить(tmp_path, "qwen2.5/transformers/14b-instruct-awq/1")
    положить(tmp_path, "qwen2.5/transformers/14b-instruct-awq/1/onnx")
    assert common.resolve_model("Qwen/Qwen2.5-14B-Instruct-AWQ",
                                root=str(tmp_path)) == корень
