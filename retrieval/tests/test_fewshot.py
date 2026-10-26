"""Примеры живых вопросов в подсказке генератора (E4).

Проверяется то, что ломается молча. Примеры обязаны попадать в подсказку
вопросных стилей и не попадать в короткий поисковый запрос: живой вопрос это
письмо в Банк России, и запрос из трёх слов по такому образцу не напишешь.
Набор примеров обязан меняться от фрагмента к фрагменту и повторяться при том
же семени: иначе вся выборка выйдет одного речевого режима, а прогон не
воспроизведётся.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))

# vLLM на маке не стоит и стоять не должен: восемь гигабайтов памяти и
# движок под CUDA. Сборку подсказки это не трогает, поэтому на место движка
# кладётся заглушка с тем же единственным классом, который берёт генерация.
if "vllm" not in sys.modules:
    заглушка = types.ModuleType("vllm")
    заглушка.SamplingParams = lambda **kw: kw
    sys.modules["vllm"] = заглушка

import gpu_gen as G  # noqa: E402

ПРИМЕРЫ = [f"живой вопрос номер {i} про обстоятельства дела" for i in range(20)]


class Шаблон:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class Выход:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text})()]


class Движок:
    """Вместо модели: запоминает подсказки и отвечает заготовкой."""

    def __init__(self):
        self.prompts = []

    def get_tokenizer(self):
        return Шаблон()

    def generate(self, prompts, params, use_tqdm=False):
        self.prompts = list(prompts)
        return [Выход("какой-то вопрос?") for _ in prompts]


def стиль(имя):
    return next(s for s in G.STYLES if s.name == имя)


def test_примеры_попадают_в_подсказку_вопросного_стиля():
    движок = Движок()
    G.generate(движок, стиль("бытовая ситуация"), ["фрагмент"], examples=ПРИМЕРЫ, shots=3)
    подсказка = движок.prompts[0]
    assert "как пишут вопросы те, кто приходит" in подсказка
    assert сколько_примеров(подсказка) == 3


def test_короткому_поисковому_запросу_примеры_не_показываются():
    движок = Движок()
    G.generate(движок, стиль("короткий поисковый запрос"), ["фрагмент"],
               examples=ПРИМЕРЫ, shots=3)
    assert "живой вопрос номер" not in движок.prompts[0]


def test_без_примеров_подсказка_прежняя():
    движок = Движок()
    G.generate(движок, стиль("вопрос профессионала"), ["фрагмент"])
    assert "живой вопрос номер" not in движок.prompts[0]


def test_набор_примеров_разный_у_разных_фрагментов():
    движок = Движок()
    G.generate(движок, стиль("вопрос профессионала"), ["а", "б", "в", "г"],
               examples=ПРИМЕРЫ, shots=3, seed=13)
    наборы = {tuple(номера(p)) for p in движок.prompts}
    assert len(наборы) > 1


def test_тот_же_номер_и_то_же_семя_дают_тот_же_набор():
    первый, второй = Движок(), Движок()
    for движок in (первый, второй):
        G.generate(движок, стиль("вопрос профессионала"), ["фрагмент"],
                   examples=ПРИМЕРЫ, shots=3, seed=13, offset=7)
    assert номера(первый.prompts[0]) == номера(второй.prompts[0])


def номера(подсказка: str) -> list[int]:
    return [int(часть.split()[0]) for часть in подсказка.split("живой вопрос номер ")[1:]]


def сколько_примеров(подсказка: str) -> int:
    return len(номера(подсказка))
