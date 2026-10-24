"""Генеративный режим: ответ пишет модель и только по найденным фрагментам.

Ключа к модели у проекта нет и может не появиться вовсе, поэтому режим выдержек
с цитатами это основной режим работы сервиса, а этот слой необязательный
(PLAN, этап 7). Без переменных окружения он просто недоступен, и сервис говорит
об этом прямо, вместо того чтобы падать.

Разговор идет по совместимому с OpenAI протоколу `/chat/completions`. Такой
протокол дает и vLLM на карте (DECISIONS, решение 8), и llama.cpp в режиме
сервера, и платный ключ: замер 7.3 и любой другой поставщик подключаются
переменной окружения без правки кода. Своего клиента здесь нет сверх `urllib`:
одна ручка в одном месте дешевле зависимости, которую пришлось бы тащить в
образ хостинга ради режима, который в онлайне выключен.

Модель отвечает одним объектом JSON: либо ответ и номера использованных
выдержек, либо отказ. Номера выдержек нужны затем, что ссылки собирает сервис
по своей выдаче, а не модель текстом: цитату, которую модель написала словами,
проверить нечем, а номер выдержки проверяется на месте.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .config import ConfigError, env

# Имя ключа уже описано в `.env.example`, и второго имени у одного ключа быть
# не должно.
KEY_VAR = "RUFINRAG_LLM_API_KEY"
BASE_URL_VAR = "RUFINRAG_LLM_BASE_URL"
MODEL_VAR = "RUFINRAG_LLM_MODEL"

SYSTEM = (
    "Ты отвечаешь на вопросы по нормативным актам Банка России. "
    "Отвечай только по приведенным выдержкам и ничего не добавляй от себя. "
    "Если выдержки на вопрос не отвечают, откажись. "
    "Номера актов, даты и номера пунктов бери только из выдержек. "
    "Ответ отдай одним объектом JSON без пояснений вокруг него: "
    '{"answer": "текст ответа", "used": [номера выдержек]} '
    'либо {"refusal": "чего не хватает в выдержках"}. '
    "Ответ не длиннее шести предложений."
)

# Ответ модели в тройных кавычках это частая манера чата: разметку снимаем, а
# не объявляем ответ неразбираемым.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


class GenerationError(Exception):
    pass


Completer = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    base_url: str
    model: str
    key: str
    max_tokens: int
    temperature: float
    timeout_seconds: float

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def settings_from_env(
    max_tokens: int, temperature: float, timeout_seconds: float
) -> tuple[GenerationSettings | None, str]:
    """Настройки модели из окружения и причина, по которой режим выключен.

    Ни одной переменной нет, значит режим выключен штатно. Заполнена часть,
    значит это опечатка на хостинге, и она обязана быть громкой: тихо
    выключенный режим выглядел бы так же, как отсутствующий ключ (SPEC, раздел 3).
    """
    values = {name: env(name) for name in (KEY_VAR, BASE_URL_VAR, MODEL_VAR)}
    if not any(values.values()):
        return None, f"нет переменных {KEY_VAR}, {BASE_URL_VAR}, {MODEL_VAR}"
    empty = sorted(name for name, value in values.items() if not value)
    if empty:
        raise ConfigError(
            f"генеративный режим настроен наполовину: не заданы {', '.join(empty)}. "
            f"Либо задайте все три переменные, либо снимите все: без них сервис "
            f"работает режимом выдержек"
        )
    return (
        GenerationSettings(
            base_url=values[BASE_URL_VAR],
            model=values[MODEL_VAR],
            key=values[KEY_VAR],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        ),
        "",
    )


def http_completer(settings: GenerationSettings) -> Completer:
    def complete(system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": settings.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
            }
        ).encode()
        request = urllib.request.Request(
            settings.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                body = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GenerationError(f"модель не ответила: {type(exc).__name__}: {exc}") from exc
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"в ответе модели нет текста: {str(body)[:200]}") from exc

    return complete


def build_prompt(query: str, excerpts: Sequence[tuple[str, str]]) -> str:
    """Запрос к модели: вопрос и пронумерованные выдержки со ссылками."""
    lines = [f"Вопрос: {query}", "", "Выдержки:"]
    for number, (reference, quote) in enumerate(excerpts, start=1):
        lines.append(f"[{number}] {reference}")
        lines.append(quote)
        lines.append("")
    return "\n".join(lines).strip()


def parse_reply(reply: str, excerpts: int) -> tuple[str, tuple[int, ...], str]:
    """Разбор ответа модели: текст, номера использованных выдержек, отказ.

    Разбор строгий. Модель, которая написала ответ мимо формата, считается
    отказавшейся: ответ, в котором неизвестно, на какие выдержки он опирается,
    опубликовать нельзя, а ссылки собирает сервис по номерам.
    """
    fenced = _FENCE.match(reply)
    text = fenced.group(1) if fenced else reply.strip()
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return "", (), f"модель вернула не JSON: {reply.strip()[:200]}"
    if not isinstance(body, dict):
        return "", (), f"модель вернула {type(body).__name__}, ожидался объект"
    refusal = body.get("refusal")
    if refusal:
        return "", (), f"модель отказалась: {str(refusal)[:300]}"
    answer = str(body.get("answer") or "").strip()
    if not answer:
        return "", (), "модель вернула пустой ответ"
    used = body.get("used")
    if not isinstance(used, list):
        return "", (), "модель не назвала номера использованных выдержек"
    numbers = tuple(
        sorted({int(item) for item in used if isinstance(item, int) and 1 <= item <= excerpts})
    )
    if not numbers:
        return "", (), f"модель назвала выдержки вне выдачи: {str(used)[:120]}"
    return answer, numbers, ""
