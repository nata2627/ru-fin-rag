"""Разметка пула кандидатов языковой моделью-судьей.

Пул живых вопросов на глубине 50 это 142 вопроса и сорок тысяч пар. Руками
столько не размечает никто, и разметка моделью здесь не отговорка, а принятая
практика. Цена ее называется тремя числами, а не одним (`scripts/kappa.py`):
каппа с человеком на выборке, согласие двух прогонов одной модели между собой
и записанный стек вывода. Третье число появилось после 23.10.2026, когда одна
и та же модель на тех же 150 парах дала каппу 0,307 вместо 0,408 и причиной
оказалась смена сборки движка, а не модель.

Четыре правила, без которых разметка судьей подыгрывает.

**Судья не видит, какая конфигурация нашла кандидата.** В листе этого нет
вовсе, и порядок перемешан. Иначе оценка подстроилась бы под знакомый метод.

**Инструкция судье и инструкция человеку это один текст.** Он лежит
константой `RUBRIC` в `rufin.annotation`, оттуда попадает и в подсказку, и в
`docs/ANNOTATION_GUIDE.md`. Разойдись они, каппа померила бы разницу
в инструкциях.

**Судья видит фрагмент целиком.** Подавать обрезанный текст значило бы
спрашивать его о другом, а число обрезанных пар делать известной
погрешностью разметки.

**Оценка не ставится, а собирается.** Судья отвечает на два отдельных
вопроса «да» или «нет» и ставит пометку типа, оценку 0/1/2 собирает
`annotation.scale_of` по правилу, напечатанному в самой инструкции. Все, что
разобрать не удалось, идет в отдельный счетчик, а не в оценку по умолчанию.

Температура нулевая и семя закреплено: разметка обязана воспроизводиться,
и это проверяется повторным прогоном, а не обещанием.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import platform
import re
import time

import gpu_gen as G

from rufin.annotation import ANSWER_FORMAT, RUBRIC, parse_answers, scale_of

DEFAULT_JUDGE = "Qwen/Qwen2.5-14B-Instruct-AWQ"
# Сколько неразобранных ответов сохранить дословно. Десяти хватает, чтобы
# увидеть, чем ответ модели отличается от формата, и не хватит, чтобы
# превратить метаданные в лог.
UNPARSED_SAMPLE = 10

# Второй судья, крупнее первого и тоже в четырех битах. Голосуют двое, потому
# что один судья неотличим от собственной погрешности: пара, в которой двое
# разошлись, в метрику не идет, и число таких пар печатается со знаменателем.
SECOND_JUDGE = "Qwen/Qwen2.5-32B-Instruct-AWQ"

SYSTEM = ("Ты оцениваешь, отвечает ли фрагмент нормативного акта Банка России "
          "на заданный вопрос. Следуй инструкции дословно. Отвечай тремя "
          "короткими строками и ничего не объясняй.")

PROMPT = """{rubric}
--- запрос ---
{query}

--- фрагмент ---
{passage}

{format}
"""

FIELDS = ["query_id", "chunk_id", "otvet_celikom", "otvet_vmeste", "tip",
          "otsenka_0_1_2", "syroy_otvet"]

_SPACE = re.compile(r"\s+")


def read_pool(path: str) -> list[dict]:
    """Лист кандидатов: пары «запрос, фрагмент».

    Текст вопроса в листе стоит только в первой строке группы, остальные
    пустые, так лист читается человеком. Здесь он раздается всей группе.
    """
    rows: list[dict] = []
    current = ""
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("vopros"):
                current = row["vopros"]
            rows.append({"query_id": row["query_id"], "chunk_id": row["chunk_id"],
                         "vopros": row.get("vopros") or current})
    return rows


def read_done(path: str) -> set[tuple[str, str]]:
    """Что уже размечено. Прогон на часы, и снятый снаружи процесс не должен
    уносить с собой сделанное: повторный запуск продолжает с места обрыва."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["query_id"], r["chunk_id"]) for r in csv.DictReader(f, delimiter="\t")
                if (r.get("otsenka_0_1_2") or "").strip() in ("0", "1", "2")}


def stack() -> dict:
    """Стек вывода: то, чем прогон отличается от другого прогона той же модели.

    Это третье число приемки судьи, и оно не число, а запись. 23.10.2026
    каппа той же модели на тех же парах упала с 0,408 до 0,307, и четыре
    отличия прогонов пришлось восстанавливать по памяти. Больше не придется.
    """
    out = {"питон": platform.python_version(), "платформа": platform.platform()}
    for name in ("vllm", "torch", "transformers"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "нет"
    try:
        import torch
        if torch.cuda.is_available():
            out["карты"] = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    for key in ("KAGGLE_DOCKER_IMAGE", "KAGGLE_KERNEL_RUN_TYPE",
                "CUDA_VISIBLE_DEVICES", "VLLM_WORKER_MULTIPROC_METHOD"):
        if os.environ.get(key):
            out[key] = os.environ[key]
    out["рубрика sha256"] = hashlib.sha256(RUBRIC.encode("utf-8")).hexdigest()[:16]
    return out


def judge(llm, pairs: list[dict], texts: dict[str, str], out_path: str,
          batch: int = 512, temperature: float = 0.0, seed: int = 20261024,
          max_tokens: int = 24, model: str = "", weights: str = "") -> dict:
    """Разметить пары. Возвращает статистику; ответы пишутся в TSV по ходу."""
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)

    done = read_done(out_path)
    pending = [p for p in pairs if (p["query_id"], p["chunk_id"]) not in done]
    print(f"пар в задании {len(pairs)}, размечено прежде {len(done)}, "
          f"сейчас размечаем {len(pending)}", flush=True)

    stats: collections.Counter = collections.Counter()
    unparsed: list[str] = []
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    first = not os.path.exists(out_path)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if first:
            w.writeheader()
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            part = [p for p in chunk if p["chunk_id"] in texts]
            stats["фрагмент не найден"] += len(chunk) - len(part)
            if not part:
                continue
            prompts = [G._chat(tok, PROMPT.format(
                rubric=RUBRIC, query=p["vopros"], passage=texts[p["chunk_id"]],
                format=ANSWER_FORMAT), system=SYSTEM) for p in part]
            outs = llm.generate(prompts, params, use_tqdm=False)
            for p, o in zip(part, outs):
                raw = o.outputs[0].text.strip()
                parsed = parse_answers(raw)
                if parsed is None:
                    stats["ответ не разобран"] += 1
                    # Образцы неразобранного кладутся в метаданные. Прогон 32B
                    # 24.10 дал 150 неразобранных из 150, и разобрать причину
                    # было нечем: в лист идут только разобранные ответы, а
                    # счётчик говорит сколько, но не что именно сказала модель.
                    if len(unparsed) < UNPARSED_SAMPLE:
                        unparsed.append(_SPACE.sub(" ", raw)[:200])
                    continue
                whole, together, kind = parsed
                score = scale_of(whole, together)
                stats[f"оценка {score}"] += 1
                if kind != "-":
                    stats[f"тип {kind}"] += 1
                w.writerow({"query_id": p["query_id"], "chunk_id": p["chunk_id"],
                            "otvet_celikom": "да" if whole else "нет",
                            "otvet_vmeste": "-" if whole else
                                            ("да" if together else "нет"),
                            "tip": kind, "otsenka_0_1_2": score,
                            "syroy_otvet": _SPACE.sub(" ", raw)[:60]})
            f.flush()
            spent = time.time() - t0
            seen = min(start + batch, len(pending))
            rate = seen / max(spent, 1e-6)
            print(f"   {seen}/{len(pending)}, {spent:.0f} с, {rate:.3f} пар/с, "
                  f"осталось {(len(pending) - seen) / max(rate, 1e-6) / 60:.0f} мин",
                  flush=True)

    spent = time.time() - t0
    graded = sum(v for k, v in stats.items() if k.startswith("оценка"))
    result = {"пар размечено": graded, "пар в задании": len(pairs),
              "размечено прежде": len(done), "распределение": dict(stats),
              "секунд": round(spent, 1),
              "пар в секунду": round(graded / spent, 3) if spent else 0.0,
              "прогноз на все задание, часов":
                  round(len(pairs) / (graded / spent) / 3600, 2)
                  if graded and spent else 0.0,
              "образцы неразобранного": unparsed,
              "модель": model, "веса": weights,
              "температура": temperature, "семя": seed,
              "стек вывода": stack()}

    meta = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nразмечено {graded} пар за {spent / 60:.1f} мин, "
          f"{result['пар в секунду']} пар/с; прогноз на задание "
          f"{result['прогноз на все задание, часов']} ч", flush=True)
    print(f"распределение: {dict(stats)}", flush=True)
    print(f"метаданные прогона: {meta}", flush=True)
    return result


def load_texts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {c["chunk_id"]: c["text"] for c in (json.loads(l) for l in f if l.strip())}
