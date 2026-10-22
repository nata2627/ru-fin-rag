"""Разметка пула кандидатов языковой моделью-судьёй на видеокарте.

Задание готовит часть А (`scripts/pool_completeness.py`): пары «живой вопрос,
фрагмент» из объединённой первой десятки всех конфигураций. Инструкция судье
приезжает отдельным файлом из той же константы, по которой размечал человек,
поэтому согласие судьи с человеком измеряет согласие, а не разницу инструкций.

Прежний судья проекта `Qwen2.5-14B-Instruct-AWQ` в восемь гигабайт карты не
влезает, поэтому модель задаётся ключом, а её имя пишется в отчёт рядом с
числами: разметка двух разных моделей в одном эталоне несравнима.

Температура нулевая, ответ ожидается одной цифрой, и всё, что цифрой не
оказалось, идёт в отдельный счётчик, а не в оценку по умолчанию. Оценки
пишутся по ходу: прогон на несколько часов, и снятый снаружи процесс не
должен уносить с собой сделанное. Повторный запуск с тем же `--out`
продолжает с места обрыва.

Движков два. `transformers` ставится одной командой и годится на сотню пар.
На восьми тысячах он не годится: замерено 0,08 пары в секунду, потому что
четырёхбитная модель вместе с активациями не оставляет на карте места и
WSL начинает вытеснять видеопамять в обычную. `vllm` держит те же восемь
гигабайт под контролем и переиспользует общий префикс подсказки, а в нём
одна инструкция на 600 токенов.

Запуск: python scripts/judge_pool.py --task pool_live.tsv --rubric rubric.txt
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
import time
from pathlib import Path

SYSTEM = ("Ты оцениваешь релевантность фрагмента нормативного акта Банка России "
          "запросу. Отвечай одной цифрой: 0, 1 или 2. Ничего больше.")

PROMPT = """{rubric}
--- запрос ---
{query}

--- фрагмент ---
{passage}

Оценка (одна цифра: 0, 1 или 2):"""

DIGIT = re.compile(r"[012]")
THINK_END = "</think>"


def mark_of(answer: str, thinking: bool) -> str | None:
    """Оценка из ответа судьи.

    Без рассуждения ответ начинается цифрой, и берётся первая. С рассуждением
    цифры попадаются и в самом рассуждении («пункт 2.1»), поэтому читается
    только то, что сказано после его конца, и там берётся последняя: выводом
    судья заканчивает. Рассуждение, не уложившееся в отведённые токены,
    оценки не даёт, и выдумывать её из обрывка нельзя.
    """
    if not thinking:
        found = DIGIT.search(answer)
        return found.group() if found else None
    if THINK_END not in answer:
        return None
    found = DIGIT.findall(answer.split(THINK_END)[-1])
    return found[-1] if found else None


FIELDS = ["query_id", "chunk_id", "otsenka_0_1_2", "syroy_otvet"]


def read_task(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["query_id"], r["chunk_id"]) for r in csv.DictReader(f, delimiter="\t")
                if (r.get("otsenka_0_1_2") or "").strip() in ("0", "1", "2")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--engine", default="vllm", choices=("vllm", "transformers"))
    parser.add_argument("--limit", type=int, default=0, help="взять первые N пар")
    parser.add_argument("--prioritet", default="", help="только эти приоритеты, через запятую")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-prompt", type=int, default=2048, help="токенов на промпт")
    parser.add_argument("--thinking", action="store_true",
                        help="дать судье рассуждать перед оценкой: дороже, но на "
                             "четырёхмиллиардной модели согласие с человеком выше")
    parser.add_argument("--max-answer", type=int, default=4,
                        help="токенов на ответ; с рассуждением нужны сотни")
    parser.add_argument("--max-seqs", type=int, default=32,
                        help="сколько подсказок движок держит в работе разом")
    parser.add_argument("--gpu-util", type=float, default=0.85,
                        help="доля видеопамяти под движок; на карте 8 ГБ часть уже занята")
    args = parser.parse_args()

    rubric = args.rubric.read_text(encoding="utf-8")
    rows = read_task(args.task)
    if args.prioritet:
        keep = set(args.prioritet.split(","))
        rows = [r for r in rows if r.get("prioritet") in keep]
    done = read_done(args.out)
    pending = [r for r in rows if (r["query_id"], r["chunk_id"]) not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"задание: {len(rows)} пар, размечено прежде {len(done)}, "
          f"сейчас размечаем {len(pending)}", flush=True)
    if not pending:
        return 0

    t0 = time.time()
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams

        # enforce_eager: на карте 8 ГБ графы CUDA съедают гигабайт, которого
        # потом не хватает на кэш ключей, и движок отказывается стартовать.
        llm = LLM(model=args.model, dtype="float16",
                  max_model_len=args.max_prompt + args.max_answer + 8,
                  gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True,
                  max_num_seqs=args.max_seqs, enforce_eager=True)
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_answer)

        def answers_for(prompts: list[str]) -> list[str]:
            outs = llm.generate(prompts, sampling, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise SystemExit("карта не видна: torch.cuda.is_available() ложно")
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda",
                                                     dtype=torch.float16)
        model.eval()

        def answers_for(prompts: list[str]) -> list[str]:
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_prompt).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**enc, max_new_tokens=args.max_answer,
                                           do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            return tokenizer.batch_decode(generated[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)

    print(f"модель {args.model} поднята за {time.time() - t0:.0f} с "
          f"движком {args.engine}", flush=True)

    stats: collections.Counter = collections.Counter()

    def chat_of(passage: str, row: dict) -> str:
        text = PROMPT.format(rubric=rubric, query=row["vopros"], passage=passage)
        # enable_thinking важно для Qwen3: с рассуждением модель выдаст
        # <think> вместо цифры, а четырёх токенов на ответ ей не хватит.
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)

    def prompt_of(row: dict) -> str:
        """Подсказка, укороченная при нужде за счёт фрагмента.

        Резать инструкцию или вопрос нельзя: судья начнёт отвечать на другое.
        Длинные фрагменты режутся с хвоста, и число таких пар печатается:
        оценка по неполному тексту это известная погрешность, а не мелочь.
        """
        chat = chat_of(row["fragment"], row)
        length = len(tokenizer(chat).input_ids)
        if length <= args.max_prompt:
            return chat
        ids = tokenizer(row["fragment"], add_special_tokens=False).input_ids
        keep = max(0, len(ids) - (length - args.max_prompt))
        stats["фрагмент обрезан"] += 1
        return chat_of(tokenizer.decode(ids[:keep]), row)

    new = not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, delimiter="\t")
        if new:
            writer.writeheader()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            answers = answers_for([prompt_of(r) for r in batch])
            for row, answer in zip(batch, answers, strict=True):
                mark = mark_of(answer, args.thinking)
                if mark is None:
                    stats["ответ не цифра"] += 1
                    continue
                stats[f"оценка {mark}"] += 1
                writer.writerow({"query_id": row["query_id"], "chunk_id": row["chunk_id"],
                                 "otsenka_0_1_2": mark,
                                 "syroy_otvet": " ".join(answer.split())[-40:]})
            out.flush()
            spent = time.time() - t0
            seen = min(start + args.batch_size, len(pending))
            rate = seen / spent
            print(f"   {seen}/{len(pending)}, {spent:.0f} с, {rate:.1f} пар/с, "
                  f"осталось {(len(pending) - seen) / rate / 60:.0f} мин", flush=True)

    print(f"\nразмечено: {sum(v for k, v in stats.items() if k.startswith('оценка'))} пар "
          f"за {(time.time() - t0) / 60:.1f} мин")
    print("распределение: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    print(f"файл: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
