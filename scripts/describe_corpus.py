"""Снимок корпуса: сколько фрагментов и актов в нем сейчас.

Зачем отдельный файл, а не чтение корпуса в проверке чисел. Числа «1554 акта» и
«62 594 фрагмента» стоят в README, а корпус растет: ночной прогон докачивает
новые акты. Если проверка чисел будет сверять README с живым корпусом, то
первое же пополнение уронит ночной прогон на его собственной проверке, и он
откроет issue про поломку там, где он сделал свою работу.

Поэтому снимок обновляется руками, вместе с правкой текста, и README говорит
«на дату». Ночной прогон этот скрипт не запускает.

Запуск: python scripts/describe_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import load_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/corpus.json")
    args = parser.parse_args()

    corpus = load_corpus()
    acts = {chunk.act.act_id for chunk in corpus.chunks}
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "chunks": len(corpus),
        "acts": len(acts),
        "version": corpus.version,
        # МиБ, а не МБ: делитель 1048576. Один и тот же файл успел побывать в
        # README и как 23,8 МБ, и как 22,7 МиБ, поэтому единица в имени поля.
        "file_mib": round(corpus.source.stat().st_size / 1048576, 1),
        "file_bytes": corpus.source.stat().st_size,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
