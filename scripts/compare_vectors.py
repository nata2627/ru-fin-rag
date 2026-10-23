#!/usr/bin/env python3
"""Насколько разошлись две матрицы векторов одной нарезки.

Зачем. Выдача плотного поиска по свежим векторам совпала с сохраненной
выдачей этапа B на 86,7% первой десятки, и надо было понять, где разница:
в этой реализации или в том, что этап B считался на Kaggle, а теперь на
другой карте и другой версии библиотек. Вопрос решается прямо: взять два
файла векторов одной и той же модели по одной и той же нарезке и посчитать
косинус построчно.

Матрицы сравниваются по именам фрагментов, а не по порядку строк: порядок
задается тем, кто считал, и совпадение порядка это как раз то, что надо
проверить, а не предположить.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(folder: str) -> tuple[list[str], np.ndarray]:
    ids = [l.rstrip("\n") for l in open(os.path.join(folder, "ids.txt"),
                                        encoding="utf-8")]
    vectors = np.load(os.path.join(folder, "vectors.npy"))
    if len(ids) != vectors.shape[0]:
        raise SystemExit(f"{folder}: имен {len(ids)}, векторов {vectors.shape[0]}")
    return ids, vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="папка с vectors.npy и ids.txt")
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", default=None, help="куда положить числа (json)")
    args = ap.parse_args()

    left_ids, left = load(args.left)
    right_ids, right = load(args.right)
    common = [c for c in left_ids if c in set(right_ids)]
    if not common:
        raise SystemExit("общих фрагментов нет: это разные нарезки")
    place_l = {c: i for i, c in enumerate(left_ids)}
    place_r = {c: i for i, c in enumerate(right_ids)}
    a = left[[place_l[c] for c in common]].astype("float32")
    b = right[[place_r[c] for c in common]].astype("float32")
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    report = {
        "слева": os.path.relpath(args.left, ROOT),
        "справа": os.path.relpath(args.right, ROOT),
        "фрагментов слева": len(left_ids),
        "фрагментов справа": len(right_ids),
        "общих фрагментов": len(common),
        "порядок строк совпадает": left_ids == right_ids,
        "косинус медиана": round(float(np.median(cos)), 5),
        "косинус минимум": round(float(cos.min()), 4),
        "ниже 0,999": int((cos < 0.999).sum()),
        "ниже 0,99": int((cos < 0.99).sum()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"файл: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
