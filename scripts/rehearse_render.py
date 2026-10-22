"""Репетиция бесплатного тарифа: команды из `render.yaml` в контейнере на 512 МБ.

Зачем. Бесплатный план Render это 512 МБ памяти на сборку и на работу, и узнать,
что сервис туда не влезает, лучше здесь, чем на чужой машине через пять минут
сборки. Репетиция берет чистую выгрузку репозитория (`git archive`), то есть
ровно то, что Render клонирует, читает `buildCommand` и `startCommand` из
`render.yaml`, чтобы числа не разъехались с чертежом, и исполняет их в
`python:3.11-slim` с жестким потолком памяти.

Чего репетиция не мерит: сеть Render, их процессор и засыпание после простоя.
Это снимается на самом хостинге в задаче 2.4.

Запуск: python scripts/rehearse_render.py [--memory 512m] [--queries 20]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = "ru-fin-rag-rehearsal"
IMAGE = "python:3.11-slim"
PORT = 8099


def render_commands() -> tuple[str, str, dict[str, str]]:
    """Команды и переменные из `render.yaml` без зависимости от пакета yaml.

    Чертеж простой и плоский, поэтому разбор идет регулярками: лишняя
    зависимость ради четырех строк не нужна.
    """
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    build = re.search(r"^\s*buildCommand:\s*(.+)$", text, re.M).group(1).strip()
    start = re.search(r"^\s*startCommand:\s*(.+)$", text, re.M).group(1).strip()
    env = dict(re.findall(r"^\s*-\s*key:\s*(\S+)\s*\n\s*value:\s*\"?([^\"\n]+)\"?", text, re.M))
    return build, start, env


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def peak_memory_mb(samples: list[float]) -> float:
    return round(max(samples), 1) if samples else 0.0


def sample_memory(name: str) -> float | None:
    out = docker(
        "stats", "--no-stream", "--format", "{{.MemUsage}}", name, check=False
    ).stdout.strip()
    if not out:
        return None
    used = out.split("/")[0].strip()
    value = float(re.sub(r"[^0-9.]", "", used))
    if used.endswith("GiB"):
        value *= 1024
    elif used.endswith("KiB"):
        value /= 1024
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results/render-rehearsal.json")
    args = parser.parse_args()

    build_cmd, start_cmd, env = render_commands()
    work = Path(tempfile.mkdtemp(prefix="render-rehearsal-"))
    checkout = work / "repo"
    checkout.mkdir()
    subprocess.run(f"git -C {ROOT} archive HEAD | tar -x -C {checkout}", shell=True, check=True)

    docker("rm", "-f", CONTAINER, check=False)
    env_args: list[str] = []
    for key, value in env.items():
        env_args += ["-e", f"{key}={value}"]

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "memory_limit": args.memory,
        "image": IMAGE,
        "build_command": build_cmd,
        "start_command": start_cmd,
        "env": env,
        "note": "репетиция на маке, процессор и сеть Render сюда не входят",
    }
    try:
        docker(
            "run",
            "-d",
            "--name",
            CONTAINER,
            "--memory",
            args.memory,
            "--memory-swap",
            args.memory,
            "-v",
            f"{checkout}:/app",
            "-w",
            "/app",
            "-p",
            f"{PORT}:{PORT}",
            "-e",
            f"PORT={PORT}",
            *env_args,
            IMAGE,
            "sleep",
            "infinity",
        )

        t0 = time.monotonic()
        build = docker("exec", CONTAINER, "sh", "-lc", build_cmd, check=False)
        report["build_seconds"] = round(time.monotonic() - t0, 1)
        report["build_ok"] = build.returncode == 0
        if build.returncode != 0:
            report["build_tail"] = build.stdout[-2000:] + build.stderr[-2000:]
            return finish(report, args.out, work, failed=True)

        docker("exec", "-d", CONTAINER, "sh", "-lc", f"{start_cmd} > /tmp/serve.log 2>&1")
        samples: list[float] = []
        health = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 180:
            probe = docker(
                "exec",
                CONTAINER,
                "python",
                "-c",
                f"import urllib.request,json;print(urllib.request.urlopen('http://127.0.0.1:{PORT}/health').read().decode())",
                check=False,
            )
            if probe.returncode == 0:
                health = json.loads(probe.stdout)
                break
            value = sample_memory(CONTAINER)
            if value:
                samples.append(value)
            time.sleep(2)
        report["startup_seconds"] = round(time.monotonic() - t0, 1)
        report["health"] = health
        if health is None:
            report["serve_tail"] = docker(
                "exec", CONTAINER, "tail", "-40", "/tmp/serve.log", check=False
            ).stdout
            return finish(report, args.out, work, failed=True)

        queries = [
            row["text"]
            for row in map(
                json.loads,
                (ROOT / "data/benchmark/queries.jsonl").read_text(encoding="utf-8").splitlines(),
            )
        ][: args.queries]
        times = []
        for text in queries:
            probe = docker(
                "exec",
                CONTAINER,
                "python",
                "-c",
                "import json,sys,time,urllib.request\n"
                "body=json.dumps({'query':sys.argv[1],'top':10}).encode()\n"
                f"req=urllib.request.Request('http://127.0.0.1:{PORT}/search',data=body,"
                "headers={'Content-Type':'application/json'})\n"
                "t=time.monotonic();r=urllib.request.urlopen(req).read()\n"
                "print(json.dumps({'ms':(time.monotonic()-t)*1000,"
                "'n':len(json.loads(r).get('results',[]))}))",
                text,
                check=False,
            )
            if probe.returncode != 0:
                report["search_error"] = probe.stderr[-1000:]
                break
            row = json.loads(probe.stdout)
            times.append(row["ms"])
            value = sample_memory(CONTAINER)
            if value:
                samples.append(value)
        if times:
            ordered = sorted(times)
            p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
            report["search"] = {
                "queries": len(times),
                "p50_ms": round(statistics.median(times), 1),
                "p95_ms": round(p95, 1),
            }
        report["peak_memory_mb"] = peak_memory_mb(samples)
        report["fits_free_plan"] = bool(times) and report["peak_memory_mb"] < 512
        return finish(report, args.out, work, failed=not report["fits_free_plan"])
    finally:
        docker("rm", "-f", CONTAINER, check=False)


def finish(report: dict, out: Path, work: Path, failed: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
