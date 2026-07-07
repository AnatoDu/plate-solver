#!/usr/bin/env python3
"""reproduce_all.py — конвейер воспроизводимости (§9.1 ТЗ v0.6.0).

«Одна команда → все результаты из исходников». Реестр соответствия
``результат → case-файл → артефакт``: каждый case-файл — самостоятельный
воспроизводимый результат; скрипт прогоняет весь набор, собирает ключевые
числа и (по флагу) артефакты, пишет сводный отчёт соответствия. Актив под
вопрос «вы правда это посчитали?»: любую цифру можно перевычислить из
``cases/*.toml`` одним прогоном.

Состав:

* ``cases/ci/*.toml`` — быстрый набор (по умолчанию), каждый уже CI-ворота;
* ``cases/ladder/*.toml`` — боевые ступени (``--with-ladder``, тяжёлые; кэш).

Выход (в ``results/reproduce/`` — вне git, как все ``results/``):
``summary.md`` (таблица соответствия) + ``summary.csv`` (числа) + при
``--artifacts`` — поля/рисунки каждого случая. Кэш (``--cache``): случай с
неизменным (mtime) case-файлом не пересчитывается — экономит тяжёлую лестницу.

Запуск (из корня): ``.venv/bin/python scripts/reproduce_all.py [--with-ladder]
[--artifacts] [--cache] [--out DIR]``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plate_solver.dispatch import solve  # noqa: E402
from plate_solver.problem import CaseError, Problem  # noqa: E402

#: колонки сводки (реестр соответствия результат → case → артефакт → числа)
FIELDS = ("case", "group", "geometry", "theory", "bc", "contact",
          "w_max", "w_max_classic", "force_total", "level", "status", "artifact")


def discover_cases(with_ladder: bool = False) -> list[tuple[str, Path]]:
    """Собрать case-файлы: ci всегда, ladder — по флагу. Возвращает (группа, путь)."""
    out: list[tuple[str, Path]] = [("ci", p)
                                   for p in sorted((ROOT / "cases" / "ci").glob("*.toml"))]
    if with_ladder:
        out += [("ladder", p) for p in sorted((ROOT / "cases" / "ladder").glob("*.toml"))]
    return out


def _fmt(v) -> str:
    """Число → строка (единый формат) либо пусто для отсутствующих величин."""
    if v is None:
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def run_case(group: str, path: Path, *, artifacts: bool = False,
             out_dir: Path | None = None) -> dict:
    """Прогнать один case: постановка → решение → числа (+ артефакт). Ошибка → status=FAIL."""
    row = {k: "" for k in FIELDS}
    row["case"], row["group"] = path.stem, group
    try:
        problem = Problem.from_toml(path)
        row["geometry"] = problem.geometry.kind
        row["theory"] = getattr(problem.model, "theory", "")
        row["bc"] = getattr(problem.bc, "type", "")
        row["contact"] = "да" if getattr(problem, "contact", None) is not None else ""
        result = solve(problem)
        sc = result.scalars()
        row["w_max"] = _fmt(sc.get("w_max"))
        row["w_max_classic"] = _fmt(sc.get("w_max_classic"))
        row["force_total"] = _fmt(sc.get("force_total"))
        row["level"] = _fmt(sc.get("level"))
        if artifacts and out_dir is not None:
            art = result.save(out_dir / path.stem, fig_formats=("png",))
            row["artifact"] = str(art.relative_to(ROOT)) if art.is_relative_to(ROOT) else str(art)
        row["status"] = "OK"
    except (CaseError, ValueError, RuntimeError, NotImplementedError) as exc:
        row["status"] = f"FAIL: {type(exc).__name__}: {exc}"
    return row


def _load_cache(csv_path: Path) -> dict[str, dict]:
    """Прежние строки сводки по имени case (для пропуска неизменённых при --cache)."""
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def build_summary_md(rows: list[dict], *, elapsed: float, n_fail: int) -> str:
    """Markdown-таблица соответствия результат → case → числа → артефакт."""
    L = ["# Отчёт воспроизводимости plate-solver",
         "",
         f"Прогон `scripts/reproduce_all.py`: {len(rows)} случаев, "
         f"провалов {n_fail}, время {elapsed:.1f} с.",
         "Каждая строка — воспроизводимый результат из своего case-файла "
         "(реестр соответствия результат → case → артефакт).",
         "",
         "| Результат | Группа | Геометрия | Теория | ГУ | Контакт | w_max | "
         "w_classic | Σ силы | Уровень | Статус | Артефакт |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(r[k] or "—" for k in FIELDS) + " |")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Конвейер воспроизводимости (§9.1).")
    ap.add_argument("--with-ladder", action="store_true", help="включить cases/ladder (тяжёлые)")
    ap.add_argument("--artifacts", action="store_true", help="сохранять поля/рисунки случаев")
    ap.add_argument("--cache", action="store_true", help="пропускать неизменённые case-файлы")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "reproduce",
                    help="каталог отчёта (по умолч. results/reproduce)")
    args = ap.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "summary.csv", out_dir / "summary.md"
    cache = _load_cache(csv_path) if args.cache else {}

    cases = discover_cases(args.with_ladder)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for group, path in cases:
        prev = cache.get(path.stem)
        if prev and prev.get("_mtime") == str(path.stat().st_mtime) and prev.get("status") == "OK":
            rows.append({k: prev.get(k, "") for k in FIELDS})   # кэш: не пересчитываем
            print(f"[cache] {path.stem}")
            continue
        print(f"[run  ] {group}/{path.stem}")
        row = run_case(group, path, artifacts=args.artifacts, out_dir=out_dir)
        row["_mtime"] = str(path.stat().st_mtime)
        rows.append(row)
    elapsed = time.perf_counter() - t0

    n_fail = sum(1 for r in rows if not r["status"].startswith("OK"))
    md_path.write_text(build_summary_md(rows, elapsed=elapsed, n_fail=n_fail), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=(*FIELDS, "_mtime"))
        w.writeheader()
        w.writerows(rows)

    print(f"\nготово: {md_path} ({len(rows)} случаев, провалов {n_fail}, {elapsed:.1f} с)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
