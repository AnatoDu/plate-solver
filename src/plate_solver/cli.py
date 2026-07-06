"""cli.py — командная строка комплекса.

v0.2: генератор шаблонов case-файлов::

    plate-solve --new circle|rectangle|L|annulus [--out путь.toml]

Пишет закомментированный case-файл (обязательные секции заполнены,
необязательные показаны комментариями с дефолтами) — новый случай делается
копией шаблона и правкой нескольких строк (docs/CASE_SCHEMA.md).

Решение case-файлов (`plate-solve case.toml`), `plate-verify` и
`plate-ladder` появляются в P4 (после диспетчера P2 и эталонов P3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .problem import CaseError, Problem

_TEMPLATE_KINDS = ("circle", "rectangle", "L", "annulus")

_GEOMETRY = {
    "circle": '''[geometry]
kind = "circle"
a = 1.0                  # радиус''',
    "rectangle": '''[geometry]
kind = "rectangle"
x1 = 0.0                 # [x1, x2] × [y1, y2]
x2 = 1.0
y1 = 0.0
y2 = 1.0''',
    "L": '''[geometry]
kind = "L"
side = 1.0               # сторона квадрата
cut = 0.5                # квадратный вырез (0 < cut < side, входящий угол)''',
    "annulus": '''[geometry]
kind = "annulus"
a = 1.0                  # внешний радиус
b = 0.4                  # внутренний радиус (0 < b < a)''',
}

# Эталон по умолчанию — что доступно данной геометрии (см. CASE_SCHEMA#verify).
_VERIFY = {
    "circle": '''[verify]
reference = "analytic"   # analytic | mms | fem | none
cross_1d = true          # сверка с 1D-Ритцем по радиусу
tol = 1.0e-2
# model_gap = false      # строка «истинный Кирхгоф» (вне допуска)''',
    "annulus": '''[verify]
reference = "analytic"   # analytic | mms | fem | none
cross_1d = true          # сверка с 1D-Ритцем по радиусу [b, a]
tol = 1.0e-2
# model_gap = false''',
    "rectangle": '''[verify]
reference = "mms"        # mms | fem | none (analytic для прямоугольника нет)
tol = 1.0e-2''',
    "L": '''[verify]
reference = "fem"        # fem | mms | none (нужен pip install -e ".[fem]")
tol = 5.0e-2''',
}


def template(kind: str) -> str:
    """Текст закомментированного case-файла для геометрии ``kind``."""
    if kind not in _TEMPLATE_KINDS:
        raise CaseError(
            f"--new: получено {kind!r}, ожидалось {' | '.join(_TEMPLATE_KINDS)}, "
            "см. docs/CASE_SCHEMA.md#geometry"
        )
    return f'''# case-файл plate-solver (v0.2) — шаблон «{kind}».
# Схема и все ключи: docs/CASE_SCHEMA.md. Обязательны [geometry], [bc], [load];
# закомментированные ключи показывают дефолты (живут в plate_solver.config.Config).

{_GEOMETRY[kind]}

[bc]
type = "soft_hinge"      # soft_hinge (M=0, расщепление) | clamped (w=∂w/∂n=0)

[load]
type = "uniform"         # uniform | patch | point (см. CASE_SCHEMA.md#load)
q0 = 4.0                 # равномерная нагрузка (q0 > 0 «вниз»)
# точечная сила: type = "point", P = 1.0, x0 = 0.0, y0 = 0.0
#   (регуляризованный patch; eps по умолчанию 0.05·min(ширина, высота bbox))

[model]
theory = "classic"       # classic | ktn (поправки Кармана–Тимошенко–Нагди)
# E = 2.1e6              # дефолты Config — раскомментировать при необходимости
# nu = 0.3
# h = 1.0                # толщина (существенно для ktn)

# [contact]              # односторонний контакт (МОР); в v0.2 — только soft_hinge
# enabled = true
# gap_factor = 0.5       # Δ = gap_factor·w_free; либо абсолютный gap = 5.0e-5
# beta = 1.2             # 0 < β < 2 (теорема 4)
# max_iter = 8000
# [contact.zone]         # зона препятствия (дефолт: вся Ω); плоский штамп:
# kind = "rectangle"
# x1 = 0.15
# x2 = 0.45
# y1 = 0.15
# y2 = 0.45

[discretization]
p = 10                   # степень Чебышёва по оси (N = (p+1)²)
Q = 256                  # узлов квадратуры по оси (точность маски ~1/Q)
# grid_n = 80            # фоновая сетка вывода

{_VERIFY[kind]}

[output]
dir = "results/{kind}_case"
figures = false          # true — сохранить фигуры viz.py
'''


def write_template(kind: str, out: str | Path | None = None) -> Path:
    """Записать шаблон в файл (по умолчанию ``<kind>.toml``); перезапись запрещена."""
    path = Path(out) if out is not None else Path(f"{kind}.toml")
    if path.exists():
        raise CaseError(f"{path}: файл уже существует — перезапись запрещена, "
                        "укажите другой --out")
    text = template(kind)
    Problem.from_dict(_parse_for_selfcheck(text), source=str(path))  # самопроверка шаблона
    path.write_text(text, encoding="utf-8")
    return path


def _parse_for_selfcheck(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


# --------------------------------------------------------------------------- #
#  Свип по дискретизации: --sweep p=2:12:2 [--sweep Q=64:256:64]
# --------------------------------------------------------------------------- #
def _parse_sweep(spec: str) -> tuple[str, list[int]]:
    """Разобрать ``p=A:B:S`` / ``Q=A:B:S`` в (ключ, список значений)."""
    key, _, rng = spec.partition("=")
    key = key.strip()
    if key not in ("p", "Q"):
        raise CaseError(f"--sweep: получено {key!r}, ожидалось p | Q, "
                        "см. docs/CASE_SCHEMA.md#discretization")
    parts = rng.split(":")
    try:
        a, b, s = (int(v) for v in parts)
    except (ValueError, TypeError):
        raise CaseError(f"--sweep {spec!r}: ожидался формат КЛЮЧ=нач:кон:шаг "
                        "(целые числа)") from None
    if len(parts) != 3 or s <= 0 or b < a:
        raise CaseError(f"--sweep {spec!r}: ожидался формат КЛЮЧ=нач:кон:шаг, "
                        "шаг > 0, кон ≥ нач")
    return key, list(range(a, b + 1, s))


def _sweep_rows(problem: Problem, sweeps: list[tuple[str, list[int]]],
                do_verify: bool) -> list[dict]:
    """Прогнать декартово произведение точек свипа; собрать строки таблицы."""
    import dataclasses
    import itertools

    from .dispatch import solve as _solve

    keys = [k for k, _ in sweeps]
    rows: list[dict] = []
    for combo in itertools.product(*[vals for _, vals in sweeps]):
        disc = dataclasses.replace(problem.discretization, **dict(zip(keys, combo, strict=True)))
        prob = dataclasses.replace(problem, discretization=disc)
        res = _solve(prob)
        row: dict = dict(zip(keys, combo, strict=True))
        row["w_max"] = res.w_max
        row["cond_A"] = res.cond
        if do_verify:
            from .references import verify_result

            rep = verify_result(res)
            gated = [r for r in rep.rows if r.gated]
            row["rel"] = max((r.rel for r in gated), default=float("nan"))
            row["ok"] = rep.ok
        rows.append(row)
    return rows


def _write_sweep_outputs(rows: list[dict], keys: list[str], out_dir: Path,
                         do_verify: bool) -> None:
    """md + csv + png (semilogy rel против параметра) — публикационный формат."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    csv = [",".join(cols)]
    for r in rows:
        md.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
        csv.append(",".join(str(r[c]) for c in cols))
    (out_dir / "sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "sweep.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                            # png — опциональный артефакт
        return
    xs = [r[keys[0]] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    if do_verify:
        ax.semilogy(xs, [r["rel"] for r in rows], "o-")
        ax.set_ylabel("относительная ошибка (max по эталонам)")
    else:
        ax.plot(xs, [r["w_max"] for r in rows], "o-")
        ax.set_ylabel("w_max")
    ax.set_xlabel(keys[0])
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_dir / "sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def _run_case(args, do_verify: bool) -> int:
    """Общий путь plate-solve/plate-verify: решить case (или свип) и отчитаться."""
    from .dispatch import solve as _solve

    problem = Problem.from_toml(args.case)
    out_dir = Path(args.out) if args.out else Path(problem.output.dir)

    if args.sweep:
        sweeps = [_parse_sweep(s) for s in args.sweep]
        keys = [k for k, _ in sweeps]
        if len(set(keys)) != len(keys):
            raise CaseError("--sweep: ключи повторяются; допустимо по одному p и Q")
        if len(sweeps) > 1:
            print("предупреждение: два свипа — декартово произведение "
                  f"{'×'.join(str(len(v)) for _, v in sweeps)} прогонов", file=sys.stderr)
        rows = _sweep_rows(problem, sweeps, do_verify)
        _write_sweep_outputs(rows, keys, out_dir, do_verify)
        for line in (out_dir / "sweep.md").read_text(encoding="utf-8").splitlines():
            print(line)
        print(f"артефакты свипа: {out_dir}/sweep.{{md,csv,png}}")
        if do_verify:
            return 0 if rows[-1]["ok"] else 1      # вердикт — по самой точной точке
        return 0

    if getattr(args, "figures", False):
        import dataclasses

        problem = dataclasses.replace(
            problem, output=dataclasses.replace(problem.output, figures=True))
    res = _solve(problem)
    if do_verify:
        from .references import verify_result

        rep = verify_result(res)
        print(rep.table())
        print(f"допуск tol = {rep.tol:g}; вердикт: {'PASS' if rep.ok else 'FAIL'}")
        return 0 if rep.ok else 1
    formats = tuple(f.strip() for f in getattr(args, "fig_format", "png,pdf")
                    .split(",") if f.strip())
    path = res.save(out_dir, fig_formats=formats,
                    surface=getattr(args, "surface", "mid"))
    s = res.scalars()
    print(f"{args.case}: w_max = {res.w_max:.6e}, cond(A) = {res.cond:.2e}")
    if res.contact is not None:
        print(f"контакт: итераций {s['iters']}, узлов {s['n_contact']}/{s['n_quad']}, "
              f"r_max = {s['r_max']:.4e}, комплементарность {s['comp_residual']:.2e}")
    for w in res.warnings:
        print(f"предупреждение: {w}")
    if getattr(args, "report", False):
        rp = _write_report(args.case, res, out_dir)
        print(f"отчёт: {rp}")
    print(f"результат: {path}")
    return 0


def _write_report(case_path: str, res, out_dir: Path) -> Path:
    """Одностраничный md-отчёт по кейсу (F2.10): «нажал — получил документ».

    Состав: постановка листингом (исходный TOML), сводные числа
    (Result.scalars), verify-таблица при наличии эталона, фигуры
    относительными ссылками (если сохранены рядом).
    """
    lines = [f"# Отчёт: {Path(case_path).name}", ""]
    lines += ["## Постановка", "", "```toml",
              Path(case_path).read_text(encoding="utf-8").rstrip(), "```", ""]
    lines += ["## Сводные числа", "", "| величина | значение |", "|---|---|"]
    for k, v in res.scalars().items():
        if v is None:
            continue
        val = f"{v:.6e}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val} |")
    lines.append("")
    if res.problem.verify.reference != "none":
        from .references import verify_result

        rep = verify_result(res)
        verdict = "PASS" if rep.ok else "FAIL"
        lines += ["## Верификация", "", "```", rep.table(),
                  f"допуск tol = {rep.tol:g}; вердикт: {verdict}", "```", ""]
    figs = sorted(q.name for q in out_dir.glob("*.png"))
    if figs:
        lines += ["## Фигуры", ""]
        lines += [f"![{f}]({f})" for f in figs]
        lines.append("")
    for w in res.warnings:
        lines.append(f"> предупреждение: {w}")
    out = out_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
#  Точки входа
# --------------------------------------------------------------------------- #
_EXAMPLES = {
    "plate-solve": """примеры:
  plate-solve --new annulus            шаблон case-файла annulus.toml
  plate-solve case.toml                решить: result.json + fields.npz
  plate-solve case.toml --figures --report --out results/run1
  plate-solve case.toml --check        только валидация (exit 0/1)
  plate-solve case.toml --surface bottom --figures   прогиб нижней лицевой""",
    "plate-verify": """примеры:
  plate-verify case.toml               таблица эталонов, exit 0/1 по tol
  plate-verify case.toml --sweep p=2:12:2      сходимость по p (md+csv+png)
  plate-verify case.toml --sweep p=4:12:4 --sweep Q=64:256:64""",
}


def _base_parser(prog: str, descr: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description=descr, epilog=_EXAMPLES.get(prog),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", nargs="?", help="case-файл TOML (docs/CASE_SCHEMA.md)")
    parser.add_argument("--sweep", action="append", metavar="p=2:12:2",
                        help="свип по p или Q (можно оба — декартово произведение)")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help="каталог результатов (по умолчанию output.dir case-файла)")
    parser.add_argument("--figures", action="store_true",
                        help="форсировать output.figures = true (png 300 dpi + pdf)")
    parser.add_argument("--fig-format", metavar="png,pdf", default="png,pdf",
                        help="форматы фигур через запятую (по умолчанию png,pdf)")
    parser.add_argument("--surface", choices=("mid", "top", "bottom"),
                        default="mid",
                        help="поверхность на w-фигуре: срединная (mid) или "
                             "лицевые top/bottom (theory=ktn, NOTES §21)")
    from . import __version__

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """``plate-solve``: решить case-файл (``--new`` — сгенерировать шаблон)."""
    parser = _base_parser("plate-solve", "Решение постановки из case-файла "
                                         "(геометрия+КУ+нагрузка -> Result).")
    parser.add_argument("--new", dest="new_kind", metavar="KIND",
                        help=f"создать шаблон case-файла: {' | '.join(_TEMPLATE_KINDS)}")
    parser.add_argument("--check", action="store_true",
                        help="только валидация постановки (схема + статические "
                             "несовместимости), НИЧЕГО не считает; exit 0/1 — "
                             "для пользовательских CI")
    parser.add_argument("--report", action="store_true",
                        help="одностраничный md-отчёт по кейсу (постановка, "
                             "сводные числа, verify-таблица, фигуры) в каталог "
                             "результата")
    args = parser.parse_args(argv)
    try:
        if args.new_kind is not None:
            path = write_template(args.new_kind, args.out)
            print(f"шаблон записан: {path} (схема — docs/CASE_SCHEMA.md)")
            return 0
        if args.case is None:
            parser.print_help()
            return 0
        if args.check:
            Problem.from_toml(args.case)             # вся статика — валидатор
            print(f"{args.case}: постановка валидна (схема v0.3)")
            return 0
        return _run_case(args, do_verify=False)
    except CaseError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1


def main_verify(argv: list[str] | None = None) -> int:
    """``plate-verify``: таблица «эталон | значение | rel | статус», exit 0/1."""
    parser = _base_parser("plate-verify", "Верификация постановки по эталонам "
                                          "case-файла (секция [verify]).")
    args = parser.parse_args(argv)
    if args.case is None:
        parser.print_help()
        return 0
    try:
        return _run_case(args, do_verify=True)
    except CaseError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1


def main_ladder(argv: list[str] | None = None) -> int:
    """``plate-ladder``: каталог case-файлов → сводный md с провенансом."""
    parser = argparse.ArgumentParser(
        prog="plate-ladder",
        description="Прогнать каталог case-файлов (лестница верификации) и "
                    "собрать сводный markdown-отчёт.",
        epilog="пример:\n  plate-ladder cases/ci --out ladder.md",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="каталог с *.toml")
    parser.add_argument("--out", metavar="FILE", default=None,
                        help="файл отчёта (по умолчанию <каталог>/ladder_summary.md)")
    from . import __version__

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    folder = Path(args.directory)
    cases = sorted(folder.glob("*.toml"))
    if not cases:
        print(f"ошибка: в {folder} нет case-файлов *.toml", file=sys.stderr)
        return 1
    from .dispatch import _provenance
    from .dispatch import solve as _solve
    from .references import verify_result

    lines = ["# Лестница верификации — сводка", "",
             "| case | w_max | эталоны (rel) | статус |", "|---|---|---|---|"]
    all_ok = True
    for case in cases:
        try:
            problem = Problem.from_toml(case)
            res = _solve(problem)
            rep = verify_result(res)
            rels = "; ".join(f"{r.name}: {r.rel:.2e}" for r in rep.rows) or "—"
            ok = rep.ok
            lines.append(f"| {case.name} | {res.w_max:.6e} | {rels} | "
                         f"{'PASS' if ok else 'FAIL'} |")
        except CaseError as e:
            ok = False
            lines.append(f"| {case.name} | — | ошибка: {e} | FAIL |")
        all_ok &= ok
        print(f"{case.name}: {'PASS' if ok else 'FAIL'}")
    prov = _provenance()
    lines += ["", "## Провенанс",
              "", f"- plate-solver {prov['plate_solver']}, git {prov['git']}",
              f"- numpy {prov['numpy']}, scipy {prov['scipy']}, sympy {prov['sympy']}"]
    out = Path(args.out) if args.out else folder / "ladder_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"сводка: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
