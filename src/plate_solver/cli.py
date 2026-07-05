"""cli.py — командная строка комплекса.

v0.2, P0.4: генератор шаблонов case-файлов::

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


def main(argv: list[str] | None = None) -> int:
    """Точка входа ``plate-solve`` (v0.2: только --new; решение случаев — P4)."""
    parser = argparse.ArgumentParser(
        prog="plate-solve",
        description="Комплекс plate-solver: постановка задачи case-файлом "
                    "(docs/CASE_SCHEMA.md).",
    )
    parser.add_argument("case", nargs="?", help="case-файл TOML (решение — начиная с P4)")
    parser.add_argument("--new", dest="new_kind", metavar="KIND",
                        help=f"создать шаблон case-файла: {' | '.join(_TEMPLATE_KINDS)}")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help="куда писать шаблон (по умолчанию <KIND>.toml)")
    args = parser.parse_args(argv)

    if args.new_kind is not None:
        try:
            path = write_template(args.new_kind, args.out)
        except CaseError as e:
            print(f"ошибка: {e}", file=sys.stderr)
            return 1
        print(f"шаблон записан: {path} (схема — docs/CASE_SCHEMA.md)")
        return 0

    if args.case is not None:
        print("решение case-файлов появится на шаге P4 фазы 2; "
              "пока доступно только plate-solve --new <KIND>", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
