#!/usr/bin/env python3
"""Генератор docs/dispatch_flow.png — блок-схема диспетчера (ГОСТ 19.701-90).

Без новых зависимостей: только matplotlib. Обозначения: параллелограмм —
данные, ромб — решение, прямоугольник — процесс. Источник истины по
маршрутизации — докстринг plate_solver.dispatch и dispatch_flow.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402


def _box(ax, x, y, text, kind="proc", w=3.4, h=0.72):
    common = dict(ha="center", va="center", fontsize=8.2, wrap=True)
    if kind == "data":                      # параллелограмм
        dx = 0.25
        xs = [x - w / 2 + dx, x + w / 2 + dx, x + w / 2 - dx, x - w / 2 - dx]
        ys = [y + h / 2, y + h / 2, y - h / 2, y - h / 2]
        ax.fill(xs, ys, facecolor="#eef4fb", edgecolor="k", lw=1)
    elif kind == "dec":                     # ромб
        xs = [x, x + w / 2, x, x - w / 2]
        ys = [y + h / 2 + 0.12, y, y - h / 2 - 0.12, y]
        ax.fill(xs, ys, facecolor="#fdf3e3", edgecolor="k", lw=1)
    else:                                   # процесс
        ax.fill([x - w / 2, x + w / 2, x + w / 2, x - w / 2],
                [y + h / 2, y + h / 2, y - h / 2, y - h / 2],
                facecolor="white", edgecolor="k", lw=1)
    ax.text(x, y, text, **common)
    return (x, y)


def _arrow(ax, a, b, label=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 lw=1, color="k", shrinkA=22, shrinkB=22))
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx + 0.12, my, label, fontsize=7.6, ha="left", va="center",
                color="#444444")


def main() -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    a = _box(ax, 5, 14.3, "case-файл TOML", "data")
    b = _box(ax, 5, 13.2, "Problem.from_toml — валидатор\n(CaseError: получено/ожидалось)")
    c = _box(ax, 5, 12.1, "build_domain: реестр геометрий\ncircle|rectangle|L|annulus|compose")
    d = _box(ax, 5, 11.0, "bc.type ?", "dec", w=2.6)
    e = _box(ax, 2.6, 9.8, "PlateBending\n(расщепление, две Пуассоны)")
    f = _box(ax, 7.4, 9.8, "ClampedPlate\n(w = ω²Φ, прямой Ритц)")
    g = _box(ax, 5, 8.6, "нагрузка в узлах квадратуры:\nuniform | patch | point (≥ 20 узлов)")
    h = _box(ax, 5, 7.5, "contact.enabled ?", "dec", w=3.2)
    i = _box(ax, 2.6, 6.3, "theory=ktn: corrected_deflection\nпри r = 0 (Δw = −M/D)")
    ll = _box(ax, 7.4, 6.3, "Δ = gap | gap_factor·w_free;\nзона → foundation_mask")
    m = _box(ax, 7.4, 5.1, "ContactMOR:\nr ← [r + β_eff(u − Δ)]₊")
    n = _box(ax, 5, 3.9, "Result: w_max, cond(A), поля,\nконтакт, warnings, тайминги", "data")
    o = _box(ax, 5, 2.8, "verify_result: analytic | mms | fem\n| cross_1d | model_gap (инфо)")
    p = _box(ax, 5, 1.7, "result.json + фигуры + таблица", "data")

    _arrow(ax, a, b)
    _arrow(ax, b, c)
    _arrow(ax, c, d)
    _arrow(ax, d, e, "soft_hinge")
    _arrow(ax, d, f, "clamped")
    _arrow(ax, e, g)
    _arrow(ax, f, g)
    _arrow(ax, g, h)
    _arrow(ax, h, i, "нет")
    _arrow(ax, h, ll, "да")
    _arrow(ax, ll, m)
    _arrow(ax, i, n)
    _arrow(ax, m, n)
    _arrow(ax, n, o)
    _arrow(ax, o, p)
    ax.set_title("Диспетчер plate-solver v0.2 (ГОСТ 19.701-90)", fontsize=11)

    out = Path(__file__).with_name("dispatch_flow.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
