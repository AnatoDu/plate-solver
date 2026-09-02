#!/usr/bin/env python3
"""Генератор docs/dispatch_flow.png — блок-схема диспетчера (ГОСТ 19.701-90).

Без новых зависимостей: только matplotlib. Обозначения: параллелограмм —
данные, ромб — решение, прямоугольник — процесс.

Источник истины по маршрутизации — исходный текст
``src/plate_solver/dispatch.py`` (``solve``, ``_solve_routed``,
``_solve_bending``, ``_solve_contact``); текстовая (mermaid) версия схемы —
docs/dispatch_flow.md, и она подробнее: растр сознательно огрубляет ветви
теорий и целей контакта до перечислений внутри блока, иначе схема
перестаёт читаться на листе.

Закоммиченный dispatch_flow.png отвечает РАННЕЙ маршрутизации (v0.2) и не
перерисовывается автоматически: файл двоичный. Чтобы обновить растр,
запустите этот скрипт по месту (``python docs/make_dispatch_flow.py``).
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
    fig, ax = plt.subplots(figsize=(8.6, 12.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 17)
    ax.axis("off")

    a = _box(ax, 5, 16.2, "case-файл TOML", "data")
    b = _box(ax, 5, 15.1, "Problem.from_toml — валидатор схемы\n"
                          "(CaseError: получено/ожидалось)", w=5.2)
    c = _box(ax, 5, 14.0, "build_domain — реестр геометрий:\n"
                          "circle|rectangle|ellipse|L|annulus|compose", w=5.6)
    d = _box(ax, 5, 12.9, "ограды постановки: M ≥ N (узлы/базис),\n"
                          "опоры внутри Ω, h(x, y) > 0", w=5.6)
    e = _box(ax, 5, 11.8, "секция [eigen]?", "dec", w=2.8)
    # левая колонка — ветвь собственных задач: она минует нагрузку, контакт
    # и опоры и сразу собирает Result (ранний возврат в _solve_routed)
    f = _box(ax, 1.9, 4.1, "_solve_eigen:\nbuckling | vibration\n"
                           "(+ преднапряжение N(w))", w=3.2, h=0.9)
    g = _box(ax, 7.2, 10.7, "theory, затем bc.type", "dec", w=3.4)
    hh = _box(ax, 7.2, 9.6, "решатель: KTNSolver (контакт+КТН) |\n"
                            "KTNPlate | KarmanPlate | ClampedPlate |\n"
                            "MixedRectPlate | PlateBending", w=5.6, h=0.9)
    i = _box(ax, 7.2, 8.5, "нагрузка в узлах: uniform | patch |\n"
                           "point (≥ 20 узлов) | gaussian | expr;\n"
                           "line и точная δ — вектором b", w=5.6, h=0.9)
    j = _box(ax, 7.2, 7.4, "contact.enabled?", "dec", w=3.2)
    k = _box(ax, 5.0, 6.3, "изгиб по теории: classic |\nktn_linear | karman | ktn_full",
             w=3.6, h=0.82)
    ll = _box(ax, 9.0, 6.3, "контакт: ContactMOR |\nNonlinearContactMOR\n"
                            "(karman | ktn_full)", w=3.6, h=0.9)
    m = _box(ax, 9.0, 5.2, "цель: основание |\nforce = P (∫r = P) | plate2", w=3.6,
             h=0.82)
    n = _box(ax, 7.2, 4.1, "реакции опор R_j = k·w(P_j)", w=5.0)
    o = _box(ax, 6.0, 3.0, "Result: w_max, cond(A), поля, контакт,\n"
                           "warnings (в т.ч. FactorizationWarning), тайминги",
             "data", w=6.0, h=0.82)
    p = _box(ax, 6.0, 1.9, "verify_result: analytic | mms | fem | cross_1d |\n"
                           "model_gap + ворота инвариантов контакта", w=6.0, h=0.82)
    q = _box(ax, 6.0, 0.8, "result.json + fields.npz + фигуры + таблица", "data",
             w=6.0)

    _arrow(ax, a, b)
    _arrow(ax, b, c)
    _arrow(ax, c, d)
    _arrow(ax, d, e)
    _arrow(ax, e, f, "да")
    _arrow(ax, e, g, "нет")
    _arrow(ax, g, hh)
    _arrow(ax, hh, i)
    _arrow(ax, i, j)
    _arrow(ax, j, k, "нет")
    _arrow(ax, j, ll, "да")
    _arrow(ax, ll, m)
    _arrow(ax, k, n)
    _arrow(ax, m, n)
    # ветвь eigen: от НИЖНЕЙ кромки своего блока к ЛЕВОЙ кромке Result
    _arrow(ax, (f[0], f[1] - 0.45), (o[0] - 2.4, o[1]))
    _arrow(ax, n, o)
    _arrow(ax, o, p)
    _arrow(ax, p, q)
    ax.set_title("Диспетчер plate-solver v0.8.0 (ГОСТ 19.701-90)", fontsize=11)

    out = Path(__file__).with_name("dispatch_flow.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
