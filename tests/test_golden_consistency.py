"""Согласованность реестра с golden (фаза 2, P4.3): круг ↔ таблица 4.1.

p-свип круга через ДИСПЕТЧЕР обязан воспроизводить w_max таблицы 4.1
golden_results.md до 12 значащих цифр: и case-слой, и golden-скрипты
строят один и тот же PlateBending — арифметика совпадает бит-в-бит,
таблица напечатана с 6 цифрами, поэтому сравниваем разбор текста
с прогоном на rel ≤ 5e-7 (полуединица последней печатной цифры),
а «12 цифр» гарантируем сравнением самих чисел прогона с прямым API.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.dispatch import solve
from plate_solver.plate import PlateBending
from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]


def _golden_table_41() -> dict[int, float]:
    """Разобрать строки | p | w_max | ... | таблицы 4.1 golden_results.md."""
    text = (_ROOT / "golden_results.md").read_text(encoding="utf-8")
    block = text.split("Таблица 4.1")[1].split("##")[0]
    rows = {}
    for m in re.finditer(r"^\|\s*(\d+)\s*\|\s*([0-9.e+-]+)\s*\|", block, re.MULTILINE):
        rows[int(m.group(1))] = float(m.group(2))
    return rows


def _problem(p: int) -> Problem:
    return Problem.from_dict({
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"h": 1.0},                       # круг golden: h_circle = 1.0
        "discretization": {"p": p, "Q": 1024, "grid_n": 80},
    })


@pytest.mark.big
def test_gate_circle_sweep_matches_golden_table41():
    """ВОРОТА: p-свип диспетчера ↔ таблица 4.1 (печать) и прямой API (12 цифр)."""
    table = _golden_table_41()
    assert sorted(table) == [2, 4, 6, 8, 10]
    dom = geometry.make_circle(1.0)
    for p, w_printed in table.items():
        res = solve(_problem(p))
        w0 = float(res._plate.deflection(res._c, 0.0, 0.0))    # тот же функционал
        # прямой API — та же арифметика: совпадение до 12 значащих цифр
        pb = PlateBending.from_config(dom, Config(a=1.0, q0=4.0, h=1.0, p=p,
                                                  Q=1024, grid_n=80))
        _, cw = pb.solve_uniform(4.0)
        w_api = float(pb.deflection(cw, 0.0, 0.0))
        assert abs(w0 - w_api) <= 1e-12 * abs(w_api), (p, w0, w_api)
        # печать golden — 6 значащих цифр
        assert abs(w0 - w_printed) <= 5e-7 * abs(w_printed), (p, w0, w_printed)
