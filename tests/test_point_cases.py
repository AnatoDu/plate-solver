"""Cases точечной силы (фаза 2, P3.5): ворота потолка на рабочем eps=0.025.

Монотонность ε-свипа НЕ гейтится (fallback, обоснование — PROGRESS.md и
cases/baselines.json): базисная ошибка глобальных полиномов на w ~ r²ln r
превышает ошибку регуляризации; свип зафиксирован информационно.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plate_solver.problem import Problem

_CASES = Path(__file__).resolve().parents[1] / "cases"


@pytest.mark.parametrize("case", ["circle_point_clamped", "circle_point_soft"])
def test_point_case_files_valid(case):
    p = Problem.from_toml(_CASES / f"{case}.toml")
    assert p.load.type == "point" and p.load.eps == 0.025
    assert (p.load.x0, p.load.y0) == (0.0, 0.0)
    assert p.to_config().p == 16


@pytest.mark.big
@pytest.mark.parametrize("case", ["circle_point_clamped", "circle_point_soft"])
def test_gate_point_case(case):
    """ВОРОТА: рабочая точка (p=16, Q=1024, eps=0.025) в замороженном допуске."""
    from plate_solver.dispatch import solve
    from plate_solver.references import verify_result

    p = Problem.from_toml(_CASES / f"{case}.toml")
    res = solve(p)
    assert not res.warnings                       # пятно не расширялось (узлов ≥ 20)
    report = verify_result(res)
    assert report.ok, "\n" + report.table()
