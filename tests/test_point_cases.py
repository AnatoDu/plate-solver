"""Cases точечной силы: ворота потолка на рабочем eps=0.025.

Монотонность ε-свипа НЕ гейтится (fallback, обоснование — PROGRESS.md и
cases/baselines.json): базисная ошибка глобальных полиномов на w ~ r²ln r
превышает ошибку регуляризации; свип зафиксирован информационно.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from plate_solver.problem import Problem

_CASES = Path(__file__).resolve().parents[1] / "cases" / "ladder"


@pytest.mark.parametrize("case", ["circle_point_clamped", "circle_point_soft"])
def test_point_case_files_valid(case):
    p = Problem.from_toml(_CASES / f"{case}.toml")
    assert p.load.type == "point" and p.load.eps == 0.025
    assert (p.load.x0, p.load.y0) == (0.0, 0.0)
    assert p.to_config().p == 16


@pytest.mark.big
@pytest.mark.parametrize("case", ["circle_point_clamped", "circle_point_soft"])
def test_gate_point_case(case):
    """ВОРОТА: рабочая точка (p=16, Q=1024, eps=0.025) в замороженном допуске.

    Проверяется ИМЕННО нагрузка: пятно не расширялось (узлов ≥ 20) и сила не
    теряется через границу. С v0.8.0 в ``Result.warnings`` попадают ещё и
    деградации факторизации: на этой рабочей точке (p = 16 ⇒ N = 289) матрица
    защемления численно вырождена, и каскад доходит до спектрального
    псевдообращения — на части платформ (BLAS) отбрасывается около
    полутора десятков почти-нулевых направлений. Прежде тот же путь МОЛЧА
    уходил в МНК (находка P02), поэтому ворота требуют не отсутствия
    предупреждения, а того, чтобы усечение оставалось МАЛЫМ и эталон
    выполнялся: сама проверка точности — строка ``verify_result`` ниже.
    """
    from plate_solver.dispatch import solve
    from plate_solver.references import verify_result

    p = Problem.from_toml(_CASES / f"{case}.toml")
    res = solve(p)
    load_warnings = [w for w in res.warnings if w.startswith("load.")]
    assert not load_warnings                      # пятно не расширялось (узлов ≥ 20)
    for w in res.warnings:                        # усечение — только «хвост» спектра
        if "отсечено" in w:
            dropped, total = (int(t) for t in re.findall(r"отсечено (\d+) из (\d+)", w)[0])
            assert dropped < 0.1 * total, w
    report = verify_result(res)
    assert report.ok, "\n" + report.table()
