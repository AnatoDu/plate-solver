"""Ступень «кольцо, изгиб» (фаза 2, P3.4): cases annulus_soft / annulus_clamped.

Допуск в case-файлах откалиброван протоколом «потолок 1e-2 → факт × 3»
(TODO_PHASE2, автономный режим п. 1); факты калибровки — в PROGRESS.md.
Полные ворота (Q=1024) — big; валидация case-файлов — лёгкая.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plate_solver.problem import Problem

_CASES = Path(__file__).resolve().parents[1] / "cases"


@pytest.mark.parametrize("case", ["annulus_soft", "annulus_clamped"])
def test_case_files_valid(case):
    """Case-файлы кольца валидируются и строят Config (лёгкая проверка)."""
    p = Problem.from_toml(_CASES / f"{case}.toml")
    assert p.geometry.kind == "annulus" and (p.geometry.a, p.geometry.b) == (1.0, 0.4)
    cfg = p.to_config()
    assert cfg.Q == 1024 and cfg.p == 10
    assert p.verify.reference == "analytic" and p.verify.cross_1d


@pytest.mark.big
@pytest.mark.parametrize("case", ["annulus_soft", "annulus_clamped"])
def test_gate_annulus_case(case):
    """ВОРОТА: 2D↔analytic и 2D↔1D в замороженном допуске (Q=1024, p=10)."""
    from plate_solver.dispatch import solve
    from plate_solver.references import verify_result

    p = Problem.from_toml(_CASES / f"{case}.toml")
    report = verify_result(solve(p))
    assert report.ok, "\n" + report.table()
    gated = [r for r in report.rows if r.gated]
    assert len(gated) == 2                        # analytic + 1D-Ритц
