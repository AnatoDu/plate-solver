"""Свободный край (F10): SFSF/CFFF, правило жёстких смещений, эталоны.

Свободная сторона — чисто ЕСТЕСТВЕННОЕ условие полной билинейной формы
(M_n = 0, обобщённая перерезывающая Кирхгофа V_n = 0): множителя в
структуру не даёт. т1 — SFSF против ряда Леви-free (фабрика); т2/т3 —
против независимого Кирхгофа (Морли, маркер fem); т5 — FFFF/SFFF
отклоняются (ядро {1, x, y}). т4 (SSSS ≡ mixed) — существующие ворота
tests/test_mixed_bc.py, не дублируются.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plate_solver.analytic_auto import levy_solution
from plate_solver.clamped import MixedRectPlate
from plate_solver.config import Config
from plate_solver.dispatch import solve
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]
CFG = Config(q0=4.0, h=0.06, p=12, Q=160)
RECT = (0.0, 1.0, 0.0, 1.0)

SFSF_SIDES = {"x1": "hinge", "x2": "hinge", "y1": "free", "y2": "free"}
CFFF_SIDES = {"x1": "clamped", "x2": "free", "y1": "free", "y2": "free"}


def _case(sides: dict, reference: str = "none", tol: float = 1.0) -> dict:
    return {"geometry": {"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                         "y1": 0.0, "y2": 1.0},
            "bc": {"type": "mixed",
                   "sides": [{"side": k, "type": v} for k, v in sides.items()]},
            "load": {"type": "uniform", "q0": 4.0},
            "model": {"h": 0.06},
            "discretization": {"p": 12, "Q": 160, "grid_n": 16},
            "verify": {"reference": reference, "tol": tol}}


def test_gate_t1_sfsf_vs_levy_free():
    """т1: SFSF ↔ Леви-free — потолок 1e-4 перекрыт; факт 6.7e-9 × 3 = 2e-8.

    Плюс SPD после исключения жёстких мод (cond(S) конечен) и физика:
    прогиб свободной кромки БОЛЬШЕ центрального (кромка не несёт).
    """
    m = MixedRectPlate(*RECT, SFSF_SIDES, CFG)
    c = m.solve_uniform()
    sol = levy_solution(x1=0, x2=1, y1=0, y2=1, D=CFG.D, q0=CFG.q0,
                        bc_y1="free", bc_y2="free", nu=CFG.nu)
    worst = 0.0
    for pt in ((0.5, 0.5), (0.5, 0.0), (0.5, 1.0), (0.3, 0.8), (0.7, 0.2)):
        w_num = float(m.deflection(c, *pt))
        w_ref = float(sol.w(*pt))
        worst = max(worst, abs(w_num - w_ref) / abs(w_ref))
    assert worst <= 2.0e-8
    assert np.isfinite(np.linalg.cond(m.S))
    assert float(m.deflection(c, 0.5, 0.0)) > float(m.deflection(c, 0.5, 0.5))


@pytest.mark.fem
def test_gate_t2_sfsf_vs_morley():
    """т2: SFSF ↔ Морли (независимый Кирхгоф) — факт 1.10e-3 × 3 = 3.3e-3."""
    from plate_solver.references import verify_result

    rep = verify_result(solve(Problem.from_dict(_case(SFSF_SIDES, "fem"))))
    assert rep.rows[0].rel <= 3.3e-3, "\n" + rep.table()


@pytest.mark.fem
def test_gate_t3_cfff_cantilever_vs_morley():
    """т3: консоль CFFF ↔ Морли — факт 1.19e-3 × 3 = 3.6e-3; физика:
    w_max на свободном крае напротив заделки, монотонный рост к кромке."""
    from plate_solver.references import verify_result

    res = solve(Problem.from_dict(_case(CFFF_SIDES, "fem")))
    rep = verify_result(res)
    assert rep.rows[0].rel <= 3.6e-3, "\n" + rep.table()
    m, c = res._plate, res._c
    xs = np.linspace(0.05, 1.0, 24)
    w_line = np.asarray(m.deflection(c, xs, 0.5 + 0 * xs), float)
    assert np.all(np.diff(w_line) > 0.0)             # монотонный рост к кромке
    w_edge = float(m.deflection(c, 1.0, 0.5))
    inside = np.isfinite(res.w_grid)
    assert w_edge >= 0.99 * float(np.nanmax(np.abs(res.w_grid[inside])))


def test_gate_t5_rigid_motion_rule():
    """т5: FFFF и SFFF отклоняются (жёсткие смещения); CFFF/SFSF проходят."""
    ffff = {"x1": "free", "x2": "free", "y1": "free", "y2": "free"}
    sfff = {"x1": "hinge", "x2": "free", "y1": "free", "y2": "free"}
    for sides in (ffff, sfff):
        with pytest.raises(CaseError, match="жёсткие смещения"):
            Problem.from_dict(_case(sides))
        with pytest.raises(ValueError, match="жёсткие смещения"):
            MixedRectPlate(*RECT, sides, CFG)
    for sides in (CFFF_SIDES, SFSF_SIDES,
                  {"x1": "hinge", "x2": "free", "y1": "hinge", "y2": "free"}):
        Problem.from_dict(_case(sides))              # смежные hinge — тоже ок


def test_free_ci_case_runs_and_ladder_schema():
    """ci-копии валидны; ladder-файлы парсятся с верными эталонами."""
    for name, ref in (("rect_sfsf", "analytic"), ("rect_cfff", "fem")):
        p = Problem.from_toml(_ROOT / "cases" / "ladder" / f"{name}.toml")
        assert p.verify.reference == ref
    p = Problem.from_toml(_ROOT / "cases" / "ci" / "rect_cfff.toml")
    assert p.verify.reference == "none"              # CI без scikit-fem


@pytest.mark.big
@pytest.mark.fem
def test_ladder_cfff_gate():
    """Боевые ворота ladder-ступени консоли (fem-эталон)."""
    from plate_solver.references import verify_result

    res = solve(Problem.from_toml(_ROOT / "cases" / "ladder" / "rect_cfff.toml"))
    rep = verify_result(res)
    assert rep.ok, "\n" + rep.table()


@pytest.mark.big
def test_ladder_sfsf_gate():
    """Боевые ворота ladder-ступени SFSF (Леви-free)."""
    from plate_solver.references import verify_result

    res = solve(Problem.from_toml(_ROOT / "cases" / "ladder" / "rect_sfsf.toml"))
    rep = verify_result(res)
    assert rep.ok, "\n" + rep.table()


def test_symmetry_sfsf():
    """Симметрия SFSF: w(x, y) = w(x, 1 − y) = w(1 − x, y) (машинно)."""
    m = MixedRectPlate(*RECT, SFSF_SIDES, CFG)
    c = m.solve_uniform()
    pts = [(0.3, 0.2), (0.6, 0.75), (0.25, 0.4)]
    for x, y in pts:
        w0 = float(m.deflection(c, x, y))
        assert w0 == pytest.approx(float(m.deflection(c, x, 1 - y)), rel=1e-9)
        assert w0 == pytest.approx(float(m.deflection(c, 1 - x, y)), rel=1e-9)
