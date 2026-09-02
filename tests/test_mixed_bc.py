"""Смешанные краевые условия на прямоугольнике.

C1: полная билинейная форма (гауссов член); унификация НЕ форсируется —
дискретный нуль-лагранжиан нарушается маской/изломами ω (факты в PROGRESS).
C2: структура (∏ω_c²)(∏ω_h)·Φ — полиномиальная, квадратура точна.
C3: т1 CCCC ≡ clamped-путь (потолок по факту разных структур);
т2 Навье; т3 Леви (sympy-чек ОДУ/КУ); т4 монотонность.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import sympy as sp

from plate_solver import analytic, geometry
from plate_solver.clamped import ClampedPlate, MixedRectPlate
from plate_solver.config import Config
from plate_solver.dispatch import solve
from plate_solver.problem import CaseError, Problem

CFG = Config(q0=4.0, h=0.06, p=12, Q=160)
RECT = (0.0, 1.0, 0.0, 1.0)


def _mixed(**sides) -> MixedRectPlate:
    return MixedRectPlate(*RECT, sides, CFG)


def test_levy_sympy_ode_and_bc():
    """т3-символика: Y_m удовлетворяет ОДУ и КУ Леви точно (для любого m)."""
    y, al, qm, D, c = sp.symbols("y alpha q_m D c", positive=True)
    yp = qm / (D * al**4)
    ch, sh = sp.cosh(al * c), sp.sinh(al * c)
    det = ch * (sh + al * c * ch) - c * sh * al * sh
    A = (-yp) * (sh + al * c * ch) / det
    B = yp * al * sh / det
    Y = yp + A * sp.cosh(al * y) + B * y * sp.sinh(al * y)
    ode = sp.simplify(sp.diff(Y, y, 4) - 2 * al**2 * sp.diff(Y, y, 2)
                      + al**4 * Y - qm / D)
    assert ode == 0
    assert sp.simplify(Y.subs(y, c)) == 0
    assert sp.simplify(sp.diff(Y, y).subs(y, c)) == 0


def test_gate_t1_all_clamped_equals_clamped_path():
    """т1: CCCC-mixed ≡ clamped-путь. Потолок пересмотрен по факту (журнал):

    структуры РАЗНЫЕ (полином против R-функции с изломами) — тождества
    1e-12 нет; факт 4.18e-7 → заморожено 1.3e-6.
    """
    mc = _mixed(x1="clamped", x2="clamped", y1="clamped", y2="clamped")
    cc = mc.solve_uniform()
    cp = ClampedPlate.from_config(geometry.make_rectangle(*RECT), CFG)
    c2 = cp.solve_uniform(CFG.q0)
    w1 = float(mc.deflection(cc, 0.5, 0.5))
    w2 = float(cp.deflection(c2, 0.5, 0.5))
    assert abs(w1 - w2) / abs(w2) <= 1.3e-6


def test_gate_t2_all_hinge_navier():
    """т2: SSSS ↔ Навье (контроль остатка ряда) — факт 7.12e-9 × 3."""
    mp_ = _mixed(x1="hinge", x2="hinge", y1="hinge", y2="hinge")
    c = mp_.solve_uniform()
    w = float(mp_.deflection(c, 0.5, 0.5))
    w_ref = float(analytic.navier_rect_uniform(0.5, 0.5, *RECT, CFG.q0, CFG.D))
    assert abs(w - w_ref) / abs(w_ref) <= 2.2e-8


def test_levy_series_stable_for_elongated_plates():
    """ГЛАВНЫЕ ВОРОТА (v0.8.0): ряд Леви конечен и даёт оба цилиндрических предела.

    Прямое вычисление ``ch(αc)``/``sh(αc)`` переполняется при ``αc > 710`` —
    уже при отношении сторон ≈ 1.9 и 60 членах ряда эталон обращался в NaN, и
    ``plate-verify`` «проваливал» ВЕРНЫЙ расчёт (аудит V01). Устойчивая форма
    (деление на ``ch²``) проверяется двумя ТОЧНЫМИ асимптотами:

    * вытянутая в y (шарнирные короткие кромки уходят на бесконечность):
      цилиндрический изгиб шарнирной полосы ``w = 5qL_x⁴/(384D)``;
    * вытянутая в x: цилиндрический изгиб ЗАЩЕМЛЁННОЙ полосы
      ``w = qL_y⁴/(384D)``; здесь синус-разложение идёт вдоль ДЛИННОЙ стороны и
      ряд сходится медленно — при 60 членах остаточная ошибка 1.6e-5, при 200 —
      3.9e-8 (проверено), поэтому предел берётся с 200 членами.
    """
    for ratio in (1.0, 1.9, 2.5, 4.0, 8.0, 20.0):
        w = analytic.levy_rect_uniform(np.array([0.5]), np.array([0.5 * ratio]),
                                       0.0, 1.0, 0.0, ratio, q=1.0, D=1.0)
        assert np.all(np.isfinite(w)), f"NaN/inf при Ly/Lx = {ratio}"
    # предел вытянутой в y: шарнирная полоса 5qL⁴/384D
    w_y = float(analytic.levy_rect_uniform(np.array([0.5]), np.array([10.0]),
                                           0.0, 1.0, 0.0, 20.0, q=1.0, D=1.0)[0])
    assert w_y == pytest.approx(5.0 / 384.0, rel=1e-6)
    # предел вытянутой в x: защемлённая полоса qL⁴/384D
    w_x = float(analytic.levy_rect_uniform(np.array([10.0]), np.array([0.5]),
                                           0.0, 20.0, 0.0, 1.0, q=1.0, D=1.0,
                                           n_terms=200)[0])
    assert w_x == pytest.approx(1.0 / 384.0, rel=1e-6)


def test_gate_t3_levy():
    """т3: SCSC (x-hinge, y-clamped) ↔ ряд Леви — факт 6.96e-9 × 3."""
    ml = _mixed(x1="hinge", x2="hinge", y1="clamped", y2="clamped")
    c = ml.solve_uniform()
    w = float(ml.deflection(c, 0.5, 0.5))
    w_ref = float(analytic.levy_rect_uniform(0.5, 0.5, *RECT, CFG.q0, CFG.D))
    assert abs(w - w_ref) / abs(w_ref) <= 2.1e-8


def test_gate_t4_monotone_between_limits():
    """т4 (физика-инфо): w_max(SSSS) > w_max(Леви) > w_max(CCCC)."""
    ws = {}
    for name, sides in (
        ("ssss", dict(x1="hinge", x2="hinge", y1="hinge", y2="hinge")),
        ("levy", dict(x1="hinge", x2="hinge", y1="clamped", y2="clamped")),
        ("cccc", dict(x1="clamped", x2="clamped", y1="clamped", y2="clamped")),
    ):
        m = _mixed(**sides)
        ws[name] = m.w_max_on_grid(m.solve_uniform())
    assert ws["ssss"] > ws["levy"] > ws["cccc"]


# --------------------------------------------------------------------------- #
#  Схема и диспетчер
# --------------------------------------------------------------------------- #
BASE = {
    "geometry": {"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
    "bc": {"type": "mixed", "sides": [
        {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
        {"side": "y1", "type": "clamped"}, {"side": "y2", "type": "clamped"}]},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"h": 0.06},
    "discretization": {"p": 12, "Q": 160, "grid_n": 16},
    "verify": {"reference": "analytic", "tol": 1.0e-4},
}


def _problem(**over) -> Problem:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = copy.deepcopy(v)
    return Problem.from_dict(d)


def test_mixed_schema_errors():
    with pytest.raises(CaseError, match="четыре стороны"):
        _problem(bc={"type": "mixed", "sides": [{"side": "x1", "type": "hinge"}]})
    with pytest.raises(CaseError, match="один раз"):
        _problem(bc={"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x1", "type": "clamped"},
            {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]})
    with pytest.raises(CaseError, match="rectangle"):
        _problem(geometry={"kind": "circle", "a": 1.0})
    with pytest.raises(CaseError, match="контакт"):
        _problem(contact={"enabled": True, "gap_factor": 0.5})
    with pytest.raises(CaseError, match="только при type"):
        _problem(bc={"type": "clamped", "sides": [{"side": "x1", "type": "hinge"}]})


def test_mixed_dispatch_and_reference_end_to_end(tmp_path):
    """Диспетчер + резолвер (Леви в центре) + fields.npz с моментами mixed."""
    res = solve(_problem())
    from plate_solver.references import verify_result

    rep = verify_result(res)
    assert rep.ok, "\n" + rep.table()
    assert rep.rows[0].rel <= 2.1e-8 * 3            # запас поверх замороженного
    res.save(tmp_path)
    data = np.load(tmp_path / "fields.npz")
    inside = np.isfinite(data["w"])
    assert np.isfinite(data["Mx"][inside]).all()    # моменты mixed-структуры


def test_rect_soft_hinge_gets_navier_reference():
    """C3-т2 (попутно): rect/soft_hinge гейтится Навье (прямые края, NOTES §8)."""
    from plate_solver.references import verify_result

    p = _problem(bc={"type": "soft_hinge"},
                 discretization={"p": 10, "Q": 160, "grid_n": 16},
                 verify={"reference": "analytic", "tol": 5.6e-6})
    rep = verify_result(solve(p))
    assert rep.ok, "\n" + rep.table()               # факт 1.85e-6 × 3
