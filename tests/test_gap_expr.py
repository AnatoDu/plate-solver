r"""Ворота профиля зазора выражением — [contact] gap_expr (v0.7.0).

Три уровня верификации:

* ТОЖДЕСТВО константы: gap_expr="0.005" редуцируется в float и идёт путём
  скалярного ``gap`` — решение БИТ-ТОЧНО совпадает;
* ТОЖДЕСТВО параболоида: gap_expr-парабола против уже верифицированного
  ``[contact.gap] kind="paraboloid"`` (различие — только fp-ассоциативность
  lambdify, измерено ~1e-15);
* ВЗАИМНЫЙ СЕРТИФИКАТ двух независимых дискретизаций (методика NOTES §25):
  КР-кирпич ``fd_contact`` с gap-МАССИВОМ против RFM+Ритц с тем же
  параболическим штампом — измерено rel(w_max)=4.8e-3, rel(∫r)=8.0e-3 на
  КР 135×81 (убывает с h: сеточная природа), допуск 1.5e-2;
* редукция: большой зазор ⇒ r ≡ 0, прогиб = свободному.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.fd_contact import fd_contact_foundation
from plate_solver.ladder import navier_uniform_center
from plate_solver.problem import CaseError, Problem

_RECT = (0.0, 2.0, 0.0, 1.2)


def _contact_case(**contact):
    x1, x2, y1, y2 = _RECT
    return {
        "geometry": {"kind": "rectangle", "x1": x1, "x2": x2, "y1": y1, "y2": y2},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 1.0},
        # E, h подобраны так, что D = 1 (как в test_fd_contact)
        "model": {"theory": "classic", "E": 12 * (1 - 0.3**2), "nu": 0.3, "h": 1.0},
        "contact": {"enabled": True, "max_iter": 4000, "tol": 1.0e-8, **contact},
        "discretization": {"p": 12, "Q": 200, "grid_n": 32},
        "verify": {"reference": "none"},
    }


def test_gap_expr_const_identity():
    """gap_expr-константа = скалярный gap БИТ-ТОЧНО (редукция в float)."""
    r1 = dispatch.solve(Problem.from_dict(_contact_case(gap=0.005)))
    r2 = dispatch.solve(Problem.from_dict(_contact_case(gap_expr="0.005")))
    assert r1.w_max == r2.w_max
    assert np.array_equal(r1.contact.r_nodes, r2.contact.r_nodes)


@pytest.mark.big          # ~60 c МОР Q=200: тяжёлый дублирующий
def test_gap_expr_matches_paraboloid_spec():
    """gap_expr-парабола = [contact.gap] paraboloid (один gap-массив)."""
    z0, R, cx, cy = 5.0e-3, 3.0, 1.0, 0.6
    d1 = _contact_case(gap={"kind": "paraboloid", "apex": z0, "r_curv": R,
                            "cx": cx, "cy": cy})
    d2 = _contact_case(
        gap_expr=f"{z0} + ((x-{cx})**2+(y-{cy})**2)/(2*{R})")
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 1e-13
    tot1 = float(np.sum(r1.contact.r_nodes * r1._plate.quad.w))
    tot2 = float(np.sum(r2.contact.r_nodes * r2._plate.quad.w))
    assert abs(tot1 - tot2) / tot1 < 1e-13


def test_gap_expr_fd_certificate():
    """Взаимный сертификат КР↔RFM на параболическом штампе (NOTES §25).

    Независимые дискретизации, один профиль Δ(x, y); измерено 4.8e-3/8.0e-3
    на КР 135×81 (при 269×161 — 2.1e-3/5.8e-3: расхождение сеточное).
    """
    x1, x2, y1, y2 = _RECT
    w_ex, _ = navier_uniform_center(x2 - x1, y2 - y1, 1.0, 1.0)
    z0, R, cx, cy = 0.3 * w_ex, 3.0, 1.0, 0.6
    res = dispatch.solve(Problem.from_dict(_contact_case(
        gap_expr=f"{z0} + ((x-{cx})**2+(y-{cy})**2)/(2*{R})")))
    r_total_rfm = float(np.sum(res.contact.r_nodes * res._plate.quad.w))

    # тот же профиль Δ(x, y) как gap-МАССИВ на внутренней сетке КР
    from plate_solver.fd_contact import FDPlateSS
    fd = FDPlateSS(x1, x2, y1, y2, 135, 81, 1.0)
    gap_arr = z0 + ((fd.X - cx) ** 2 + (fd.Y - cy) ** 2) / (2.0 * R)
    fdc = fd_contact_foundation(x1, x2, y1, y2, D=1.0, q0=1.0, gap=gap_arr,
                                nx=135, ny=81, tol=1e-7)
    assert fdc.n_contact > 0 and np.all(fdc.r >= 0.0)
    assert abs(fdc.w_max - res.w_max) / res.w_max < 1.5e-2
    assert abs(fdc.r_total - r_total_rfm) / r_total_rfm < 1.5e-2


def test_gap_expr_reduction_big_gap():
    """Большой зазор-ПОЛЕ ⇒ r ≡ 0, прогиб = свободному (редукция R1)."""
    d = _contact_case(gap_expr="1.0 + 0.001*x")
    res = dispatch.solve(Problem.from_dict(d))
    assert float(np.max(res.contact.r_nodes)) == 0.0
    free = copy.deepcopy(d)
    free["contact"] = {"enabled": False}
    res_free = dispatch.solve(Problem.from_dict(free))
    assert res.w_max == res_free.w_max


def test_gap_expr_nonlinear_route():
    """Маршрутизация в нелинейный тракт: gap_expr = paraboloid у МОР+КТН."""
    z0, R, cx, cy = 2.0e-1, 3.0, 0.0, 0.0
    base = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "ktn_full"},
        "contact": {"enabled": True, "max_iter": 300, "tol": 1.0e-6},
        "discretization": {"p": 7, "Q": 48, "grid_n": 24},
        "verify": {"reference": "none"},
    }
    d1 = copy.deepcopy(base)
    d1["contact"]["gap"] = {"kind": "paraboloid", "apex": z0, "r_curv": R,
                            "cx": cx, "cy": cy}
    d2 = copy.deepcopy(base)
    d2["contact"]["gap_expr"] = f"{z0} + ((x-{cx})**2+(y-{cy})**2)/(2*{R})"
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 1e-13


@pytest.mark.parametrize("bad", [
    "__import__('os').system('true')",
    "z + x",
    "open('f')",
    "'abc'",
])
def test_gap_expr_rejected(bad):
    with pytest.raises(CaseError, match="gap_expr"):
        Problem.from_dict(_contact_case(gap_expr=bad))


def test_gap_expr_both_sources_rejected():
    """gap_expr вместе с gap | [contact.gap] — «ровно одно из»."""
    with pytest.raises(CaseError):
        Problem.from_dict(_contact_case(gap=0.005, gap_expr="0.005"))
    with pytest.raises(CaseError, match="gap_expr"):
        Problem.from_dict(_contact_case(
            gap={"kind": "const", "value": 0.005}, gap_expr="0.005"))


def test_gap_expr_nonpositive_rejected():
    """Δ ≤ 0 где-то на основании — существующий отказ позиционного контакта."""
    with pytest.raises(CaseError):
        dispatch.solve(Problem.from_dict(_contact_case(gap_expr="0.005*(x-1.0)")))


def test_gap_expr_nonfinite_rejected():
    """Выражение с NaN на области (log вне определения) — CaseError."""
    with pytest.raises(CaseError, match="gap_expr"):
        dispatch.solve(Problem.from_dict(_contact_case(gap_expr="log(x - 1.0)")))


def test_gap_expr_pair_negative_rejected():
    """Пара пластин: gap_expr с min Δ < 0 — отказ (зеркало скалярного правила)."""
    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "classic"},
        "contact": {"enabled": True, "target": "plate2",
                    "gap_expr": "0.005*(x - 0.5)"},
        "plate2": {"bc": {"type": "soft_hinge"},
                   "load": {"type": "uniform", "q0": 0.0}},
        "discretization": {"p": 8, "Q": 48, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    with pytest.raises(CaseError, match="gap_expr"):
        dispatch.solve(Problem.from_dict(d))
