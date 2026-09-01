r"""Ворота термоизгиба — [load] thermal_moment = M_T (v0.7.0).

Эталоны (Boley–Weiner гл. 12, Тимошенко §14):

* защемление НЕ гнётся константным M_T: ``−∫M_T·Δψ dΩ = −M_T·∮ψ_n ds ≡ 0``
  для ψ = ω²T — прогиб w бит-точно прежний (b не собирается по построению),
  сдвигаются только моменты (−M_T);
* ИСТИННЫЙ шарнир (полная форма, karman): круг → точная СФЕРА
  ``w = M_T(a²−r²)/(2D(1+ν))`` (измерено 8.1e-11 — предел нелинейной
  поправки O((w/h)²));
* модель РАСЩЕПЛЕНИЯ (classic soft_hinge): на прямых кромках = истинный SS
  (квадрат: w_центра = (M_T/D)·u_c, u_c — функция кручения, 2.4e-6);
  на КРИВОЙ границе — модельное решение ``M_T(a²−r²)/(4D)`` (не истинный
  SS: фактор 2/(1+ν); документировано — аналог парадокса мягкого шарнира);
* суперпозиция и редукция M_T=0 — бит-точно.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.config import Config
from plate_solver.problem import CaseError, Problem

_D = Config().D
_MT = 3.0


def _case(bc="clamped", theory="classic", q0=4.0, mt=None, kind="circle",
          p=12, Q=64):
    d = {
        "geometry": ({"kind": "circle", "a": 1.0} if kind == "circle"
                     else {"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                           "y1": 0.0, "y2": 1.0}),
        "bc": {"type": bc},
        "load": {"type": "uniform", "q0": q0},
        "model": {"theory": theory},
        "discretization": {"p": p, "Q": Q, "grid_n": 32},
        "verify": {"reference": "none"},
    }
    if mt is not None:
        d["load"]["thermal_moment"] = mt
    return d


def test_thermal_clamped_w_bitexact_moments_shifted():
    """Защемление: w бит-точно не меняется; моменты сдвинуты на −M_T."""
    r0 = dispatch.solve(Problem.from_dict(_case()))
    r1 = dispatch.solve(Problem.from_dict(_case(mt=_MT)))
    assert np.array_equal(r0.w_grid, r1.w_grid, equal_nan=True)
    mx0, my0, mxy0 = r0.moments_on_grid()
    mx1, my1, mxy1 = r1.moments_on_grid()
    m = np.isfinite(mx0)
    assert np.allclose(mx1[m] - mx0[m], -_MT, atol=1e-10 * _MT)
    assert np.allclose(my1[m] - my0[m], -_MT, atol=1e-10 * _MT)
    assert np.array_equal(mxy0, mxy1, equal_nan=True)   # Mxy без сдвига


def test_thermal_clamped_pure_zero():
    """Чистый термо на защемлении: w ≡ 0 ТОЧНО (b — аналитический ноль)."""
    res = dispatch.solve(Problem.from_dict(_case(q0=0.0, mt=_MT)))
    assert res.w_max == 0.0
    res_k = dispatch.solve(Problem.from_dict(
        _case(theory="karman", q0=0.0, mt=_MT)))
    assert res_k.w_max == 0.0


def test_thermal_karman_true_hinge_sphere():
    """ГЛАВНЫЕ ворота: истинный шарнир (полная форма) → точная сфера.

    ``w = M_T(a²−r²)/(2D(1+ν))``; измерено 8.1e-11 (нелинейная поправка
    O((w/h)²) ≈ 4e-11 — предел); допуск 1e-8 — запас ×100 БЕЗ ослабления сути.
    """
    res = dispatch.solve(Problem.from_dict(
        _case(bc="soft_hinge", theory="karman", q0=0.0, mt=_MT)))
    rr = np.array([0.0, 0.3, 0.6, 0.9])
    w_num = np.asarray(res._plate.deflection(res._c, rr, np.zeros(4)), float)
    w_ex = _MT * (1.0 - rr**2) / (2.0 * _D * (1.0 + 0.3))
    assert np.max(np.abs(w_num - w_ex) / np.abs(w_ex)) < 1e-8


def test_thermal_soft_hinge_circle_model():
    """Модель расщепления на круге: w(0) = M_T·a²/(4D) (модельная точная).

    НЕ истинный SS (тот — сфера /2D(1+ν); фактор 2/(1+ν) ≈ 1.54 при ν=0.3 —
    документированный аналог парадокса мягкого шарнира); измерено 5.3e-3
    при Q=64 (маска O(1/Q), сходимость немонотонна) — допуск 2e-2.
    """
    res = dispatch.solve(Problem.from_dict(
        _case(bc="soft_hinge", q0=0.0, mt=_MT)))
    w0 = float(res._plate.deflection(res._c, np.array([0.0]),
                                     np.array([0.0]))[0])
    w_model = _MT / (4.0 * _D)
    assert abs(w0 - w_model) / w_model < 2e-2


def test_thermal_soft_hinge_square_true_ss():
    """Квадрат (прямые кромки): расщепление = истинный SS.

    w_центра = (M_T/D)·u_c, u_c = 0.0736713523 — функция кручения
    (−Δu = 1, u|∂ = 0 на [0,1]²; ряд); измерено 2.4e-6 — допуск 1e-4.
    """
    res = dispatch.solve(Problem.from_dict(
        _case(bc="soft_hinge", q0=0.0, mt=_MT, kind="rect")))
    w_c = float(res._plate.deflection(res._c, np.array([0.5]),
                                      np.array([0.5]))[0])
    w_ex = (_MT / _D) * 0.0736713523
    assert abs(w_c - w_ex) / w_ex < 1e-4


def test_thermal_superposition_linear():
    """Линейность: w(q0, M_T) = w(q0, 0) + w(0, M_T) (измерено 5e-16)."""
    r_q = dispatch.solve(Problem.from_dict(_case(bc="soft_hinge", kind="rect")))
    r_t = dispatch.solve(Problem.from_dict(
        _case(bc="soft_hinge", q0=0.0, mt=_MT, kind="rect")))
    r_qt = dispatch.solve(Problem.from_dict(
        _case(bc="soft_hinge", mt=_MT, kind="rect")))
    m = np.isfinite(r_qt.w_grid)
    lhs = r_qt.w_grid[m]
    rhs = r_q.w_grid[m] + r_t.w_grid[m]
    assert np.max(np.abs(lhs - rhs)) <= 1e-12 * np.max(np.abs(lhs))


def test_thermal_zero_reduction_bitexact():
    """thermal_moment = 0.0 явно ≡ ключ отсутствует (бит-точно, все тракты)."""
    for bc, theory in (("clamped", "classic"), ("soft_hinge", "classic"),
                       ("soft_hinge", "karman")):
        r0 = dispatch.solve(Problem.from_dict(_case(bc=bc, theory=theory)))
        r1 = dispatch.solve(Problem.from_dict(_case(bc=bc, theory=theory,
                                                    mt=0.0)))
        assert np.array_equal(r0.w_grid, r1.w_grid, equal_nan=True)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["model"].update(theory="ktn_linear"), "model.theory"),
    (lambda d: d["model"].update(theory="ktn_full"), "model.theory"),
    (lambda d: d.update(contact={"enabled": True, "gap": 0.5}),
     "contact.enabled"),
    (lambda d: d.update(supports={"points": [[0.0, 0.0]], "stiffness": 1e3}),
     "supports.points"),
])
def test_thermal_guards(mutate, match):
    d = _case(mt=_MT)
    mutate(d)
    with pytest.raises(CaseError, match=match):
        dispatch.solve(Problem.from_dict(d))


def test_thermal_mixed_rejected():
    d = _case(mt=_MT, kind="rect")
    d["bc"] = {"type": "mixed", "sides": [
        {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
        {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]}
    with pytest.raises(CaseError, match="bc.type"):
        Problem.from_dict(d)


def test_thermal_non_uniform_rejected():
    """thermal_moment принят только у uniform (у прочих — неизвестный ключ)."""
    d = _case(mt=_MT)
    d["load"] = {"type": "gaussian", "q0": 4.0, "x0": 0.0, "y0": 0.0,
                 "sigma": 0.3, "thermal_moment": _MT}
    with pytest.raises(CaseError, match="load"):
        Problem.from_dict(d)


def test_thermal_eigen_prestress_rejected():
    d = _case(theory="karman", mt=_MT)
    d["eigen"] = {"kind": "vibration", "n_modes": 3, "prestress": True}
    with pytest.raises(CaseError, match="thermal_moment"):
        Problem.from_dict(d)


def test_thermal_orthotropy_rejected():
    d = _case(mt=_MT, kind="rect")
    d["model"] = {"theory": "classic", "h": 1.0,
                  "orthotropy": {"D11": 2.0, "D12": 0.5, "D22": 1.0,
                                 "D66": 0.4}}
    with pytest.raises(CaseError, match="orthotropy"):
        Problem.from_dict(d)
